#!/usr/bin/env python3
"""
Protonate a protein active site cluster for QM/MLFF calculations.

Strategy:
  1. reduce (Richardson lab) — adds hydrogens with clash-aware placement,
     handles HIS flip optimization, respects metal coordination
  2. propka — predicts pKa of ionizable residues for the given pH
  3. Custom post-processing — neutral termini, user overrides, charge calculation

Only ATOM records are protonated; HETATM (ligand) is left untouched.
Net charge = sum of protein residue charges + user-specified ligand charge.

Usage:
  python protonate_active_site.py input.pdb -o output.pdb --ligand-charge 3 --pH 7.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("protonate")

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

REDUCE_BIN = "/net/software/utils/reduce"

# Standard amino acid formal charges at neutral pH
# Positive: LYS (+1), ARG (+1)
# Negative: ASP (-1), GLU (-1)
# Neutral: HIS (0) by default — protonation depends on pKa & environment
# CYS: 0 by default (can be -1 if deprotonated, but rare in active sites)
STANDARD_CHARGES = {
    "ALA": 0, "ARG": 1, "ASN": 0, "ASP": -1, "CYS": 0,
    "GLN": 0, "GLU": -1, "GLY": 0, "HIS": 0, "ILE": 0,
    "LEU": 0, "LYS": 1, "MET": 0, "PHE": 0, "PRO": 0,
    "SER": 0, "THR": 0, "TRP": 0, "TYR": 0, "VAL": 0,
    # Modified residues
    "KCX": 0,   # Carboxylated lysine: NH charge (+1) + COO (-1) = 0
    "HIP": 1,   # Doubly-protonated histidine
    "HID": 0,   # ND1-protonated histidine
    "HIE": 0,   # NE2-protonated histidine
}


# ══════════════════════════════════════════════════════════════
#  STEP 1: REDUCE — add hydrogens to protein ATOM records
# ══════════════════════════════════════════════════════════════


def run_reduce(input_pdb: Path, output_pdb: Path, flip: bool = True) -> str:
    """Run reduce to add hydrogens. Only modifies ATOM records.

    Args:
        input_pdb: Input PDB path
        output_pdb: Output PDB path (with H added)
        flip: Whether to optimize HIS/ASN/GLN orientations (-FLIP)

    Returns:
        reduce stdout (contains diagnostic info about H placement)
    """
    if not os.path.isfile(REDUCE_BIN):
        raise FileNotFoundError(f"reduce not found at {REDUCE_BIN}")

    # Build reduce command — only add H, don't modify existing
    # -NOFLIP or -FLIP for sidechain optimization
    # -Quiet suppresses most output
    # -ALLALT keeps all alt conformations
    args = [REDUCE_BIN, "-Quiet"]
    if flip:
        args.append("-FLIP")
    else:
        args.append("-NOFLIP")
    args.append(str(input_pdb))

    result = subprocess.run(args, capture_output=True, text=True, timeout=60)

    # reduce writes the protonated PDB to stdout
    pdb_output = result.stdout
    diagnostics = result.stderr

    if not pdb_output.strip():
        raise RuntimeError(f"reduce produced no output. stderr:\n{diagnostics}")

    # Write output, preserving HETATM lines from original
    _write_reduce_output(input_pdb, pdb_output, output_pdb)

    return diagnostics


def _write_reduce_output(original_pdb: Path, reduce_output: str, output_pdb: Path):
    """Merge reduce's ATOM output with original HETATM and header lines.

    reduce modifies ATOM records but may mangle HETATM. We keep the original
    HETATM lines and header/remarks, and take only ATOM lines from reduce.
    """
    # Parse original for HETATM and non-coordinate lines
    original_lines = Path(original_pdb).read_text().splitlines()
    header_lines = [l for l in original_lines if not l.startswith(("ATOM", "HETATM", "TER", "END", "CONECT"))]
    hetatm_lines = [l for l in original_lines if l.startswith("HETATM")]
    ter_after_hetatm = any(
        l.startswith("TER") for l in original_lines
        if original_lines.index(l) > max(
            (i for i, x in enumerate(original_lines) if x.startswith("HETATM")), default=-1
        )
    )

    # Parse reduce output for ATOM lines (including new H atoms)
    reduce_lines = reduce_output.splitlines()
    atom_lines = [l for l in reduce_lines if l.startswith("ATOM")]
    # Also grab USER MOD lines (reduce diagnostics)
    user_lines = [l for l in reduce_lines if l.startswith("USER")]

    # Assemble output
    out = []
    out.extend(header_lines)
    out.extend(user_lines)
    out.extend(atom_lines)
    out.append("TER")
    out.extend(hetatm_lines)
    if ter_after_hetatm or hetatm_lines:
        out.append("TER")
    out.append("END")

    Path(output_pdb).write_text("\n".join(out) + "\n")


# ══════════════════════════════════════════════════════════════
#  STEP 1b: PARSE REDUCE DIAGNOSTICS — metal coordination
# ══════════════════════════════════════════════════════════════


def parse_reduce_metal_coordination(protonated_pdb: Path) -> dict:
    """Parse reduce's USER MOD lines to identify metal-coordinating residues.

    reduce marks residues whose H atoms would clash with metals as "NoAdj-H".
    For example:
      USER  MOD NoAdj-H: A   1 HIS HE2 : A   1 HIS NE2 : Z   9 YYE ZN1 :(H bumps)

    This means HIS A:1 NE2 coordinates to ZN1 and should NOT be protonated there.
    A HIS coordinating a metal via NE2 cannot be doubly-protonated (HIP), so it
    should be forced to neutral (0) regardless of what propka says.

    Returns:
        Dict of {(chain, resnum): {'resn': str, 'atom': str, 'metal': str, 'reason': str}}
    """
    metal_coord = {}

    with open(protonated_pdb) as f:
        for line in f:
            if "NoAdj-H" not in line:
                continue

            # Parse: "USER  MOD NoAdj-H: A   1 HIS HE2 : A   1 HIS NE2 : Z   9 YYE ZN1 :(H bumps)"
            parts = line.split(":")
            if len(parts) < 4:
                continue

            # The residue info is in the second segment (the coordinating atom)
            coord_part = parts[2].strip()  # "A   1 HIS NE2"
            metal_part = parts[3].strip()  # "Z   9 YYE ZN1"

            # Parse residue
            tokens = coord_part.split()
            if len(tokens) >= 4:
                chain = tokens[0]
                try:
                    resnum = int(tokens[1])
                except ValueError:
                    continue
                resn = tokens[2]
                coord_atom = tokens[3]

                # Parse metal
                metal_tokens = metal_part.split()
                metal_name = metal_tokens[3] if len(metal_tokens) >= 4 else "metal"

                key = (chain, resnum)
                metal_coord[key] = {
                    "resn": resn,
                    "coord_atom": coord_atom,
                    "metal": metal_name,
                    "reason": f"NE2/ND1 coordinates {metal_name} (reduce NoAdj-H)",
                }
                log.info(
                    f"  Metal coordination: {resn} {chain}:{resnum} "
                    f"{coord_atom} → {metal_name} (cannot be doubly-protonated)"
                )

    return metal_coord


# ══════════════════════════════════════════════════════════════
#  STEP 2: PROPKA — predict pKa values
# ══════════════════════════════════════════════════════════════


def run_propka(pdb_path: Path, pH: float = 7.0) -> dict:
    """Run propka to predict pKa of ionizable residues.

    Returns dict of {(chain_id, res_num): {'pKa': float, 'resn': str, 'protonated': bool}}
    """
    try:
        import propka.input
        import propka.lib
        import propka.molecular_container
        import propka.parameters
    except ImportError:
        log.warning("propka not available — using default protonation states")
        return {}

    pdb_path = Path(pdb_path).resolve()

    try:
        propka_opts = propka.lib.loadOptions([str(pdb_path)])
        propka_params = propka.input.read_parameter_file(
            propka_opts.parameters, propka.parameters.Parameters()
        )
        propka_molc = propka.molecular_container.MolecularContainer(propka_params, propka_opts)
        propka.input.read_molecule_file(propka_opts.input_pdb, propka_molc)
        propka_molc.calculate_pka()
    except Exception as e:
        log.warning(f"propka failed: {e} — using default protonation states")
        return {}

    resis_dict = {}
    for group in propka_molc.conformations["AVR"].groups:
        pka_value = group.pka_value
        resis_dict[(group.atom.chain_id, group.atom.res_num)] = {
            "pKa": pka_value,
            "resn": group.atom.res_name,
            "protonated": (pka_value >= pH) if pka_value is not None else None,
        }

    return resis_dict


# ══════════════════════════════════════════════════════════════
#  STEP 3: POST-PROCESSING — termini, overrides, charge calc
# ══════════════════════════════════════════════════════════════


def determine_residue_charges(
    pdb_path: Path,
    pH: float = 7.0,
    neutral_termini: bool = True,
    user_overrides: dict | None = None,
    pka_dict: dict | None = None,
    metal_coord: dict | None = None,
) -> dict:
    """Determine the formal charge of each protein residue.

    Args:
        pdb_path: PDB file to analyze
        pH: pH for protonation state decisions
        neutral_termini: If True, terminal amines/carboxyls are neutral (capped model)
        user_overrides: Dict of {(chain, resnum): charge} for manual overrides
        pka_dict: Pre-computed propka results (or None to run propka)
        metal_coord: Dict from parse_reduce_metal_coordination() — residues
                     coordinating metals are forced to neutral charge

    Returns:
        Dict of {(chain, resnum): {'resn': str, 'charge': int, 'reason': str}}
    """
    if pka_dict is None:
        pka_dict = run_propka(pdb_path, pH=pH)

    # Parse residues from PDB
    residues = {}
    terminal_n = set()  # (chain, resnum) of N-terminal residues
    terminal_c = set()  # (chain, resnum) of C-terminal residues

    with open(pdb_path) as f:
        chain_residues = {}  # {chain: [resnum, ...]}
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            resnum = int(line[22:26])
            resn = line[17:20].strip()
            key = (chain, resnum)
            if key not in residues:
                residues[key] = resn
            if chain not in chain_residues:
                chain_residues[chain] = []
            if resnum not in chain_residues[chain]:
                chain_residues[chain].append(resnum)

    # Identify termini
    for chain, resnums in chain_residues.items():
        resnums_sorted = sorted(resnums)
        if resnums_sorted:
            terminal_n.add((chain, resnums_sorted[0]))
            terminal_c.add((chain, resnums_sorted[-1]))

    # Assign charges
    charges = {}
    for key, resn in residues.items():
        chain, resnum = key

        # Check user override first
        if user_overrides and key in user_overrides:
            charges[key] = {
                "resn": resn,
                "charge": user_overrides[key],
                "reason": "user override",
            }
            continue

        # Start with standard charge
        base_charge = STANDARD_CHARGES.get(resn, 0)
        reason = "standard"

        # Check propka for pKa-shifted protonation
        if key in pka_dict:
            pka_info = pka_dict[key]
            pka_val = pka_info["pKa"]

            if resn in ("ASP", "GLU"):
                # Normally deprotonated (charge -1). Protonated if pKa > pH
                if pka_val is not None and pka_val > pH:
                    base_charge = 0  # protonated = neutral
                    reason = f"propka pKa={pka_val:.1f} > pH={pH} → protonated (neutral)"
                else:
                    reason = f"propka pKa={pka_val:.1f} → deprotonated (-1)"

            elif resn == "HIS":
                # Normally neutral. Doubly-protonated (+1) if pKa > pH
                if pka_val is not None and pka_val > pH:
                    base_charge = 1  # HIP = +1
                    reason = f"propka pKa={pka_val:.1f} > pH={pH} → doubly protonated (+1)"
                else:
                    reason = f"propka pKa={pka_val:.1f} → singly protonated (0)"

            elif resn == "LYS":
                # Normally protonated (+1). Deprotonated (0) if pKa < pH
                if pka_val is not None and pka_val < pH:
                    base_charge = 0
                    reason = f"propka pKa={pka_val:.1f} < pH={pH} → deprotonated (0)"
                else:
                    reason = f"propka pKa={pka_val:.1f} → protonated (+1)"

            elif resn == "CYS":
                # Normally protonated (0). Deprotonated (-1) if pKa < pH
                if pka_val is not None and pka_val < pH:
                    base_charge = -1
                    reason = f"propka pKa={pka_val:.1f} < pH={pH} → deprotonated (-1)"

        # Metal-coordination override: if reduce says this residue coordinates
        # a metal, it CANNOT be doubly-protonated. Force to neutral.
        # This overrides propka's pKa prediction for metal-coordinating HIS.
        if metal_coord and key in metal_coord:
            mc = metal_coord[key]
            if resn == "HIS" and base_charge == 1:
                base_charge = 0
                reason = (
                    f"METAL OVERRIDE: {mc['coord_atom']} coordinates {mc['metal']} "
                    f"→ cannot be doubly-protonated (propka pKa ignored)"
                )
            elif resn in ("CYS",) and base_charge == 0:
                # CYS coordinating metal is typically deprotonated (thiolate)
                base_charge = -1
                reason = (
                    f"METAL OVERRIDE: coordinates {mc['metal']} "
                    f"→ deprotonated thiolate (-1)"
                )
            elif resn in ("ASP", "GLU") and base_charge == 0:
                # ASP/GLU coordinating metal: keep as deprotonated (carboxylate)
                base_charge = -1
                reason = (
                    f"METAL OVERRIDE: coordinates {mc['metal']} "
                    f"→ deprotonated carboxylate (-1)"
                )

        # Handle KCX (carboxylated lysine)
        if resn == "KCX":
            base_charge = 0  # NH group (+1) + carboxylate (-1) = 0
            reason = "KCX: carbamylated lysine (NH+ + COO- = 0)"

        # Handle termini — make neutral for capped/cropped models
        if neutral_termini:
            if key in terminal_n:
                # N-terminus: don't add +1 for NH3+ (treat as neutral NH2)
                if resn in ("LYS", "ARG"):
                    pass  # sidechain charge already counted
                reason += " | N-terminus (neutral)"
            if key in terminal_c:
                # C-terminus: don't add -1 for COO- (treat as neutral COOH)
                reason += " | C-terminus (neutral)"

        charges[key] = {
            "resn": resn,
            "charge": base_charge,
            "reason": reason,
        }

    return charges


def calculate_total_charge(
    residue_charges: dict,
    ligand_charge: int = 0,
) -> tuple[int, dict]:
    """Calculate total system charge.

    Returns (total_charge, breakdown_dict)
    """
    protein_charge = sum(v["charge"] for v in residue_charges.values())
    total = protein_charge + ligand_charge

    breakdown = {
        "protein_charge": protein_charge,
        "ligand_charge": ligand_charge,
        "total_charge": total,
        "residue_charges": {
            f"{k[0]}:{k[1]}:{v['resn']}": v["charge"]
            for k, v in residue_charges.items()
            if v["charge"] != 0
        },
    }

    return total, breakdown


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════


def protonate_active_site(
    input_pdb: str | Path,
    output_pdb: str | Path,
    pH: float = 7.0,
    ligand_charge: int = 0,
    neutral_termini: bool = True,
    user_overrides: dict | None = None,
    flip: bool = True,
) -> dict:
    """Protonate a protein active site cluster.

    Args:
        input_pdb: Input PDB file
        output_pdb: Output protonated PDB file
        pH: pH for protonation state decisions
        ligand_charge: Net charge of HETATM atoms (user-specified)
        neutral_termini: Treat termini as neutral (for cropped models)
        user_overrides: {(chain, resnum): charge} for manual protonation control
        flip: Optimize HIS/ASN/GLN sidechain orientations

    Returns:
        Summary dict with charges, pKa values, diagnostics
    """
    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)

    log.info(f"Input: {input_pdb}")
    log.info(f"Output: {output_pdb}")
    log.info(f"pH: {pH}, ligand charge: {ligand_charge:+d}")

    # Step 1: Add hydrogens with reduce
    log.info("Step 1: Running reduce (hydrogen addition) ...")
    reduce_diag = run_reduce(input_pdb, output_pdb, flip=flip)

    # Parse reduce diagnostics
    h_added = 0
    for line in reduce_diag.splitlines():
        m = re.search(r"found=(\d+), std=(\d+), add=(\d+)", line)
        if m:
            h_added = int(m.group(3))
    log.info(f"  Hydrogens added: {h_added}")

    # Step 1b: Parse reduce's metal coordination diagnostics
    log.info("Step 1b: Parsing metal coordination from reduce ...")
    metal_coord = parse_reduce_metal_coordination(output_pdb)
    if metal_coord:
        log.info(f"  Found {len(metal_coord)} metal-coordinating residue(s)")
    else:
        log.info("  No metal coordination detected")

    # Step 2: propka pKa prediction
    log.info("Step 2: Running propka (pKa prediction) ...")
    pka_dict = run_propka(output_pdb, pH=pH)
    if pka_dict:
        log.info(f"  Predicted pKa for {len(pka_dict)} groups")
        for key, info in sorted(pka_dict.items()):
            if info["resn"] in ("HIS", "ASP", "GLU", "LYS", "CYS"):
                state = "protonated" if info["protonated"] else "deprotonated"
                coord_tag = " [METAL-COORD]" if key in metal_coord else ""
                log.info(f"    {info['resn']} {key[0]}:{key[1]} pKa={info['pKa']:.1f} → {state}{coord_tag}")

    # Step 3: Determine charges (with metal-coordination override)
    log.info("Step 3: Determining residue charges ...")
    residue_charges = determine_residue_charges(
        output_pdb, pH=pH, neutral_termini=neutral_termini,
        user_overrides=user_overrides, pka_dict=pka_dict,
        metal_coord=metal_coord,
    )

    for key, info in sorted(residue_charges.items()):
        if info["charge"] != 0:
            log.info(f"  {info['resn']} {key[0]}:{key[1]}: charge={info['charge']:+d} ({info['reason']})")

    # Step 4: Total charge
    total_charge, breakdown = calculate_total_charge(residue_charges, ligand_charge)

    log.info("")
    log.info("=" * 50)
    log.info(f"  Protein charge:  {breakdown['protein_charge']:+d}")
    log.info(f"  Ligand charge:   {breakdown['ligand_charge']:+d}")
    log.info(f"  TOTAL CHARGE:    {total_charge:+d}")
    log.info("=" * 50)

    # Write charge info as REMARK in the output PDB
    _add_charge_remarks(output_pdb, total_charge, breakdown)

    summary = {
        "input_pdb": str(input_pdb),
        "output_pdb": str(output_pdb),
        "pH": pH,
        "total_charge": total_charge,
        "protein_charge": breakdown["protein_charge"],
        "ligand_charge": ligand_charge,
        "pka_predictions": {
            f"{v['resn']}_{k[0]}:{k[1]}": v["pKa"]
            for k, v in pka_dict.items()
            if v["resn"] in ("HIS", "ASP", "GLU", "LYS", "CYS")
        },
        "charged_residues": breakdown["residue_charges"],
        "h_added": h_added,
    }

    return summary


def _add_charge_remarks(pdb_path: Path, total_charge: int, breakdown: dict):
    """Add REMARK lines with charge information to the PDB."""
    lines = Path(pdb_path).read_text().splitlines()

    # Find where to insert (after existing REMARK lines)
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith(("REMARK", "HEADER", "USER")):
            insert_idx = i + 1
        else:
            break

    charge_remarks = [
        f"REMARK QCB TOTAL_CHARGE {total_charge:+d}",
        f"REMARK QCB PROTEIN_CHARGE {breakdown['protein_charge']:+d}",
        f"REMARK QCB LIGAND_CHARGE {breakdown['ligand_charge']:+d}",
    ]
    for key, charge in breakdown["residue_charges"].items():
        charge_remarks.append(f"REMARK QCB CHARGED_RES {key} {charge:+d}")

    for i, remark in enumerate(charge_remarks):
        lines.insert(insert_idx + i, remark)

    Path(pdb_path).write_text("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(
        description="Protonate a protein active site cluster for QM/MLFF calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python protonate_active_site.py input.pdb -o protonated.pdb --ligand-charge 3
  python protonate_active_site.py input.pdb -o protonated.pdb --ligand-charge 3 --pH 8.0
  python protonate_active_site.py input.pdb -o protonated.pdb --ligand-charge 3 --override A:1:+1 C:3:0
        """,
    )
    p.add_argument("input_pdb", help="Input PDB file (protein active site cluster)")
    p.add_argument("-o", "--output", required=True, help="Output protonated PDB file")
    p.add_argument("--ligand-charge", type=int, default=0,
                   help="Net formal charge of HETATM atoms (default: 0)")
    p.add_argument("--pH", type=float, default=7.0, help="pH for protonation (default: 7.0)")
    p.add_argument("--no-flip", action="store_true",
                   help="Don't optimize HIS/ASN/GLN orientations (faster)")
    p.add_argument("--charged-termini", action="store_true",
                   help="Treat termini as charged (NH3+/COO-) instead of neutral")
    p.add_argument("--override", nargs="*", default=[],
                   help="Manual charge overrides: CHAIN:RESNUM:CHARGE (e.g. A:1:+1 C:3:0)")
    p.add_argument("--summary-json", default=None,
                   help="Write summary JSON to this path")

    args = p.parse_args()

    # Parse user overrides
    user_overrides = {}
    for ov in args.override:
        parts = ov.split(":")
        if len(parts) != 3:
            print(f"ERROR: Override format is CHAIN:RESNUM:CHARGE, got: {ov}")
            sys.exit(1)
        chain, resnum, charge = parts[0], int(parts[1]), int(parts[2])
        user_overrides[(chain, resnum)] = charge

    summary = protonate_active_site(
        args.input_pdb,
        args.output,
        pH=args.pH,
        ligand_charge=args.ligand_charge,
        neutral_termini=not args.charged_termini,
        user_overrides=user_overrides if user_overrides else None,
        flip=not args.no_flip,
    )

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Summary written to {args.summary_json}")


if __name__ == "__main__":
    main()
