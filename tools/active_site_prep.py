#!/usr/bin/env python
"""active_site_prep.py — preprocess a chimeric active-site PDB for TS-search.

Generalizable tool for chimeric / Frankenstein PDBs whose ligand atom names
differ from project conventions (e.g. YYL has P1/O1/O5 instead of the PTE
default P1/O3/O7). Computes per-atom chain/residue census, sums an estimated
formal charge, detects intermolecular clashes, and emits a cleaned PDB with
QCB REMARKs + Rosetta enzyme-matcher REMARK 666 anchors.

Usage:
    active_site_prep.py INPUT.pdb \
        --out preprocessed.pdb \
        --substrate-resname YYL \
        --p-name P1 --nuc-name O1 --lg-name O5 \
        --target-charge -2 \
        --clash-cutoff 1.8 \
        --clash-report clashes.txt \
        --json-summary summary.json

Notes
-----
* Atom *renaming* is intentionally NEVER performed — downstream tools accept
  --p-name/--nuc-name/--lg-name overrides.
* Charge calculation is best-effort heuristic; report any discrepancy with
  --target-charge but do not modify the structure to match.
* The substrate is taken from the residue named via --substrate-resname; the
  enzyme matcher REMARK 666 lines are emitted for active-site residues (those
  declared via --motif-resids or --motif-distance).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# Heuristic per-residue formal charges at neutral pH for typical ionic protein
# residues with default (mainly carboxylate/ammonium/imidazole) protonation.
NEUTRAL_PH_RESIDUE_CHARGES: dict[str, int] = {
    # negatively charged
    "ASP": -1, "GLU": -1,
    # positively charged
    "LYS": +1, "ARG": +1,
    "HIP": +1,                         # doubly protonated histidine
    # zwitter / typically neutral residues all map to 0 by default; HIS
    # default state is neutral (HID/HIE)
    "HIS": 0, "HID": 0, "HIE": 0,
    "TYR": 0, "CYS": 0,                # ionizable but mostly neutral at pH 7
    "ALA": 0, "VAL": 0, "LEU": 0, "ILE": 0, "MET": 0,
    "PHE": 0, "TRP": 0, "PRO": 0,
    "GLY": 0, "SER": 0, "THR": 0, "ASN": 0, "GLN": 0,
    # Special PTE / metalloenzyme:
    "KCX": -1,                         # carboxylated lysine (deprotonated carbamate)
    "OCS": -1, "CSO": -1, "SEP": -1,   # phosphoserine / sulfocysteine etc., often -1
    # waters & solvent
    "HOH": 0, "WAT": 0, "H2O": 0,
}

# Metal residue charges (residue name == element symbol).
METAL_CHARGES: dict[str, int] = {
    "ZN":  +2, "MG": +2, "CA": +2, "MN": +2, "FE": +2,
    "FE2": +2, "FE3": +3, "CU": +2, "NI": +2, "CO": +2,
    "K":   +1, "NA": +1,
    "OH":  -1, "OHX": -1, "HYD": -1,                 # hydroxide naming variants
    "CL":  -1, "BR": -1, "I":  -1, "F":  -1,
}

PROTEIN_RES = set(NEUTRAL_PH_RESIDUE_CHARGES) | {"HIE", "HID", "HIP"}
LIGAND_LIKE_DEFAULTS = {"YYL", "YZW", "YYE", "XUW", "SUB", "LIG", "MOL"}

# Optional override: residue-name -> formal charge for known substrates whose
# bridging vs non-bridging O bookkeeping is fixed (e.g. YYL has one
# non-bridging P=O carrying -1 + protonated O1 nucleophile carrying 0 = -1
# overall in our PTE TS convention). Override per-call via --substrate-charge.
KNOWN_SUBSTRATE_CHARGES: dict[str, int] = {
    "YYL": -1,    # paraoxon-derived TS; P=O3 carries -1 (one non-bridging O)
    "YZW": -1,
    "YYE": -1,
    "XUW": -1,
}


def configure_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("active_site_prep")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Input PDB file")
    p.add_argument("--out", type=Path, default=None,
                   help="Output cleaned PDB (default: <input>.clean.pdb)")
    p.add_argument("--substrate-resname", default="YYL",
                   help="Residue name of the substrate ligand (default: YYL)")
    p.add_argument("--p-name", default="P1",
                   help="Atom name of the electrophilic phosphorus (default: P1)")
    p.add_argument("--nuc-name", default="O1",
                   help="Atom name of the nucleophile (default: O1)")
    p.add_argument("--lg-name", default="O5",
                   help="Atom name of the leaving group (default: O5)")
    p.add_argument("--target-charge", type=int, default=None,
                   help="Expected net charge for the system; reported but not enforced.")
    p.add_argument("--substrate-charge", type=int, default=None,
                   help="Formal charge attributed to the substrate ligand "
                        "(default: lookup in KNOWN_SUBSTRATE_CHARGES, else 0).")
    p.add_argument("--clash-cutoff", type=float, default=1.8,
                   help="Distance below which to report inter-residue clashes (Å, "
                        "default 1.8). Tightening below 1.4 may yield true bonds; "
                        "loosening above 2.4 will report H-bonds.")
    p.add_argument("--clash-report", type=Path, default=None,
                   help="Write per-clash report TSV to this path.")
    p.add_argument("--json-summary", type=Path, default=None,
                   help="Write JSON summary of charge ledger + clashes here.")
    p.add_argument("--motif-resids", nargs="+", type=int, default=None,
                   help="Specific chain-A residue IDs to include in REMARK 666 "
                        "matcher anchors. Default: all chain-A non-water "
                        "non-metal residues.")
    p.add_argument("--motif-chain", default="A",
                   help="Chain ID containing the active-site motif (default: A).")
    p.add_argument("--substrate-chain", default="Z",
                   help="Chain ID containing the substrate (default: Z).")
    p.add_argument("--add-remark-666", action="store_true",
                   help="Add Rosetta REMARK 666 matcher anchor lines.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


# ────────────────────────────── PDB parsing ───────────────────────────────
class PdbAtom:
    __slots__ = ("raw", "serial", "name", "altloc", "resname", "chain",
                 "resseq", "icode", "x", "y", "z", "occ", "bfac",
                 "element", "charge_field", "is_hetatm")

    def __init__(self, line: str):
        self.raw = line.rstrip("\n")
        self.is_hetatm = line.startswith("HETATM")
        self.serial = int(line[6:11])
        self.name = line[12:16].strip()
        self.altloc = line[16:17]
        self.resname = line[17:20].strip()
        self.chain = line[21:22]
        self.resseq = int(line[22:26])
        self.icode = line[26:27]
        self.x = float(line[30:38])
        self.y = float(line[38:46])
        self.z = float(line[46:54])
        self.occ = float(line[54:60]) if line[54:60].strip() else 1.0
        self.bfac = float(line[60:66]) if line[60:66].strip() else 0.0
        self.element = line[76:78].strip() if len(line) >= 78 else ""
        self.charge_field = line[78:80].strip() if len(line) >= 80 else ""

    @property
    def coord(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def res_key(self) -> tuple[str, str, int]:
        return (self.chain, self.resname, self.resseq)

    @property
    def res_kw(self) -> str:
        return f"{self.chain}-{self.resname}{self.resseq}"


def read_pdb(path: Path) -> tuple[list[PdbAtom], list[str], list[str]]:
    """Return (atoms, header_remarks, other_pre_atom_lines).

    header_remarks: REMARK lines we will rewrite/replace
    other_pre_atom_lines: HEADER, CRYST1, etc. we should preserve verbatim.
    """
    atoms: list[PdbAtom] = []
    remarks: list[str] = []
    other: list[str] = []
    seen_atoms = False
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            atoms.append(PdbAtom(line))
            seen_atoms = True
        elif line.startswith("REMARK") and not seen_atoms:
            remarks.append(line)
        elif not seen_atoms and line.strip():
            if line.startswith(("CRYST1", "HEADER", "TITLE", "SCALE",
                                "ORIGX", "MASTER", "EXPDTA", "MODEL", "ENDMDL")):
                other.append(line)
    return atoms, remarks, other


# ───────────────────────────── Charge ledger ──────────────────────────────
def estimate_total_charge(
    atoms: list[PdbAtom],
    substrate_resname: str,
    log: logging.Logger,
    substrate_charge: int | None = None,
) -> tuple[int, dict]:
    """Return (estimated_charge, breakdown) summing per-residue defaults."""
    breakdown: dict[str, int] = {}
    seen_residues: set[tuple] = set()
    metal_charge_total = 0
    metal_count: dict[str, int] = {}
    protein_charge_total = 0
    ligand_residues: list[str] = []
    unknown_residues: dict[str, int] = {}
    ligand_total_q = 0

    if substrate_charge is None:
        substrate_charge = KNOWN_SUBSTRATE_CHARGES.get(substrate_resname.upper(), 0)
        log.info(f"Substrate {substrate_resname!r} default charge: "
                 f"{substrate_charge:+d} ({'KNOWN_SUBSTRATE_CHARGES' if substrate_resname.upper() in KNOWN_SUBSTRATE_CHARGES else 'fallback 0'})")

    for a in atoms:
        key = a.res_key
        if key in seen_residues:
            continue
        seen_residues.add(key)
        rn = a.resname.upper()

        if rn in METAL_CHARGES:
            q = METAL_CHARGES[rn]
            metal_charge_total += q
            metal_count[rn] = metal_count.get(rn, 0) + 1
            breakdown[f"metal:{a.res_kw}"] = q
            continue

        if rn == substrate_resname:
            ligand_residues.append(a.res_kw)
            breakdown[f"ligand:{a.res_kw}"] = substrate_charge
            ligand_total_q += substrate_charge
            continue

        if rn in NEUTRAL_PH_RESIDUE_CHARGES:
            q = NEUTRAL_PH_RESIDUE_CHARGES[rn]
            protein_charge_total += q
            if q != 0:
                breakdown[f"residue:{a.res_kw}"] = q
            continue

        # Unknown residue
        unknown_residues[rn] = unknown_residues.get(rn, 0) + 1

    total = metal_charge_total + protein_charge_total + ligand_total_q
    summary = {
        "metal_total_q": metal_charge_total,
        "metal_count": metal_count,
        "protein_total_q": protein_charge_total,
        "ligand_total_q": ligand_total_q,
        "ligand_residues": ligand_residues,
        "ligand_charge_each": substrate_charge,
        "unknown_residues": unknown_residues,
        "estimated_total_q": total,
        "breakdown": breakdown,
        "residues_total": len(seen_residues),
    }
    if unknown_residues:
        log.warning(f"Unknown residues with no default charge mapping: "
                    f"{sorted(unknown_residues.items())} — assumed neutral.")
    return total, summary


# ────────────────────────────── Clash analysis ────────────────────────────
def find_clashes(
    atoms: list[PdbAtom],
    cutoff: float,
    log: logging.Logger,
    skip_peptide_bonds: bool = True,
    skip_zn_coord: bool = True,
    classify_hbond: bool = True,
) -> list[dict]:
    """Return list of {a_idx, b_idx, dist, a_label, b_label, kind} for clashes.

    Filters out:
    - intra-residue contacts
    - peptide bond C(i)-N(i+1) (~1.32 Å, expected)
    - Zn coordination contacts (Zn-O, Zn-N within 2.5 Å, expected)

    Annotates each remaining contact with kind:
    - "h_bond"           : H–{O,N,S} within 1.4–2.2 Å — expected hydrogen bond
    - "h_proton_transfer": H–{O,N} below 1.2 Å — likely proton-transferred / shared proton
    - "h_clash"          : H–H or H–C (non donor-acceptor) below 1.4 Å — true clash
    - "heavy"            : two non-H atoms close — true heavy-atom clash
    """
    from scipy.spatial import cKDTree
    coords = np.array([a.coord for a in atoms], dtype=float)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=cutoff)
    clashes: list[dict] = []
    for i, j in pairs:
        a, b = atoms[i], atoms[j]
        if (a.chain, a.resseq) == (b.chain, b.resseq):
            continue
        if skip_peptide_bonds and a.chain == b.chain:
            sequential = abs(a.resseq - b.resseq) == 1
            backbone_pair = ({a.name, b.name} == {"C", "N"})
            if sequential and backbone_pair:
                continue
        if skip_zn_coord:
            elems = {a.element.upper(), b.element.upper()}
            if "ZN" in elems and any(e in {"O", "N", "S"} for e in elems if e != "ZN"):
                continue
        d = float(np.linalg.norm(coords[i] - coords[j]))
        # Classify
        elems = {a.element.upper(), b.element.upper()}
        kind = "heavy"
        if classify_hbond:
            has_donor = elems & {"O", "N", "S"}
            if "H" in elems and has_donor:
                if d < 1.2:
                    kind = "h_proton_transfer"  # H<->O or H<->N at covalent distance
                elif 1.4 <= d <= 2.4:
                    kind = "h_bond"
                else:
                    kind = "h_clash"
            elif "H" in elems and d < 1.4:
                kind = "h_clash"
        clashes.append({
            "a_idx": i, "b_idx": j, "dist": d,
            "a_label": f"{a.res_kw}/{a.name}({a.element})",
            "b_label": f"{b.res_kw}/{b.name}({b.element})",
            "a_res": a.res_kw, "b_res": b.res_kw, "kind": kind,
        })
    clashes.sort(key=lambda c: c["dist"])
    by_kind = {}
    for c in clashes:
        by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
    log.info(f"{len(clashes)} inter-residue contacts under {cutoff:.2f} Å "
             f"(peptide bonds + Zn-coord excluded). By kind: {by_kind}")
    return clashes


# ─────────────────────── Reactive-atom inspection ─────────────────────────
def find_reactive_distances(
    atoms: list[PdbAtom],
    substrate_resname: str,
    p_name: str,
    nuc_name: str,
    lg_name: str,
    log: logging.Logger,
) -> dict:
    """Locate the substrate's P/Nuc/LG atoms and report distances + Zn neighbors."""
    sub = [a for a in atoms if a.resname == substrate_resname]
    if not sub:
        raise ValueError(f"No atoms in residue {substrate_resname!r} found")
    by_name = {a.name: a for a in sub}
    if p_name not in by_name:
        raise KeyError(f"P atom {p_name!r} not found in {substrate_resname}")
    if nuc_name not in by_name:
        raise KeyError(f"Nuc atom {nuc_name!r} not found in {substrate_resname}")
    if lg_name not in by_name:
        raise KeyError(f"LG atom {lg_name!r} not found in {substrate_resname}")
    P, Onuc, Olg = by_name[p_name], by_name[nuc_name], by_name[lg_name]

    Pc = np.array(P.coord)
    d_P_Onuc = float(np.linalg.norm(Pc - np.array(Onuc.coord)))
    d_P_Olg = float(np.linalg.norm(Pc - np.array(Olg.coord)))

    # Other O's bonded to P: anything within 1.8 Å of P in the same residue
    other_O = []
    for a in sub:
        if a.name in {p_name, nuc_name, lg_name}:
            continue
        if a.element != "O":
            continue
        d = float(np.linalg.norm(np.array(a.coord) - Pc))
        if d < 1.9:
            other_O.append({"name": a.name, "dist_from_P": d})

    # Find metal coordinations to nucleophile/LG
    zn_atoms = [a for a in atoms
                if a.resname.upper() in METAL_CHARGES and METAL_CHARGES[a.resname.upper()] >= 2]
    nuc_zn = []
    for zn in zn_atoms:
        d = float(np.linalg.norm(np.array(zn.coord) - np.array(Onuc.coord)))
        if d < 3.0:
            nuc_zn.append({"zn": zn.res_kw, "dist": d})
    info = {
        "P": {"name": P.name, "res": P.res_kw, "idx": atoms.index(P), "coord": list(P.coord)},
        "Nuc": {"name": Onuc.name, "res": Onuc.res_kw, "idx": atoms.index(Onuc),
                "coord": list(Onuc.coord)},
        "LG": {"name": Olg.name, "res": Olg.res_kw, "idx": atoms.index(Olg),
               "coord": list(Olg.coord)},
        "d_P_Nuc": d_P_Onuc,
        "d_P_LG": d_P_Olg,
        "other_P_O_bonds": other_O,
        "Nuc_zinc_coord": nuc_zn,
    }
    log.info(f"P-Nuc distance: {d_P_Onuc:.3f} Å   P-LG distance: {d_P_Olg:.3f} Å")
    if nuc_zn:
        zn_str = ", ".join(f"{z['zn']}={z['dist']:.2f}Å" for z in nuc_zn)
        log.info(f"Nucleophile coordinated to: {zn_str}")
    return info


# ─────────────────────────── REMARK assembly ──────────────────────────────
def build_remark_header(
    *,
    target_charge: int | None,
    estimated_charge: int,
    substrate_resname: str,
    p_name: str, nuc_name: str, lg_name: str,
    motif_residues: list[tuple[str, str, int]],
    substrate_residue: tuple[str, str, int] | None,
    reactive_info: dict,
    autodetect_notes: str,
) -> list[str]:
    """Return the list of REMARK lines to prepend to the cleaned PDB."""
    lines: list[str] = []
    # REMARK 665 legend
    lines.append("REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors")
    lines.append("REMARK 665 REMARK QCB <NNN> <LABEL> ... = quantum_cowboy_biochemistry lineage")

    # Standard QCB lineage
    if target_charge is not None:
        lines.append(f"REMARK QCB 001 TOTAL_CHARGE value={target_charge:+d}")
    else:
        lines.append(f"REMARK QCB 001 TOTAL_CHARGE value={estimated_charge:+d} (estimated)")
    lines.append(f"REMARK QCB 011 LIGAND_RENAME substrate_resname={substrate_resname}")
    lines.append(
        f"REMARK QCB 005 REACTIVE_DISTANCES "
        f"P={p_name} Nuc={nuc_name} LG={lg_name} "
        f"d_P_Nuc={reactive_info['d_P_Nuc']:.3f} "
        f"d_P_LG={reactive_info['d_P_LG']:.3f}"
    )
    lines.append(
        f"REMARK QCB 020 AUTODETECT notes={autodetect_notes!r}"
    )

    # REMARK 666 matcher anchors (Rosetta enzyme-matcher format)
    if substrate_residue is not None:
        sub_chain, sub_resn, sub_resi = substrate_residue
        for k, (mch, mrn, mri) in enumerate(motif_residues, start=1):
            lines.append(
                f"REMARK 666 MATCH TEMPLATE {sub_chain} {sub_resn:>4s} {sub_resi:>4d} "
                f"MATCH MOTIF {mch} {mrn:>3s} {mri:>4d} {k:>3d}   1"
            )
    return lines


def emit_clean_pdb(
    out_path: Path,
    remark_header: list[str],
    other_pre_atom_lines: list[str],
    atoms: list[PdbAtom],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pieces = []
    # Preserve HEADER/CRYST1/etc
    if other_pre_atom_lines:
        # HEADER first if present
        head = [l for l in other_pre_atom_lines if l.startswith("HEADER")]
        rest = [l for l in other_pre_atom_lines if not l.startswith("HEADER")]
        pieces.extend(head)
    pieces.extend(remark_header)
    if other_pre_atom_lines:
        pieces.extend([l for l in other_pre_atom_lines if not l.startswith("HEADER")])

    # Preserve atoms with TER between chain transitions
    last_chain = None
    last_was_protein = False
    for a in atoms:
        is_protein = a.resname in PROTEIN_RES and not a.is_hetatm
        if last_chain is not None and (a.chain != last_chain) and last_was_protein:
            pieces.append("TER")
        pieces.append(a.raw)
        last_chain = a.chain
        last_was_protein = is_protein
    pieces.append("END")
    out_path.write_text("\n".join(pieces) + "\n")


# ─────────────────────────────── main ─────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = configure_logging(args.log_level)

    if args.out is None:
        args.out = args.input.with_suffix(".clean.pdb")
    if args.clash_report is None:
        args.clash_report = args.out.with_suffix(".clash_report.tsv")
    if args.json_summary is None:
        args.json_summary = args.out.with_suffix(".summary.json")

    log.info(f"Reading {args.input}")
    atoms, _orig_remarks, other = read_pdb(args.input)
    log.info(f"Total atoms: {len(atoms)}")

    # Census
    chains = sorted({a.chain for a in atoms})
    res_per_chain = {}
    for ch in chains:
        res_per_chain[ch] = sorted({(a.resname, a.resseq) for a in atoms if a.chain == ch},
                                   key=lambda kv: kv[1])
    for ch, rs in res_per_chain.items():
        log.info(f"  Chain {ch!r}: {len(rs)} residues")

    # Charge ledger
    est_q, q_breakdown = estimate_total_charge(
        atoms, args.substrate_resname, log,
        substrate_charge=args.substrate_charge)
    log.info(f"Estimated net charge: {est_q:+d}")
    if args.target_charge is not None:
        diff = est_q - args.target_charge
        if diff != 0:
            log.warning(f"Estimate ({est_q:+d}) ≠ target ({args.target_charge:+d}); diff={diff:+d}")

    # Reactive triplet
    reactive_info = find_reactive_distances(
        atoms, args.substrate_resname, args.p_name, args.nuc_name, args.lg_name, log)

    # Substrate residue identity for REMARK 666 matcher
    sub_atoms = [a for a in atoms if a.resname == args.substrate_resname]
    substrate_residue = None
    if sub_atoms:
        substrate_residue = (sub_atoms[0].chain, sub_atoms[0].resname, sub_atoms[0].resseq)

    # Motif residues for REMARK 666
    motif: list[tuple[str, str, int]] = []
    if args.add_remark_666:
        chain_a_atoms = [a for a in atoms if a.chain == args.motif_chain]
        seen = set()
        for a in chain_a_atoms:
            key = (a.chain, a.resname, a.resseq)
            if key in seen:
                continue
            seen.add(key)
            if a.resname.upper() in METAL_CHARGES and len(a.resname) <= 3:
                continue
            if a.resname in {"HOH", "WAT", "H2O"}:
                continue
            if args.motif_resids and a.resseq not in args.motif_resids:
                continue
            motif.append(key)

    # Clashes
    clashes = find_clashes(atoms, args.clash_cutoff, log)
    near = sum(1 for c in clashes if c["dist"] < 1.5)
    log.info(f"  severe (<1.5 Å): {near}")

    # Write reports
    args.clash_report.parent.mkdir(parents=True, exist_ok=True)
    with args.clash_report.open("w") as fh:
        fh.write("# inter-residue contacts under cutoff\n")
        fh.write("# columns: dist_A\tatom_a\tatom_b\n")
        for c in clashes:
            fh.write(f"{c['dist']:.3f}\t{c['a_label']}\t{c['b_label']}\n")
    log.info(f"Clash report written: {args.clash_report}")

    summary = {
        "input": str(args.input),
        "out": str(args.out),
        "n_atoms": len(atoms),
        "chains": chains,
        "n_residues_per_chain": {c: len(r) for c, r in res_per_chain.items()},
        "charge": {
            "estimated": est_q,
            "target": args.target_charge,
            "breakdown": q_breakdown,
        },
        "reactive": reactive_info,
        "clashes": {
            "cutoff_A": args.clash_cutoff,
            "n_total": len(clashes),
            "n_severe_lt_1.5A": near,
            "top10": clashes[:10],
        },
        "motif_residues": motif,
        "substrate_residue": substrate_residue,
    }
    args.json_summary.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"JSON summary written: {args.json_summary}")

    # REMARK header
    autodetect_notes = (
        f"input atom names use {args.substrate_resname}.{args.p_name}/{args.nuc_name}/{args.lg_name} "
        f"(not the project default). Downstream tools must receive --p-name/--nuc-name/--lg-name."
    )
    remarks = build_remark_header(
        target_charge=args.target_charge,
        estimated_charge=est_q,
        substrate_resname=args.substrate_resname,
        p_name=args.p_name, nuc_name=args.nuc_name, lg_name=args.lg_name,
        motif_residues=motif if args.add_remark_666 else [],
        substrate_residue=substrate_residue,
        reactive_info=reactive_info,
        autodetect_notes=autodetect_notes,
    )

    emit_clean_pdb(args.out, remarks, other, atoms)
    log.info(f"Cleaned PDB written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
