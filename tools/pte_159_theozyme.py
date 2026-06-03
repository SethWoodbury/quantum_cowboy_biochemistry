"""Build a properly-cropped, properly-protonated theozyme for M-CSA 159
(phosphotriesterase, PDB 1hzy).

This is the focused, single-purpose script for the PTE step-by-step
validation push. The previous mcsa_theozyme harness was too generic —
it kept all chain B atoms (homodimer), all crystallographic Zn (4 of
4), Na+/EDO/FMT/PEL ligands, and never protonated.

Decisions made here, made explicit:
  * **Single chain** — keep chain A only. 1hzy is a homodimer; chain B
    is a symmetry copy with its own active site we don't need.
  * **Catalytic residues** — the 8 residues M-CSA 159 lists, all on
    chain A: HIS 55, 57, 201, 230, 254 (5 metal-ligating histidines);
    ASP 233 and 301 (proton-shuttle / metal ligand); LYS 169 (the PTM
    carbamylated bridge — but the deposited PDB has it as plain LYS,
    so we ALSO need to add the carbamate).
  * **Cofactor metals** — exactly the 2 chain-A Zn (Zn401 and Zn402,
    3.44 Å apart, the binuclear site). Drop:
      - chain B Zn (homodimer)
      - Na+ (crystallographic, not catalytic)
  * **No other HETATMs** — drop EDO (cryoprotectant), FMT (formate),
    PEL (ligands from co-crystal), all on both chains. None are
    catalytic.
  * **Waters** — excluded by default. They can be included explicitly
    for diagnostics with ``--include-waters``; otherwise the bridging
    hydroxide/nucleophile should be built as a deliberate mechanistic
    species rather than inherited from crystallographic waters.
  * **Carbamylation of LYS 169** — append the missing CX, OQ1, OQ2
    carbamate atoms to LYS 169's NZ, then rename the residue from
    LYS → KCX. Use literature carbamate geometry (NZ–CX 1.40 Å,
    CX=O 1.25 Å, NZ–CX–O 120°). Place the new atoms so CX bridges
    the two Zn ions.

The old consensus protonation path can still be run explicitly with
``--run-consensus-protonation`` for debugging, but it is not the default:
template-based H placement is not metal-aware enough for PTE.

Usage:
    python tools/pte_159_theozyme.py --outdir runs/PTE_159
"""
from __future__ import annotations

import argparse
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CATALYTIC_RESIDUES: list[tuple[str, str, int]] = [
    # (resname, chain, auth_seq) — 8 residues from M-CSA 159
    ("HIS", "A", 55),
    ("HIS", "A", 57),
    ("HIS", "A", 201),
    ("HIS", "A", 230),
    ("HIS", "A", 254),
    ("ASP", "A", 233),
    ("ASP", "A", 301),
    ("LYS", "A", 169),    # the PTM — will be carbamylated to KCX
]

# Zn binuclear pair on chain A (we filter by chain match + distance check;
# the seqnums here are from 1hzy specifically).
ZN_CHAIN_A_SEQS = [401, 402]


@dataclass
class AtomLine:
    """A parsed PDB ATOM/HETATM line; we keep the original raw line so
    we can write it back verbatim (preserving B-factors, occupancies,
    element columns, etc.) when we filter."""
    raw: str
    record: str          # "ATOM" or "HETATM"
    name: str            # atom name (cols 13-16)
    resname: str         # cols 18-20
    chain: str
    seq: int
    x: float
    y: float
    z: float

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.seq, self.resname)


def _parse_atom(line: str) -> AtomLine | None:
    rec = line[:6].strip()
    if rec not in ("ATOM", "HETATM"):
        return None
    try:
        return AtomLine(
            raw=line,
            record=rec,
            name=line[12:16].strip(),
            resname=line[17:20].strip(),
            chain=line[21],
            seq=int(line[22:26]),
            x=float(line[30:38]),
            y=float(line[38:46]),
            z=float(line[46:54]),
        )
    except (ValueError, IndexError):
        return None


def _dist(a: AtomLine, x: float, y: float, z: float) -> float:
    return math.sqrt((a.x-x)**2 + (a.y-y)**2 + (a.z-z)**2)


def crop_active_site(
    pdb_path: Path,
    out_path: Path,
    *,
    include_waters: bool = False,
    water_shell_A: float = 4.5,
    log: logging.Logger,
) -> dict:
    """Crop a single PTE active site from chain A. Returns a summary
    dict for the user to review.

    Algorithm (transparent on purpose):
      1. Parse all ATOM/HETATM lines.
      2. Keep ATOM lines on chain A whose (resname, seq) is in the 8
         catalytic residues list.
      3. Keep HETATM ZN lines on chain A whose seq is in ZN_CHAIN_A_SEQS
         AND that are within 7 Å of any catalytic residue heavy atom
         (sanity check; chain A Zn401/Zn402 are ~4 Å from active-site
         residues in 1hzy).
      4. Discard ALL other HETATMs except waters (HOH/WAT).
      5. By default, discard all crystallographic waters. If requested,
         keep waters whose O is within ``water_shell_A`` Å of any kept
         atom (catalytic residue or kept Zn).
      6. Write a clean PDB with HEADER + filtered lines + END.

    Reports:
      * which residues + atom counts were kept
      * which Zn made the cut + their distance to the catalytic centre
      * how many waters survived the shell filter
      * what was discarded and why
    """
    catalytic_set = {(c, s, r) for r, c, s in CATALYTIC_RESIDUES}
    kept: list[AtomLine] = []
    dropped_counts: dict[str, int] = {}

    # First pass: collect catalytic-residue + Zn atoms
    catalytic_atoms: list[AtomLine] = []
    zn_candidates: list[AtomLine] = []
    waters: list[AtomLine] = []
    other_hetatms: list[AtomLine] = []

    with pdb_path.open() as fh:
        for line in fh:
            a = _parse_atom(line)
            if a is None:
                continue
            if a.record == "ATOM":
                if (a.chain, a.seq, a.resname) in catalytic_set:
                    catalytic_atoms.append(a)
                else:
                    dropped_counts["non-catalytic-ATOM"] = (
                        dropped_counts.get("non-catalytic-ATOM", 0) + 1)
            else:  # HETATM
                if a.resname in ("HOH", "WAT"):
                    waters.append(a)
                elif a.resname == "ZN" and a.chain == "A" and a.seq in ZN_CHAIN_A_SEQS:
                    zn_candidates.append(a)
                else:
                    other_hetatms.append(a)
                    key = f"HETATM-{a.resname}-{a.chain}"
                    dropped_counts[key] = dropped_counts.get(key, 0) + 1

    # Sanity-check Zn: each chain-A Zn must be <7 Å of some catalytic atom
    kept_zn: list[AtomLine] = []
    zn_distances: list[tuple[int, float]] = []
    for zn in zn_candidates:
        d_min = min(_dist(c, zn.x, zn.y, zn.z) for c in catalytic_atoms)
        zn_distances.append((zn.seq, d_min))
        if d_min < 7.0:
            kept_zn.append(zn)
            log.info(f"  KEEP Zn A {zn.seq} (closest catalytic atom {d_min:.2f} Å)")
        else:
            dropped_counts["ZN-too-far"] = dropped_counts.get("ZN-too-far", 0) + 1
            log.warning(f"  DROP Zn A {zn.seq} (too far: {d_min:.2f} Å)")

    # Filter waters by shell distance from catalytic + Zn
    anchor = catalytic_atoms + kept_zn
    kept_waters: list[AtomLine] = []
    if include_waters:
        for w in waters:
            if w.name not in ("O", "OW"):
                continue
            d_min = min((_dist(a, w.x, w.y, w.z) for a in anchor), default=1e9)
            if d_min <= water_shell_A:
                kept_waters.append(w)

    # Final ordered list — atom serial renumbered for cleanliness
    ordered = catalytic_atoms + kept_zn + kept_waters

    # Build the output PDB
    lines_out: list[str] = []
    lines_out.append(f"HEADER  PTE M-CSA 159 cropped active site (chain A only)\n")
    lines_out.append(f"REMARK  Source: {pdb_path.name}\n")
    lines_out.append(f"REMARK  Catalytic residues: {len(catalytic_atoms)} atoms\n")
    lines_out.append(f"REMARK  Chain-A Zn pair:    {len(kept_zn)} atoms\n")
    if include_waters:
        lines_out.append(
            f"REMARK  Shell waters (<={water_shell_A} A): {len(kept_waters)} atoms\n"
        )
    else:
        lines_out.append("REMARK  Shell waters: excluded by default\n")
    lines_out.append(f"REMARK  Discarded: see crop_log.txt\n")
    for serial, a in enumerate(ordered, start=1):
        # Replace cols 7-11 (atom serial) with the new index, keep
        # everything else verbatim
        new_line = f"{a.record:<6}{serial:>5}" + a.raw[11:]
        lines_out.append(new_line)
    lines_out.append("END\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines_out))

    return {
        "n_catalytic_atoms": len(catalytic_atoms),
        "n_zn_kept": len(kept_zn),
        "zn_distances_to_catalytic": zn_distances,
        "n_waters_kept": len(kept_waters),
        "n_waters_total_chain_AB": len(waters),
        "dropped_counts": dropped_counts,
        "output_pdb": str(out_path),
    }


def add_carbamate_to_lys169(in_path: Path, out_path: Path,
                            log: logging.Logger) -> dict:
    """Convert LYS A 169 → KCX A 169 by appending the carbamate atoms
    (CX, OQ1, OQ2) to NZ. Carbamate geometry: NZ–CX = 1.40 Å, CX=O =
    1.25 Å, NZ–CX–O angles tetrahedral-ish (using ~120°).

    The carbamate is positioned so CX sits between the two Zn ions
    (the catalytic geometry — NZ–CX bridges the metals). We compute
    the midpoint of the two Zn, project onto a unit vector from NZ
    toward that midpoint, and place CX 1.40 Å along that direction.
    """
    # Find NZ of LYS A 169 + the two Zn positions
    nz: tuple[float, float, float] | None = None
    zn_positions: list[tuple[float, float, float]] = []
    lines = in_path.read_text().splitlines(keepends=True)

    for line in lines:
        a = _parse_atom(line)
        if a is None:
            continue
        if a.record == "ATOM" and a.resname == "LYS" and a.chain == "A" \
                and a.seq == 169 and a.name == "NZ":
            nz = (a.x, a.y, a.z)
        elif a.record == "HETATM" and a.resname == "ZN" and a.chain == "A":
            zn_positions.append((a.x, a.y, a.z))

    if nz is None:
        raise RuntimeError("LYS A 169 NZ not found in input PDB")
    if len(zn_positions) != 2:
        raise RuntimeError(
            f"Expected 2 chain-A Zn but found {len(zn_positions)}")

    # Midpoint of the two Zn — where the bridging CX should sit
    mx = sum(z[0] for z in zn_positions) / 2
    my = sum(z[1] for z in zn_positions) / 2
    mz = sum(z[2] for z in zn_positions) / 2

    # Unit vector from NZ toward Zn-midpoint
    dx, dy, dz = mx - nz[0], my - nz[1], mz - nz[2]
    mag = math.sqrt(dx*dx + dy*dy + dz*dz)
    ux, uy, uz = dx/mag, dy/mag, dz/mag

    # Place CX at NZ + 1.40 Å along that direction
    cx_pos = (nz[0] + 1.40 * ux, nz[1] + 1.40 * uy, nz[2] + 1.40 * uz)

    # Place OQ1 and OQ2 — perpendicular bisector trick. We pick a
    # vector orthogonal to NZ→CX and place the two oxygens symmetrically
    # at ~120° from NZ–CX axis, 1.25 Å from CX.
    # Reference perpendicular: cross(z_axis, NZ→CX) — pick whichever's not parallel
    z_ax = (0.0, 0.0, 1.0)
    perp = (uy*z_ax[2] - uz*z_ax[1],
            uz*z_ax[0] - ux*z_ax[2],
            ux*z_ax[1] - uy*z_ax[0])
    pmag = math.sqrt(sum(c*c for c in perp))
    if pmag < 0.1:
        # NZ→CX nearly parallel to z; use x-axis as fallback
        x_ax = (1.0, 0.0, 0.0)
        perp = (uy*x_ax[2] - uz*x_ax[1],
                uz*x_ax[0] - ux*x_ax[2],
                ux*x_ax[1] - uy*x_ax[0])
        pmag = math.sqrt(sum(c*c for c in perp))
    perp = tuple(c / pmag for c in perp)

    # Carbamate has trigonal planar geometry around CX. Place each O
    # at 1.25 Å from CX, one on +perp and one on −perp, with a
    # backwards component along −u (so O is on the opposite side from NZ).
    # Trigonal angle: 120° from N–C–O.
    angle = math.radians(120)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    # Direction from CX to O = cos(120°)*(-u) + sin(120°)*(±perp)
    o1_dir = (-cos_a*ux + sin_a*perp[0],
              -cos_a*uy + sin_a*perp[1],
              -cos_a*uz + sin_a*perp[2])
    o2_dir = (-cos_a*ux - sin_a*perp[0],
              -cos_a*uy - sin_a*perp[1],
              -cos_a*uz - sin_a*perp[2])
    o1_pos = (cx_pos[0] + 1.25*o1_dir[0],
              cx_pos[1] + 1.25*o1_dir[1],
              cx_pos[2] + 1.25*o1_dir[2])
    o2_pos = (cx_pos[0] + 1.25*o2_dir[0],
              cx_pos[1] + 1.25*o2_dir[1],
              cx_pos[2] + 1.25*o2_dir[2])

    log.info(f"  KCX-build: NZ at {nz}, CX at {cx_pos}, "
             f"OQ1 at {o1_pos}, OQ2 at {o2_pos}")

    # Build the new lines: rename LYS A 169 → KCX A 169 on every atom
    # of that residue, and append the 3 new HETATM-style lines after
    # the last LYS A 169 atom.
    out_lines: list[str] = []
    inserted = False
    last_lys169_idx = -1

    # First find the last LYS A 169 line so we know where to insert
    for i, line in enumerate(lines):
        a = _parse_atom(line)
        if a and a.resname == "LYS" and a.chain == "A" and a.seq == 169:
            last_lys169_idx = i

    # Rewrite: rename LYS→KCX on all chain-A 169 atoms, then insert
    # CX/OQ1/OQ2 after the last LYS atom
    for i, line in enumerate(lines):
        a = _parse_atom(line)
        if a and a.resname == "LYS" and a.chain == "A" and a.seq == 169:
            # rename: cols 18-20 are the residue name. Replace "LYS" with "KCX".
            new_line = line[:17] + "KCX" + line[20:]
            out_lines.append(new_line)
            if i == last_lys169_idx:
                # Insert the carbamate atoms. Serial numbers are normalized
                # across the whole PDB before writing.
                # Use ATOM record (not HETATM) since KCX is a modified
                # standard residue (some tools complain otherwise).
                out_lines.append(_format_atom(
                    0, "CX", "KCX", "A", 169, *cx_pos, "C"))
                out_lines.append(_format_atom(
                    0, "OQ1", "KCX", "A", 169, *o1_pos, "O"))
                out_lines.append(_format_atom(
                    0, "OQ2", "KCX", "A", 169, *o2_pos, "O"))
                inserted = True
        else:
            out_lines.append(line)

    if not inserted:
        raise RuntimeError("Failed to insert KCX carbamate atoms")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(_renumber_atom_serials(out_lines)))
    return {
        "nz_xyz": nz,
        "cx_xyz": cx_pos,
        "oq1_xyz": o1_pos,
        "oq2_xyz": o2_pos,
        "zn_midpoint": (mx, my, mz),
        "output_pdb": str(out_path),
    }


def _renumber_atom_serials(lines: list[str]) -> list[str]:
    """Return PDB lines with contiguous ATOM/HETATM serial numbers."""
    out: list[str] = []
    serial = 1
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            out.append(f"{line[:6]}{serial:>5}{line[11:]}")
            serial += 1
        else:
            out.append(line)
    return out


def _format_atom(serial: int, name: str, resname: str, chain: str,
                 seq: int, x: float, y: float, z: float, element: str) -> str:
    """Format a PDB ATOM line. Column layout (1-based):
        1-6:   record (ATOM  /HETATM)
        7-11:  serial (right-just)
        12:    space
        13-16: atom name (4 chars; lead-pad single-element names)
        17:    altLoc (space)
        18-20: resname (right-just)
        21:    space
        22:    chainID
        23-26: resSeq (right-just)
        27:    insertion code (space)
        28-30: spaces
        31-38: x (8.3f)
        39-46: y
        47-54: z
        55-60: occupancy (6.2f)
        61-66: tempFactor
        67-76: spaces
        77-78: element (right-just)
    """
    # Atom names: single-element atoms (C, N, O, H, S) lead with a space
    # ("  CA" not "CA  "). Multi-char names (CB, CG1, OQ1) also lead with
    # a space if ≤3 chars; only 4-char names occupy all 4 columns.
    if len(name) >= 4:
        name_padded = name[:4]
    else:
        name_padded = f" {name:<3}"
    return (
        f"ATOM  {serial:>5} "
        f"{name_padded} "        # name(4) + altLoc space
        f"{resname:>3} {chain}"
        f"{seq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"  1.00 20.00          "
        f"{element:>2}\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("runs/PTE_159"))
    p.add_argument("--pdb", type=Path,
                   default=Path("runs/PTE_159/1hzy.pdb"),
                   help="Source PDB (1hzy)")
    p.add_argument("--include-waters", action="store_true",
                   help="Include crystallographic waters within --water-shell-A")
    p.add_argument("--water-shell-A", type=float, default=4.5,
                   help="Water shell radius around catalytic + Zn when "
                        "--include-waters is set")
    p.add_argument("--run-consensus-protonation", action="store_true",
                   help="Run the old template/consensus protonation path. "
                        "Not recommended for PTE metal sites.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("pte_159_theozyme")

    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.pdb.is_file():
        log.error(f"Source PDB not found: {args.pdb}")
        log.error("Run: zcat /net/databases/rcsb/pdb/hz/pdb1hzy.ent.gz "
                  f"> {args.pdb}")
        return 1

    log.info(f"=== PTE 159 theozyme build ===")
    log.info(f"  source: {args.pdb}")
    log.info(f"  outdir: {args.outdir}")

    # Step 1: crop chain A active site
    log.info("--- Step 1: crop chain A active site ---")
    cropped = args.outdir / "step1_cropped.pdb"
    crop_summary = crop_active_site(
        args.pdb,
        cropped,
        include_waters=args.include_waters,
        water_shell_A=args.water_shell_A,
        log=log,
    )
    log.info(f"  → {crop_summary['output_pdb']}")
    log.info(f"  catalytic atoms:  {crop_summary['n_catalytic_atoms']}")
    log.info(f"  Zn kept:          {crop_summary['n_zn_kept']}")
    log.info(f"  Zn distances:     {crop_summary['zn_distances_to_catalytic']}")
    log.info(f"  waters kept:      {crop_summary['n_waters_kept']}")
    log.info(f"  total waters in 1hzy A+B: {crop_summary['n_waters_total_chain_AB']}")
    log.info(f"  dropped:")
    for k, v in sorted(crop_summary["dropped_counts"].items()):
        log.info(f"    {k}: {v}")

    # Step 2: add carbamate to LYS 169
    log.info("--- Step 2: carbamylate LYS A 169 → KCX ---")
    kcx_pdb = args.outdir / "step2_kcx_added.pdb"
    kcx_summary = add_carbamate_to_lys169(cropped, kcx_pdb, log)
    log.info(f"  → {kcx_summary['output_pdb']}")

    # Step 3: protonate via the canonical protonator (qcb protonate).
    # KCX 169 is the carbamylated-lysine PTM (charge -1); metals/ligand HETATM
    # are left untouched by the protonator.
    if args.run_consensus_protonation:
        log.info("--- Step 3: protonation (protonator) ---")
        final_pdb = args.outdir / "step3_protonated.pdb"
        try:
            from quantum_engine.prep import protonator
            rc = protonator.main([
                "--input-pdb", str(kcx_pdb),
                "--output-pdb", str(final_pdb),
                "--pH", "7.0",
                "--ptm", "A:169=KCX",
                "--ptm-charge", "A:169=-1",
            ])
            if rc != 0:
                raise RuntimeError(f"protonator returned {rc}")
            log.info(f"  → {final_pdb}")
        except Exception as e:
            log.error(f"  protonation failed: {type(e).__name__}: {e}")
            log.error("  shipping unprotonated KCX file as final.")
            final_pdb = kcx_pdb
    else:
        log.info("--- Step 3 SKIPPED (protonation rejected for PTE metal sites) ---")
        final_pdb = kcx_pdb

    # Final summary
    log.info("=" * 70)
    log.info("PTE 159 theozyme build complete.")
    log.info(f"  step 1 (cropped):        {cropped}")
    log.info(f"  step 2 (KCX-added):      {kcx_pdb}")
    log.info(f"  review structure:        {final_pdb}")
    log.info("")
    log.info("Open in PyMOL:")
    log.info(f"  pymol {final_pdb}")
    log.info("")
    log.info("INSPECT this output before continuing to substrate / vacuum TS.")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
