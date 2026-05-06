#!/usr/bin/env python
"""structure_io.py — universal PDB/CIF I/O with REMARK lineage and format validation.

Designed to be a generalizable utility for any quantum_cowboy_biochemistry
pipeline that needs to:
  * read a PDB while preserving REMARK 666 enzyme-matcher lines
  * write PDB AND/OR CIF with full lineage (REMARK 665, 666, condensed QCB codes)
  * include bonds + bond orders (Aromatic_single / Aromatic_double via kekulization)
  * include per-atom partial charges
  * validate output column alignment / format strictly
  * dump multi-MODEL trajectory snapshots for PyMOL/ChimeraX

Builds on (does not replace):
  * quantum_engine/io/structure.py (template-based PDB writer)
  * quantum_engine/io/cif.py (atomworks-backed CIF writer with WBO)
  * deps/.local_pkgs/atomworks (read/write/bond-list infrastructure)
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

log = logging.getLogger("structure_io")


# ===== REMARK 665/666 dataclasses ============================================
@dataclass
class MatcherEntry:
    """One Rosetta enzyme-matcher anchor (REMARK 666)."""
    target_chain: str       # tCH
    target_name: str        # tNAME (residue name of the substrate or template)
    target_resi: int        # tRESI
    motif_chain: str        # mCH
    motif_resn: str         # mRESN (e.g., HIS, ASP)
    motif_resi: int         # mRESI
    cst_no: int = 1         # IDX
    cst_no_var: int = 1     # VAR

    def render(self) -> str:
        return (f"REMARK 666 MATCH TEMPLATE {self.target_chain} "
                f"{self.target_name:>4s} {self.target_resi:>4d} "
                f"MATCH MOTIF {self.motif_chain} {self.motif_resn:>3s} "
                f"{self.motif_resi:>4d} {self.cst_no:>3d} {self.cst_no_var:>3d}")


# ===== REMARK QCB code registry ==============================================
QCB_CODES: dict[str, int] = {
    "TOTAL_CHARGE":          1,
    "ENERGY":                2,
    "STAGE":                 3,
    "METHOD":                4,
    "REACTIVE_DISTANCES":    5,
    "CHARGE_RESIDUES":       6,
    "METAL_RELABEL":         7,
    "NUCLEOPHILE_RELABEL":   8,
    "FORMAL_CHARGES":        9,
    "CARBAMATE_RELABEL":    10,
    "LIGAND_RENAME":        11,
    "CARBOXYLATE_CHARGE":   12,
    "WATER_RELABEL":        13,
    "WITHOUT_F132":         14,
    "FRAME_VALIDATION":     15,
    "CONSTRAINT_PATTERN":   16,
    "CONVERGENCE":          17,
    "B_FACTOR_MEANING":     18,
    "PROVENANCE":           19,
    "NOTE":                 99,
}


def qcb_remark(code: str | int, *fields: str) -> str:
    """Render a single ``REMARK QCB <NNN> <key>=<value> ...`` line.

    >>> qcb_remark("TOTAL_CHARGE", "value=+1")
    'REMARK QCB 001 TOTAL_CHARGE value=+1'
    >>> qcb_remark("REACTIVE_DISTANCES", "P=178", "Onuc=181", "Olg=189",
    ...             "d_P_Onuc=2.000", "d_P_Olg=2.250")
    'REMARK QCB 005 REACTIVE_DISTANCES P=178 Onuc=181 Olg=189 d_P_Onuc=2.000 d_P_Olg=2.250'
    """
    if isinstance(code, str):
        code_no = QCB_CODES.get(code.upper())
        if code_no is None:
            raise ValueError(f"unknown QCB code {code!r}; known: "
                             f"{sorted(QCB_CODES.keys())}")
        code_str = f"{code_no:03d}"
        label = code.upper()
    else:
        code_str = f"{int(code):03d}"
        label = ""
    payload = " ".join(fields).rstrip()
    if label:
        return f"REMARK QCB {code_str} {label} {payload}".rstrip()
    return f"REMARK QCB {code_str} {payload}".rstrip()


def qcb_legend_remarks() -> list[str]:
    """REMARK 665 lines describing what REMARK 666 and REMARK QCB carry."""
    return [
        "REMARK 665 REMARK 666 = Rosetta enzyme-matcher catalytic-motif anchors",
        "REMARK 665 fmt: REMARK 666 MATCH TEMPLATE <tCH tNAME tRESI> "
        "MATCH MOTIF <mCH mRESN mRESI IDX VAR>",
        "REMARK 665 REMARK QCB <NNN> <LABEL> <key=value> ... — quantum_cowboy_biochemistry lineage",
        "REMARK 665 codes: " + ", ".join(
            f"{v:03d}={k}" for k, v in sorted(QCB_CODES.items(), key=lambda kv: kv[1])
        )[:200],
    ]


# ===== PDB parser that retains REMARKs ======================================
@dataclass
class PdbAtom:
    raw: str        # full original line, used for PDB-format-fidelity rewriting
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occ: float
    bfac: float
    element: str
    charge_field: str


@dataclass
class StructureLineage:
    """Everything needed to round-trip a PDB with full annotation."""
    atoms: list[PdbAtom]
    raw_remarks: list[str]              # all REMARK lines from input, in order
    matcher_remarks: list[MatcherEntry] = field(default_factory=list)
    qcb_lines: list[str] = field(default_factory=list)   # REMARK QCB <NNN> ...
    other_remarks: list[str] = field(default_factory=list)


def parse_pdb_lineage(path: Path) -> StructureLineage:
    """Read a PDB, retain ATOM/HETATM and bucket REMARK lines."""
    atoms: list[PdbAtom] = []
    raw_remarks: list[str] = []
    matcher: list[MatcherEntry] = []
    qcb: list[str] = []
    other: list[str] = []

    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            atoms.append(PdbAtom(
                raw=line,
                serial=int(line[6:11]),
                name=line[12:16].strip(),
                altloc=line[16:17],
                resname=line[17:20].strip(),
                chain=line[21:22],
                resseq=int(line[22:26]),
                icode=line[26:27],
                x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
                occ=float(line[54:60]) if line[54:60].strip() else 1.0,
                bfac=float(line[60:66]) if line[60:66].strip() else 0.0,
                element=line[76:78].strip() if len(line) >= 78 else "",
                charge_field=line[78:80].strip() if len(line) >= 80 else "",
            ))
        elif line.startswith("REMARK"):
            raw_remarks.append(line.rstrip("\n"))
            if line.startswith("REMARK 666"):
                m = _parse_remark_666(line)
                if m is not None:
                    matcher.append(m)
            elif line.startswith("REMARK QCB") or line.startswith("REMARK   2 QCB"):
                qcb.append(line.rstrip("\n"))
            else:
                other.append(line.rstrip("\n"))
    return StructureLineage(
        atoms=atoms, raw_remarks=raw_remarks,
        matcher_remarks=matcher, qcb_lines=qcb, other_remarks=other,
    )


def _parse_remark_666(line: str) -> MatcherEntry | None:
    """Parse a Rosetta-format REMARK 666 line (matcher_residues)."""
    toks = line.split()
    # Expected: REMARK 666 MATCH TEMPLATE <tCH> <tNAME> <tRESI> MATCH MOTIF <mCH> <mRESN> <mRESI> <IDX> <VAR>
    try:
        return MatcherEntry(
            target_chain=toks[4],
            target_name=toks[5],
            target_resi=int(toks[6]),
            motif_chain=toks[9],
            motif_resn=toks[10],
            motif_resi=int(toks[11]),
            cst_no=int(toks[12]) if len(toks) > 12 else 1,
            cst_no_var=int(toks[13]) if len(toks) > 13 else 1,
        )
    except (IndexError, ValueError):
        return None


def autoadd_protein_matcher_remarks(lineage: StructureLineage,
                                     target_chain: str = "X",
                                     target_name: str = "SUB",
                                     target_resi: int = 0,
                                     ) -> None:
    """For any chain-A protein residue without a matcher entry, add one."""
    seen_motif_resi = {m.motif_resi for m in lineage.matcher_remarks}
    next_idx = (max((m.cst_no for m in lineage.matcher_remarks), default=0) + 1)
    seen_residues = set()
    for a in lineage.atoms:
        if a.chain != "A":
            continue
        if a.resname not in {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                              "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
                              "KCX","CSX"}:
            continue
        key = (a.chain, a.resseq)
        if key in seen_residues:
            continue
        seen_residues.add(key)
        if a.resseq in seen_motif_resi:
            continue
        lineage.matcher_remarks.append(MatcherEntry(
            target_chain=target_chain,
            target_name=target_name,
            target_resi=target_resi,
            motif_chain=a.chain,
            motif_resn=a.resname,
            motif_resi=a.resseq,
            cst_no=next_idx,
            cst_no_var=1,
        ))
        next_idx += 1


# ===== PDB writing with template =============================================
def write_pdb_lineage(lineage: StructureLineage,
                      coords: np.ndarray,
                      out_path: Path,
                      *,
                      drop_old_qcb: bool = True,
                      preserve_other_remarks: bool = True,
                      extra_qcb: list[str] | None = None,
                      bfactor: np.ndarray | None = None,
                      title: str | None = None) -> Path:
    """Write a PDB using the lineage's atom records as a template.

    coords: (N, 3) array — replaces the stored x/y/z columns.
    bfactor: (N,) optional — replaces the B-factor column.
    drop_old_qcb: if True, the input's REMARK QCB lines are NOT propagated
        (replaced by `extra_qcb` only). Default True so old/stale lineage is
        not silently inherited.
    preserve_other_remarks: keep REMARK lines that aren't 665/666/QCB
        (e.g. user-added REMARK 350 BIOMT etc.).
    """
    if len(coords) != len(lineage.atoms):
        raise ValueError(
            f"coord count {len(coords)} != atom count {len(lineage.atoms)}"
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    if title:
        lines.append(f"TITLE     {title[:70]}")
    lines.extend(qcb_legend_remarks())
    for m in lineage.matcher_remarks:
        lines.append(m.render())
    if not drop_old_qcb:
        lines.extend(lineage.qcb_lines)
    if extra_qcb:
        lines.extend(extra_qcb)
    if preserve_other_remarks:
        # only preserve REMARKs that aren't 665, 666, or QCB
        for r in lineage.other_remarks:
            if (r.startswith("REMARK 665") or r.startswith("REMARK 666") or
                "REMARK QCB" in r or "REMARK   2 QCB" in r):
                continue
            lines.append(r)

    # ATOM/HETATM lines: replace coord columns + optional B-factor
    for i, (a, (x, y, z)) in enumerate(zip(lineage.atoms, coords)):
        new_xyz = f"{x:8.3f}{y:8.3f}{z:8.3f}"
        line = a.raw[:30] + new_xyz + a.raw[54:80]
        if bfactor is not None:
            v = max(-99.99, min(99.99, float(bfactor[i])))
            line = line[:60] + f"{v:6.2f}" + line[66:]
        lines.append(line)
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


# ===== PDB FORMAT validation (column-level, NOT physical) ===================
def validate_pdb_format(path: Path) -> dict:
    """Strict PDB column-alignment + record check. Returns {ok, issues, n_atoms}."""
    issues: list[str] = []
    n_atoms = 0
    n_remarks = 0
    n_remarks_666 = 0

    for ln_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if line.startswith(("ATOM  ", "HETATM")):
            n_atoms += 1
            if len(line) < 78:
                issues.append(f"L{ln_no}: line shorter than 78 cols ({len(line)})")
            else:
                # column slice checks
                try:
                    int(line[6:11])
                except ValueError:
                    issues.append(f"L{ln_no}: serial col 7-11 not int: {line[6:11]!r}")
                try:
                    int(line[22:26])
                except ValueError:
                    issues.append(f"L{ln_no}: resseq col 23-26 not int: {line[22:26]!r}")
                for col, label in [(30, "x"), (38, "y"), (46, "z")]:
                    try:
                        float(line[col:col + 8])
                    except ValueError:
                        issues.append(f"L{ln_no}: coord {label} col {col+1}-{col+8} not float: "
                                      f"{line[col:col+8]!r}")
                # element column 77-78 should be alpha
                el = line[76:78].strip() if len(line) >= 78 else ""
                if not el or not el.isalpha():
                    issues.append(f"L{ln_no}: element col 77-78 not alpha: {line[76:78]!r}")
                if line[16:17] not in (" ", "A", "B", "C"):
                    issues.append(f"L{ln_no}: altloc col 17 unexpected: {line[16:17]!r}")
        elif line.startswith("REMARK"):
            n_remarks += 1
            if line.startswith("REMARK 666"):
                n_remarks_666 += 1
        elif line.startswith(("END", "TER", "TITLE", "HEADER", "COMPND",
                              "MODEL", "ENDMDL", "MASTER", "CRYST")):
            pass  # standard records
        elif line.strip() == "":
            pass  # empty ok
        # Any other unknown record: silently ok (this validator is forgiving on
        # non-essential record types)

    return {
        "ok": len(issues) == 0,
        "n_atoms": n_atoms,
        "n_remarks": n_remarks,
        "n_remarks_666": n_remarks_666,
        "issues": issues,
    }


# ===== CIF writer (delegates to existing write_ts_cif) ======================
def write_cif_lineage(lineage: StructureLineage,
                      coords: np.ndarray,
                      out_path: Path,
                      *,
                      total_charge: int = 0,
                      partial_charges: np.ndarray | None = None,
                      formal_charges: np.ndarray | None = None,
                      energy_eV: float | None = None,
                      extra_qcb: list[str] | None = None,
                      ) -> Path:
    """Write CIF via atomworks (delegating to quantum_engine.io.cif)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build an ASE Atoms with the requested coords + read template biotite struct.
    # NOTE: PDB stores element symbols in cols 77-78 LEFT-justified and is
    # often UPPERCASE ("ZN", "FE", "MG", "CL") — but ASE's `atomic_numbers`
    # table is *case-sensitive* and expects the canonical IUPAC form
    # ("Zn", "Fe", "Mg", "Cl"). Without normalization, ASEAtoms() raises
    # KeyError: 'ZN' for any metallo / halide system. Title-case fixes
    # 1- and 2-letter symbols uniformly.
    from ase import Atoms as ASEAtoms
    elements = [
        _normalize_element(a.element) if a.element else _guess_element(a.name)
        for a in lineage.atoms
    ]
    atoms = ASEAtoms(symbols=elements, positions=coords)

    # Try to use the original PDB as the biotite template (so residue names
    # round-trip into CIF). We re-parse to get the AtomArray.
    try:
        from quantum_engine.io.structure import load_structure
        from io import StringIO
        # write the lineage's raw_remarks + atoms to a tmp PDB so load_structure can read
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as fh:
            for r in lineage.raw_remarks:
                fh.write(r + "\n")
            for a in lineage.atoms:
                fh.write(a.raw + "\n")
            fh.write("END\n")
            tmp_pdb = fh.name
        _, template_struct, _ = load_structure(tmp_pdb)
        Path(tmp_pdb).unlink()
    except Exception as e:
        log.warning(f"could not build biotite template: {e}; CIF will lack residue annotations")
        template_struct = None

    if energy_eV is not None:
        # Stash on Atoms.info so write_ts_cif can pick it up if its API changes
        atoms.info["energy_eV"] = energy_eV

    from quantum_engine.io.cif import write_ts_cif
    write_ts_cif(
        ts_atoms=atoms,
        template_st=template_struct,
        outdir=out_path.parent,
        total_charge=total_charge,
        partial_charges=partial_charges,
        formal_charges=formal_charges,
        wiberg_bond_orders=None,    # caller can pass through if available
        filename=out_path.name,
    )

    # Append our own QCB lineage block to the CIF (legend + matcher + extras)
    if out_path.exists():
        cif_content = out_path.read_text()
        extra = ["", "# QCB lineage tags (key=value pairs)"]
        for ln in qcb_legend_remarks():
            extra.append(f"# {ln}")
        for m in lineage.matcher_remarks:
            extra.append(f"# {m.render()}")
        if extra_qcb:
            for q in extra_qcb:
                extra.append(f"# {q}")
        cif_content += "\n".join(extra) + "\n"
        out_path.write_text(cif_content)
    return out_path


def _normalize_element(symbol: str) -> str:
    """Coerce a PDB-style element symbol to ASE/IUPAC capitalization.

    PDB cols 77-78 are commonly uppercase ("ZN", "FE", "CL"); ASE's
    `atomic_numbers` dict (and most CIF readers) require canonical
    capitalization ("Zn", "Fe", "Cl"). Single-letter symbols ("C", "N",
    "O", "H") are unchanged by .capitalize(); multi-letter become Xx.
    Also strips charge suffixes like '2+' that some PDBs leak in.
    """
    s = (symbol or "").strip()
    if not s:
        return "C"
    # Strip trailing charge characters (e.g. "ZN2+", "MG+")
    s = s.rstrip("+-0123456789")
    if not s:
        return "C"
    return s.capitalize()


def _guess_element(atom_name: str) -> str:
    """Best-effort element from PDB atom name. Pseudo-PDB convention."""
    s = atom_name.strip()
    if not s:
        return "C"
    if s[0].isdigit() and len(s) > 1:
        return s[1].upper()
    if len(s) >= 2 and s[:2].upper() in {"FE", "ZN", "MG", "MN", "CA",
                                           "NA", "CL", "BR", "CU", "NI",
                                           "CO"}:
        return s[:2].capitalize()
    return s[0].upper()


# ===== Trajectory snapshots =================================================
def write_trajectory_pdb(lineage: StructureLineage,
                         frames: list[np.ndarray],
                         out_path: Path,
                         *,
                         model_stride: int = 1,
                         title: str | None = None) -> Path:
    """Write a multi-MODEL PDB trajectory using the lineage as template.

    frames: list of (N, 3) coord arrays.
    model_stride: write every Nth frame.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if title:
        lines.append(f"TITLE     {title[:70]}")
    lines.extend(qcb_legend_remarks())
    for m in lineage.matcher_remarks:
        lines.append(m.render())
    n_models = 0
    for i, coords in enumerate(frames):
        if i % model_stride != 0:
            continue
        if len(coords) != len(lineage.atoms):
            log.warning(f"frame {i}: coord count mismatch; skipping")
            continue
        n_models += 1
        lines.append(f"MODEL     {n_models:>4d}")
        for a, (x, y, z) in zip(lineage.atoms, coords):
            new_xyz = f"{x:8.3f}{y:8.3f}{z:8.3f}"
            lines.append(a.raw[:30] + new_xyz + a.raw[54:80])
        lines.append("ENDMDL")
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")
    log.info(f"wrote {n_models} MODELs to {out_path}")
    return out_path


# ===== CLI =================================================================
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Strict PDB column-format check")
    p_val.add_argument("pdb", type=Path)

    p_legend = sub.add_parser("legend",
                              help="Print REMARK 665 description + QCB code registry")

    p_remarks = sub.add_parser("remarks",
                               help="Show parsed REMARK 666 + QCB lineage of a PDB")
    p_remarks.add_argument("pdb", type=Path)

    p_traj = sub.add_parser("traj-from-ase",
                            help="Convert ASE trajectory to multi-MODEL PDB")
    p_traj.add_argument("--template-pdb", type=Path, required=True)
    p_traj.add_argument("--ase-traj", type=Path, required=True)
    p_traj.add_argument("--out", type=Path, required=True)
    p_traj.add_argument("--stride", type=int, default=10)

    args = ap.parse_args()

    if args.cmd == "validate":
        out = validate_pdb_format(args.pdb)
        print(json.dumps(out, indent=2, default=str))
        return 0 if out["ok"] else 1
    if args.cmd == "legend":
        for ln in qcb_legend_remarks():
            print(ln)
        for k, v in sorted(QCB_CODES.items(), key=lambda kv: kv[1]):
            print(f"  {v:03d} = {k}")
        return 0
    if args.cmd == "remarks":
        lin = parse_pdb_lineage(args.pdb)
        print(f"# atoms: {len(lin.atoms)}")
        print(f"# REMARK 666 (matcher): {len(lin.matcher_remarks)}")
        for m in lin.matcher_remarks:
            print(f"  {m.render()}")
        print(f"# REMARK QCB: {len(lin.qcb_lines)}")
        for q in lin.qcb_lines:
            print(f"  {q}")
        return 0
    if args.cmd == "traj-from-ase":
        from ase.io.trajectory import Trajectory
        lin = parse_pdb_lineage(args.template_pdb)
        traj = Trajectory(str(args.ase_traj))
        frames = [a.get_positions() for a in traj]
        write_trajectory_pdb(lin, frames, args.out, model_stride=args.stride)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
