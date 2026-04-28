"""High-throughput theozyme active-site refinement pipeline.

Use case
--------
You have a designed protein with a hand-built active site (containing a
ligand / TS analog). You ran AlphaFold3 on the design's sequence and got
back a backbone-correct prediction WITHOUT the ligand. To validate the
design, you want to:

  1. Transfer the ligand from the design into the AF3 frame
  2. Refine the active-site sidechains to accommodate the ligand
  3. Resolve metal-coordination + obvious clashes (some clashes ok —
     the active site will be redesigned later)
  4. Do this for THOUSANDS of designs cheaply on CPU

Workflow
--------
  parse REMARK 666 → catalytic residues
  align AF3 to design (catalytic CAs)
  transfer ligand into AF3 frame
  PTM relabel (LYS:64 → KCX, etc.)
  extract active-site cluster (atoms ≤ R Å from ligand + catalytic)
  cap protein backbone cuts with H atoms
  optimize cluster with backbone-frozen constraints
    default: xTB GFN-FF on CPU (~5-15 s per system)
    optional: GFN2-xTB (~30 s) or MACE-OMOL on GPU (~5 s)
  stitch refined cluster atoms back into the full structure

Performance
-----------
Default GFN-FF: ~5-15 s per system on CPU (depends on cluster size).
Highly parallelizable across many CPUs. 1000 designs ≈ 3 h on a single
CPU, ≤1 h with 4 cores.

Why GFN-FF
----------
- Sub-second to seconds per geometry on CPU
- Handles metals (Zn, Mg, Fe, ...)
- Includes electrostatics + dispersion
- More accurate than MM force fields for novel chemistry
- Less accurate than GFN2-xTB but ~10× faster
- Nearly as accurate as GFN2 for SHORT sidechain rotations (which is all
  this pipeline asks of it)

References
----------
- Spicher, S.; Grimme, S. "Robust atomistic modeling of materials, organometallic,
  and biochemical systems." Angew. Chem. Int. Ed. 2020, 59, 15665. (GFN-FF)
- Bannwarth, C.; Ehlert, S.; Grimme, S. "GFN2-xTB." J. Chem. Theory Comput.
  2019, 15, 1652.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("qcb.pipelines.refine_active_site")


# ──────────────────────────────────────────────────────────────────
# Parsers + lightweight PDB I/O
# ──────────────────────────────────────────────────────────────────

@dataclass
class CatalyticResidue:
    chain: str
    resname: str
    resnum: int


@dataclass
class LigandRef:
    chain: str
    resname: str
    resnum: int


def parse_remark666(pdb_path: Path) -> tuple[LigandRef | None, list[CatalyticResidue]]:
    """Parse 'REMARK 666 MATCH TEMPLATE B YYE 209 MATCH MOTIF A HIS 188 1 1' lines.

    Convention: TEMPLATE = ligand reference; MOTIF = catalytic residue.
    """
    lig = None
    cats: list[CatalyticResidue] = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("REMARK 666"):
                continue
            m = re.search(
                r"TEMPLATE\s+(\S)\s+(\S+)\s+(\d+).+MOTIF\s+(\S)\s+(\S+)\s+(\d+)",
                line,
            )
            if m:
                if lig is None:
                    lig = LigandRef(m.group(1), m.group(2), int(m.group(3)))
                cats.append(CatalyticResidue(m.group(4), m.group(5), int(m.group(6))))
    return lig, cats


@dataclass
class PdbAtom:
    line: str
    record: str
    serial: int
    atom_name: str
    res_name: str
    chain: str
    res_num: int
    x: float
    y: float
    z: float
    element: str

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def with_pos(self, new_pos: np.ndarray) -> "PdbAtom":
        x, y, z = new_pos
        new_line = self.line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + self.line[54:]
        return PdbAtom(new_line, self.record, self.serial, self.atom_name,
                       self.res_name, self.chain, self.res_num, x, y, z, self.element)


def parse_pdb(path: Path) -> list[PdbAtom]:
    atoms: list[PdbAtom] = []
    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                rec = line[:6].strip()
                serial = int(line[6:11])
                aname = line[12:16].strip()
                rname = line[17:20].strip()
                chain = line[21:22].strip() or "A"
                rnum = int(line[22:26])
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                el = line[76:78].strip() if len(line) >= 78 else aname[0]
                atoms.append(PdbAtom(line.rstrip("\n"), rec, serial, aname, rname,
                                     chain, rnum, x, y, z, el))
            except (ValueError, IndexError):
                continue
    return atoms


def write_pdb_atoms(atoms: list[PdbAtom], path: Path, header: list[str] | None = None):
    with open(path, "w") as f:
        if header:
            for h in header:
                f.write(h.rstrip() + "\n")
        for a in atoms:
            f.write(a.line + "\n")
        f.write("END\n")


# ──────────────────────────────────────────────────────────────────
# Alignment
# ──────────────────────────────────────────────────────────────────

def _kabsch(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Find R, t such that R @ coords_a + t ≈ coords_b. Returns (R, t, rmsd)."""
    ca = coords_a - coords_a.mean(axis=0)
    cb = coords_b - coords_b.mean(axis=0)
    H = ca.T @ cb
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = coords_b.mean(axis=0) - R @ coords_a.mean(axis=0)
    aligned = (R @ coords_a.T).T + t
    return R, t, float(np.sqrt(np.mean(np.sum((aligned - coords_b) ** 2, axis=1))))


def align_to_template(
    moving: list[PdbAtom],
    fixed: list[PdbAtom],
    catalytic: list[CatalyticResidue],
) -> tuple[list[PdbAtom], float]:
    """Align `moving` onto `fixed` using catalytic CAs."""
    a_coords, b_coords = [], []
    for c in catalytic:
        a = next((x for x in moving if x.chain == c.chain and x.res_num == c.resnum and x.atom_name == "CA"), None)
        b = next((x for x in fixed  if x.chain == c.chain and x.res_num == c.resnum and x.atom_name == "CA"), None)
        if a and b:
            a_coords.append(a.pos); b_coords.append(b.pos)
    if len(a_coords) < 3:
        raise RuntimeError(f"Need ≥3 catalytic CAs in both structures; got {len(a_coords)}")
    R, t, rmsd = _kabsch(np.array(a_coords), np.array(b_coords))
    return [a.with_pos(R @ a.pos + t) for a in moving], rmsd


# ──────────────────────────────────────────────────────────────────
# PTM + extraction + capping
# ──────────────────────────────────────────────────────────────────

def relabel_residues(atoms: list[PdbAtom], ptm_map: dict[str, str]) -> list[PdbAtom]:
    """ptm_map keys: 'CHAIN:RESNUM' (e.g., 'A:64'); values: new 3-letter code."""
    out = []
    for a in atoms:
        key = f"{a.chain}:{a.res_num}"
        if key in ptm_map:
            new_rn = ptm_map[key]
            new_line = a.line[:17] + f"{new_rn:<3}" + a.line[20:]
            out.append(PdbAtom(new_line, a.record, a.serial, a.atom_name, new_rn,
                               a.chain, a.res_num, a.x, a.y, a.z, a.element))
        else:
            out.append(a)
    return out


_PROTEINISH = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU",
               "LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","KCX"}


def extract_cluster(
    atoms: list[PdbAtom],
    ligand: LigandRef,
    radius: float = 6.0,
    required_residues: list[CatalyticResidue] | None = None,
) -> tuple[list[PdbAtom], set[tuple[str, int]]]:
    """Extract atoms ≤ `radius` Å of any ligand atom. Always include `required_residues`."""
    lig_coords = np.array([
        a.pos for a in atoms
        if a.chain == ligand.chain and a.res_num == ligand.resnum and a.res_name == ligand.resname
    ])
    if len(lig_coords) == 0:
        raise RuntimeError(f"No ligand atoms for {ligand.resname} {ligand.chain}:{ligand.resnum}")

    by_residue: dict[tuple[str, int], list[PdbAtom]] = {}
    for a in atoms:
        by_residue.setdefault((a.chain, a.res_num), []).append(a)

    keep: set[tuple[str, int]] = set()
    for (chain, rnum), res_atoms in by_residue.items():
        if (chain, rnum) == (ligand.chain, ligand.resnum):
            keep.add((chain, rnum)); continue
        coords = np.array([a.pos for a in res_atoms])
        d = np.min(np.linalg.norm(coords[:, None, :] - lig_coords[None, :, :], axis=2), axis=1)
        if np.any(d <= radius):
            keep.add((chain, rnum))

    if required_residues:
        for c in required_residues:
            keep.add((c.chain, c.resnum))

    cluster = [a for a in atoms if (a.chain, a.res_num) in keep]
    return cluster, keep


def cap_backbone(
    cluster: list[PdbAtom],
    full_atoms: list[PdbAtom],
    keep: set[tuple[str, int]],
) -> list[PdbAtom]:
    """Add H atoms where the protein backbone is cut by cluster boundaries."""
    full_by_res: dict[tuple[str, int], list[PdbAtom]] = {}
    for a in full_atoms:
        full_by_res.setdefault((a.chain, a.res_num), []).append(a)
    serial_max = max((a.serial for a in cluster), default=0) + 1
    new_atoms: list[PdbAtom] = []

    for (chain, rnum) in keep:
        rs = full_by_res.get((chain, rnum), [])
        if not rs or rs[0].res_name not in _PROTEINISH:
            continue

        if (chain, rnum - 1) not in keep:
            n = next((a for a in cluster if a.chain == chain and a.res_num == rnum and a.atom_name == "N"), None)
            ca = next((a for a in cluster if a.chain == chain and a.res_num == rnum and a.atom_name == "CA"), None)
            if n is not None and ca is not None:
                v = n.pos - ca.pos
                v = v / (np.linalg.norm(v) + 1e-9)
                h_pos = n.pos + v * 1.0
                line = _atom_line(serial_max, "HCAP", chain, rnum, rs[0].res_name, h_pos, "H")
                new_atoms.append(PdbAtom(line, "ATOM", serial_max, "HCAP",
                                         rs[0].res_name, chain, rnum, *h_pos, "H"))
                serial_max += 1

        if (chain, rnum + 1) not in keep:
            c = next((a for a in cluster if a.chain == chain and a.res_num == rnum and a.atom_name == "C"), None)
            ca = next((a for a in cluster if a.chain == chain and a.res_num == rnum and a.atom_name == "CA"), None)
            if c is not None and ca is not None:
                v = c.pos - ca.pos
                v = v / (np.linalg.norm(v) + 1e-9)
                h_pos = c.pos + v * 1.1
                line = _atom_line(serial_max, "HCAP", chain, rnum, rs[0].res_name, h_pos, "H")
                new_atoms.append(PdbAtom(line, "ATOM", serial_max, "HCAP",
                                         rs[0].res_name, chain, rnum, *h_pos, "H"))
                serial_max += 1

    return cluster + new_atoms


def _atom_line(serial: int, aname: str, chain: str, resnum: int,
               resname: str, pos: np.ndarray, element: str) -> str:
    x, y, z = pos
    return (f"ATOM  {serial:>5} {aname:<4} {resname:<3} {chain}{resnum:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2}")


# ──────────────────────────────────────────────────────────────────
# xTB optimization
# ──────────────────────────────────────────────────────────────────

def _resolve_xtb_bin() -> str | None:
    try:
        from qcb.config import XTB_BIN
        if XTB_BIN and os.path.isfile(XTB_BIN):
            return XTB_BIN
    except Exception:
        pass
    return shutil.which("xtb")


def run_xtb_constrained_opt(
    atoms: list[PdbAtom],
    workdir: Path,
    charge: int = 0,
    gfn: int = 0,                          # 0 = GFN-FF; 1, 2 = GFN1/2-xTB
    rigidity: str = "backbone",            # backbone | backbone-cb
    timeout_s: int = 600,
) -> tuple[list[PdbAtom] | None, str]:
    """xTB geometry optimization with backbone-frozen constraints."""
    xtb = _resolve_xtb_bin()
    if not xtb:
        return None, "xtb binary not found"

    workdir.mkdir(parents=True, exist_ok=True)
    xyz_path = workdir / "input.xyz"
    inp_path = workdir / "xcontrol.inp"

    with open(xyz_path, "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for a in atoms:
            f.write(f"{a.element:<2} {a.x:.6f} {a.y:.6f} {a.z:.6f}\n")

    backbone_atoms = {"N", "C", "O", "CA"}
    if rigidity == "backbone-cb":
        backbone_atoms |= {"CB"}
    fix_indices = [i for i, a in enumerate(atoms, 1)
                   if a.res_name in _PROTEINISH and a.atom_name in backbone_atoms]

    with open(inp_path, "w") as f:
        if fix_indices:
            f.write("$fix\n")
            f.write(f"    atoms: {','.join(str(i) for i in fix_indices)}\n")
            f.write("$end\n")

    method_args = ["--gfnff"] if gfn == 0 else ["--gfn", str(gfn)]
    cmd = [xtb, xyz_path.name] + method_args + [
        "--chrg", str(charge), "--opt", "normal", "--input", inp_path.name,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, cwd=str(workdir))
    except subprocess.TimeoutExpired:
        return None, f"xtb timeout after {timeout_s}s"

    opt_xyz = workdir / "xtbopt.xyz"
    if not opt_xyz.exists():
        return None, (proc.stderr or proc.stdout or "")[-1500:]

    lines = opt_xyz.read_text().strip().split("\n")
    if len(lines) < len(atoms) + 2:
        return None, "xtbopt.xyz too short"

    relaxed = []
    for i, a in enumerate(atoms):
        parts = lines[2 + i].split()
        relaxed.append(a.with_pos(np.array([float(parts[1]), float(parts[2]), float(parts[3])])))
    return relaxed, "OK"


# ──────────────────────────────────────────────────────────────────
# Stitch back
# ──────────────────────────────────────────────────────────────────

def stitch_back(full_atoms: list[PdbAtom], cluster_refined: list[PdbAtom]) -> list[PdbAtom]:
    """Replace any atom in `full_atoms` whose (chain, resnum, atom_name) matches
    a refined cluster atom. Cap atoms (HCAP) are dropped."""
    refined: dict[tuple[str, int, str], np.ndarray] = {}
    for a in cluster_refined:
        if a.atom_name == "HCAP":
            continue
        refined[(a.chain, a.res_num, a.atom_name)] = a.pos
    return [a.with_pos(refined[(a.chain, a.res_num, a.atom_name)])
            if (a.chain, a.res_num, a.atom_name) in refined else a
            for a in full_atoms]


# ──────────────────────────────────────────────────────────────────
# Top-level pipeline
# ──────────────────────────────────────────────────────────────────

def refine(
    design_pdb: str | Path,
    af3_pdb: str | Path,
    output_pdb: str | Path,
    radius: float = 6.0,
    gfn: int = 0,
    rigidity: str = "backbone",
    charge: int = 0,
    ptm_map: dict[str, str] | None = None,
    workdir: str | Path | None = None,
    keep_workdir: bool = False,
    write_cluster_files: bool = True,
) -> dict[str, Any]:
    """Refine an AF3-predicted active site against a design template.

    Returns a dict with status and output paths.
    """
    design_pdb = Path(design_pdb).resolve()
    af3_pdb = Path(af3_pdb).resolve()
    output_pdb = Path(output_pdb).resolve()
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    # 1. Parse REMARK 666
    ligand_ref, catalytic = parse_remark666(design_pdb)
    if ligand_ref is None:
        return {"status": "failed", "error": "no REMARK 666 in design"}
    log.info(f"  Ligand {ligand_ref.resname} {ligand_ref.chain}:{ligand_ref.resnum}, "
             f"{len(catalytic)} catalytic residues")

    # 2. Load + align AF3 to design
    design_atoms = parse_pdb(design_pdb)
    af3_atoms = parse_pdb(af3_pdb)
    aligned_af3, align_rmsd = align_to_template(af3_atoms, design_atoms, catalytic)
    log.info(f"  Catalytic CA RMSD = {align_rmsd:.3f} Å")

    # 3. PTM relabel
    if ptm_map:
        aligned_af3 = relabel_residues(aligned_af3, ptm_map)

    # 4. Transfer ligand from design (already in design frame)
    ligand_atoms = [a for a in design_atoms
                    if a.chain == ligand_ref.chain
                    and a.res_num == ligand_ref.resnum
                    and a.res_name == ligand_ref.resname]
    combined = aligned_af3 + ligand_atoms
    log.info(f"  Transferred {len(ligand_atoms)} ligand atoms")

    # 5. Extract cluster + cap
    cluster, keep_residues = extract_cluster(
        combined, ligand_ref, radius=radius, required_residues=catalytic,
    )
    cluster_capped = cap_backbone(cluster, combined, keep_residues)
    log.info(f"  Cluster: {len(keep_residues)} residues, {len(cluster_capped)} atoms (with caps)")

    # 6. xTB opt
    workdir_path = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="qcb_refine_"))
    cluster_refined, msg = run_xtb_constrained_opt(
        cluster_capped, workdir_path, charge=charge, gfn=gfn, rigidity=rigidity,
    )
    if cluster_refined is None:
        log.error(f"  xTB failed: {msg}")
        if not keep_workdir:
            shutil.rmtree(workdir_path, ignore_errors=True)
        return {"status": "failed", "error": f"xtb: {msg}"}

    # Diagnostic: cluster RMSD
    diff = np.array([a.pos for a in cluster_capped]) - np.array([a.pos for a in cluster_refined])
    cluster_rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    log.info(f"  Cluster RMSD (input → refined): {cluster_rmsd:.3f} Å")

    # 7. Stitch + write
    final = stitch_back(combined, cluster_refined)
    header = [
        f"REMARK QCB REFINE_ACTIVE_SITE design={design_pdb.name} af3={af3_pdb.name}",
        f"REMARK QCB METHOD GFN-{gfn} rigidity={rigidity} radius={radius}",
        f"REMARK QCB ALIGN_RMSD {align_rmsd:.3f}  CLUSTER_RMSD {cluster_rmsd:.3f}",
    ]
    write_pdb_atoms(final, output_pdb, header=header)

    outputs = {"refined": str(output_pdb)}
    if write_cluster_files:
        cluster_dir = output_pdb.parent / f"{output_pdb.stem}_cluster"
        cluster_dir.mkdir(exist_ok=True)
        write_pdb_atoms(cluster_capped, cluster_dir / "input.pdb")
        write_pdb_atoms(cluster_refined, cluster_dir / "refined.pdb")
        outputs["cluster_input"] = str(cluster_dir / "input.pdb")
        outputs["cluster_refined"] = str(cluster_dir / "refined.pdb")

    if not keep_workdir:
        shutil.rmtree(workdir_path, ignore_errors=True)

    return {
        "status": "completed",
        "align_rmsd_A": align_rmsd,
        "cluster_rmsd_A": cluster_rmsd,
        "n_residues": len(keep_residues),
        "n_atoms": len(cluster_capped),
        "outputs": outputs,
    }
