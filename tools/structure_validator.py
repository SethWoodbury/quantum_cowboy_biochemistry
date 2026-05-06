#!/usr/bin/env python
"""structure_validator.py — defensive structural sanity checks for design pipelines.

Catches the classes of failure we hit overnight:
  * Frame explosion (anchors held but rest of molecule decoupled by mace cutoff)
  * PDB metadata destruction (ASE.write losing atom names / REMARKs / residue info)
  * Bond stretching beyond physical limits
  * Reactive-distance drift past tolerance
  * Zn coordination collapse

Usage:
    from structure_validator import StructureValidator, validate_or_die
    v = StructureValidator(template_pdb="/path/to/source.pdb")
    v.check_all(candidate_pdb="/path/to/output.pdb")     # raises on failure

The template_pdb provides ground truth for atom names, residue records,
and frame anchors (fixed atoms identified by name+chain+resseq).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# ----- minimal PDB parser (preserves raw lines for fidelity checks) ---------
@dataclass
class PdbAtom:
    raw: str
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


def parse_pdb(path: Path) -> tuple[list[PdbAtom], list[str]]:
    atoms: list[PdbAtom] = []
    remarks: list[str] = []
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
            remarks.append(line.rstrip("\n"))
    return atoms, remarks


# ----- check helpers --------------------------------------------------------
def _coord(a: PdbAtom) -> np.ndarray:
    return np.array([a.x, a.y, a.z])


def _key(a: PdbAtom) -> tuple:
    """Identity of an atom that's stable across pipeline stages."""
    return (a.chain, a.resname, a.resseq, a.icode, a.name)


# ----- failure dataclass ----------------------------------------------------
@dataclass
class CheckResult:
    name: str
    ok: bool
    severity: str = "error"     # "error" | "warning" | "info"
    detail: str = ""
    data: dict = field(default_factory=dict)


# ----- main validator -------------------------------------------------------
class StructureValidator:
    """All checks operate on a (template, candidate) pair so that frame and
    metadata expectations come from the user-supplied source PDB rather than
    being hard-coded."""

    def __init__(self, template_pdb: str | Path):
        atoms, remarks = parse_pdb(Path(template_pdb))
        self.template_atoms = atoms
        self.template_remarks = remarks
        self.template_keys = {_key(a): i for i, a in enumerate(atoms)}
        if len(self.template_keys) != len(atoms):
            raise RuntimeError("template PDB has duplicate (chain,resseq,name) keys")

    # ----- individual checks -----
    def check_atom_count(self, cand_atoms: list[PdbAtom]) -> CheckResult:
        ok = len(cand_atoms) == len(self.template_atoms)
        return CheckResult(
            "atom_count", ok,
            detail=f"template={len(self.template_atoms)} candidate={len(cand_atoms)}",
        )

    def check_atom_identity(self, cand_atoms: list[PdbAtom]) -> CheckResult:
        """Every candidate atom should match template's (chain,resname,resseq,name).
        Flags ASE.write(pdb) which collapses everything into 'MOL' residues."""
        mismatches = []
        for i, (t, c) in enumerate(zip(self.template_atoms, cand_atoms)):
            if (t.name, t.resname, t.chain, t.resseq) != (c.name, c.resname, c.chain, c.resseq):
                mismatches.append(
                    f"i={i}: template={t.name!r}/{t.resname}/{t.chain}{t.resseq} "
                    f"≠ candidate={c.name!r}/{c.resname}/{c.chain}{c.resseq}"
                )
                if len(mismatches) > 5:
                    mismatches.append("…")
                    break
        ok = len(mismatches) == 0
        return CheckResult(
            "atom_identity_preserved", ok,
            detail="; ".join(mismatches) if not ok else "all atom records match template",
        )

    def check_rigid_body_anchors(self, cand_atoms: list[PdbAtom],
                                  anchor_keys: Iterable[tuple],
                                  pairwise_tol_A: float = 0.01,
                                  align_residual_tol_A: float = 0.05) -> CheckResult:
        """For ``FixInternals``-style rigid-body constraints, the *shape* of
        the anchor set must be preserved but the rigid body may translate /
        rotate. Check (a) max change in pairwise anchor-anchor distances, and
        (b) residual after Kabsch-aligning candidate anchors onto template
        anchors.
        """
        idxs = [self.template_keys[k] for k in anchor_keys]
        T = np.array([_coord(self.template_atoms[i]) for i in idxs])
        C = np.array([_coord(cand_atoms[i]) for i in idxs])

        pw_T = np.linalg.norm(T[:, None, :] - T[None, :, :], axis=2)
        pw_C = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
        max_pair_change = float(np.abs(pw_C - pw_T).max())

        # Kabsch
        cT, cC = T.mean(0), C.mean(0)
        H = (C - cC).T @ (T - cT)
        U, _, Vt = np.linalg.svd(H)
        sign = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, sign]) @ U.T
        C_aligned = (C - cC) @ R.T + cT
        residual = float(np.linalg.norm(C_aligned - T, axis=1).max())

        ok = (max_pair_change < pairwise_tol_A) and (residual < align_residual_tol_A)
        return CheckResult(
            "rigid_body_anchors_preserved", ok,
            detail=f"pairwise Δd max={max_pair_change:.5f} Å (tol {pairwise_tol_A}); "
                   f"Kabsch-aligned residual max={residual:.5f} Å (tol {align_residual_tol_A})",
            data={"max_pairwise_change_A": max_pair_change, "kabsch_residual_A": residual},
        )

    def check_frame_anchors(self, cand_atoms: list[PdbAtom],
                            anchor_keys: Iterable[tuple],
                            tol_A: float = 0.01) -> CheckResult:
        """Anchors that were $fix'd should be at template positions (rigid-body
        frame check). If the molecule was Kabsch-aligned to template, anchors
        should be at template positions exactly."""
        max_disp = 0.0
        worst = None
        for k in anchor_keys:
            i = self.template_keys[k]
            t = self.template_atoms[i]
            c = cand_atoms[i]
            d = float(np.linalg.norm(_coord(t) - _coord(c)))
            if d > max_disp:
                max_disp, worst = d, (k, d)
        ok = max_disp < tol_A
        return CheckResult(
            "frame_anchors_pinned", ok,
            detail=f"max displacement {max_disp:.4f} Å (tol {tol_A:.3f}); "
                   f"worst anchor={worst}",
            data={"max_disp_A": max_disp, "worst_anchor": list(worst) if worst else None},
        )

    def check_no_decoupling(self, cand_atoms: list[PdbAtom],
                            anchor_keys: Iterable[tuple],
                            covalent_max_A: float = 2.5,
                            cutoff_A: float = 5.0) -> CheckResult:
        """Each anchor must have at least one *non-anchor* atom within
        covalent-bond range. If an anchor is more than `cutoff_A` from any
        non-anchor atom, the molecule has decoupled into vacuum (the failure
        mode we hit tonight: CAs pinned at source while rest of molecule flew
        64 Å away through the mace cutoff)."""
        anchor_idxs = [self.template_keys[k] for k in anchor_keys]
        anchor_set = set(anchor_idxs)
        coords = np.array([_coord(a) for a in cand_atoms])
        non_anchor_idxs = [i for i in range(len(cand_atoms)) if i not in anchor_set]
        non_anchor_xyz = coords[non_anchor_idxs]

        worst = None; max_min_dist = 0.0
        for ai in anchor_idxs:
            d = np.linalg.norm(non_anchor_xyz - coords[ai], axis=1).min()
            if d > max_min_dist:
                max_min_dist = float(d)
                worst = (cand_atoms[ai].chain, cand_atoms[ai].resname,
                         cand_atoms[ai].resseq, cand_atoms[ai].name, d)
        ok = max_min_dist < cutoff_A
        sev = "error" if max_min_dist > cutoff_A else \
              ("warning" if max_min_dist > covalent_max_A else "info")
        return CheckResult(
            "anchors_not_decoupled", ok, severity=sev,
            detail=f"farthest-isolated anchor: {worst} ({max_min_dist:.2f} Å from any "
                   f"non-anchor; covalent expected ≤ {covalent_max_A} Å, "
                   f"mace-cutoff vacuum > {cutoff_A} Å)",
            data={"max_anchor_isolation_A": max_min_dist, "worst_anchor": list(worst) if worst else None},
        )

    def check_bond_integrity(self, cand_atoms: list[PdbAtom],
                             max_stretch_factor: float = 1.5,
                             exempt_pairs: list[tuple[tuple, tuple]] | None = None,
                             ) -> CheckResult:
        """For every (template,candidate) atom pair where template's nearest
        same-residue heavy atom is within 2.0 Å, candidate's same neighbor
        should be within `max_stretch_factor *` that distance.

        Args:
            exempt_pairs: list of (atom_key_a, atom_key_b) pairs to skip from
                the check — required for TS / IRC workflows where a reactive
                bond (e.g. P-Olg) is intentionally being broken or formed.
        """
        exempt_set: set[frozenset[int]] = set()
        for ka, kb in (exempt_pairs or []):
            exempt_set.add(frozenset((self.template_keys[ka], self.template_keys[kb])))

        residues = {}
        for i, a in enumerate(self.template_atoms):
            if a.element == "H" or a.name.startswith("H"):
                continue
            residues.setdefault((a.chain, a.resname, a.resseq), []).append(i)

        worst_stretch = 1.0; worst_pair = None
        for residue_idxs in residues.values():
            n = len(residue_idxs)
            if n < 2:
                continue
            for ii in range(n):
                for jj in range(ii + 1, n):
                    i, j = residue_idxs[ii], residue_idxs[jj]
                    if frozenset((i, j)) in exempt_set:
                        continue
                    t_d = np.linalg.norm(_coord(self.template_atoms[i]) - _coord(self.template_atoms[j]))
                    if t_d > 2.0:    # only consider neighbors
                        continue
                    c_d = np.linalg.norm(_coord(cand_atoms[i]) - _coord(cand_atoms[j]))
                    f = c_d / max(t_d, 1e-6)
                    if f > worst_stretch:
                        worst_stretch = f; worst_pair = (
                            f"{cand_atoms[i].name}({cand_atoms[i].resname}{cand_atoms[i].resseq})",
                            f"{cand_atoms[j].name}({cand_atoms[j].resname}{cand_atoms[j].resseq})",
                            t_d, c_d,
                        )
        ok = worst_stretch <= max_stretch_factor
        return CheckResult(
            "bond_integrity", ok, severity="error" if worst_stretch > 2.0 else "warning",
            detail=f"max bond stretch factor {worst_stretch:.3f} "
                   f"(tol {max_stretch_factor:.2f}); worst pair: {worst_pair}; "
                   f"{len(exempt_set)} reactive pair(s) exempt",
            data={"max_stretch_factor": worst_stretch, "worst_pair": list(worst_pair) if worst_pair else None},
        )

    def check_reactive_distances(self, cand_atoms: list[PdbAtom],
                                 reactive_keys: list[tuple[tuple, tuple]],
                                 targets_A: list[float],
                                 tol_A: float = 0.10) -> CheckResult:
        """Each pair (k_a, k_b) with target d_target should be within tol_A."""
        results = []
        all_ok = True
        for (ka, kb), target in zip(reactive_keys, targets_A):
            ia, ib = self.template_keys[ka], self.template_keys[kb]
            d = float(np.linalg.norm(_coord(cand_atoms[ia]) - _coord(cand_atoms[ib])))
            ok = abs(d - target) <= tol_A
            all_ok = all_ok and ok
            results.append({
                "atoms": [list(ka), list(kb)],
                "target_A": target, "actual_A": d, "drift_A": d - target, "ok": ok,
            })
        return CheckResult(
            "reactive_distances", all_ok,
            detail="; ".join(f"{r['atoms'][0][-1]}-{r['atoms'][1][-1]}: "
                              f"target={r['target_A']:.3f} actual={r['actual_A']:.3f} "
                              f"drift={r['drift_A']:+.3f}" for r in results),
            data={"pairs": results},
        )

    def check_zn_coordination(self, cand_atoms: list[PdbAtom],
                              zn_resname: str = "ZN2",
                              expect_neighbors: int = 4,
                              max_neighbor_A: float = 2.5) -> CheckResult:
        """Each Zn should have ≥ N atoms within 2.5 Å. Detects Zn flying off."""
        zn_idxs = [i for i, a in enumerate(cand_atoms) if a.resname == zn_resname]
        coords = np.array([_coord(a) for a in cand_atoms])
        problems = []
        for zi in zn_idxs:
            d = np.linalg.norm(coords - coords[zi], axis=1)
            d[zi] = np.inf
            n_close = int((d <= max_neighbor_A).sum())
            if n_close < expect_neighbors:
                problems.append(
                    f"{cand_atoms[zi].name}({cand_atoms[zi].resname}{cand_atoms[zi].resseq}): "
                    f"{n_close} neighbors within {max_neighbor_A} Å (expect ≥{expect_neighbors})"
                )
        ok = len(problems) == 0
        return CheckResult(
            "zn_coordination", ok,
            detail="; ".join(problems) if not ok else f"{len(zn_idxs)} Zn(s) with ≥{expect_neighbors} neighbors each",
        )

    # ----- driver -----
    def check_all(self, candidate_pdb: str | Path,
                  anchor_keys: Iterable[tuple] | None = None,
                  anchor_mode: str = "fixed",   # "fixed" or "rigid"
                  reactive_pairs: list[tuple[tuple, tuple, float]] | None = None,
                  ) -> dict:
        """Run all relevant checks. Returns a dict suitable for JSON dump.

        anchor_mode:
          "fixed"  — FixAtoms-style hard pin; anchors must be at template positions.
          "rigid"  — FixInternals-style rigid body; anchors can translate/rotate
                     as a unit but pairwise shape is preserved.
        """
        cand_atoms, _ = parse_pdb(Path(candidate_pdb))

        results = []
        results.append(self.check_atom_count(cand_atoms))
        if results[-1].ok:
            results.append(self.check_atom_identity(cand_atoms))
            results.append(self.check_bond_integrity(cand_atoms))
            results.append(self.check_zn_coordination(cand_atoms))
            if anchor_keys is not None:
                if anchor_mode == "rigid":
                    results.append(self.check_rigid_body_anchors(cand_atoms, anchor_keys))
                else:
                    results.append(self.check_frame_anchors(cand_atoms, anchor_keys))
                results.append(self.check_no_decoupling(cand_atoms, anchor_keys))
            if reactive_pairs is not None:
                keys = [(p[0], p[1]) for p in reactive_pairs]
                targets = [p[2] for p in reactive_pairs]
                results.append(self.check_reactive_distances(cand_atoms, keys, targets))

        all_ok = all(r.ok or r.severity != "error" for r in results)
        out = {
            "template": str(self.template_atoms[0].raw)[:50] + "...",
            "candidate": str(candidate_pdb),
            "all_pass": all(r.ok for r in results),
            "no_critical_errors": all_ok,
            "results": [
                {"name": r.name, "ok": r.ok, "severity": r.severity,
                 "detail": r.detail, "data": r.data} for r in results
            ],
        }
        return out


def validate_or_die(template_pdb: str | Path, candidate_pdb: str | Path,
                    anchor_keys: list[tuple] | None = None,
                    anchor_mode: str = "fixed",
                    reactive_pairs: list[tuple] | None = None,
                    print_summary: bool = True) -> dict:
    v = StructureValidator(template_pdb)
    out = v.check_all(candidate_pdb, anchor_keys=anchor_keys,
                      anchor_mode=anchor_mode,
                      reactive_pairs=reactive_pairs)
    if print_summary:
        print(f"\n=== StructureValidator: {Path(candidate_pdb).name} ===")
        for r in out["results"]:
            mark = "✅" if r["ok"] else ("⚠️" if r["severity"] == "warning" else "❌")
            print(f"  {mark} {r['name']:30s} {r['detail']}")
        print(f"  ALL_PASS={out['all_pass']}  NO_CRITICAL_ERRORS={out['no_critical_errors']}")
    if not out["no_critical_errors"]:
        raise RuntimeError(
            f"Structure validation FAILED for {candidate_pdb}. "
            f"Failed checks: " +
            ", ".join(r["name"] for r in out["results"] if not r["ok"] and r["severity"] == "error")
        )
    return out


# ----- CLI ------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, required=True,
                   help="ground-truth source PDB (atom names, residues, anchor positions)")
    p.add_argument("--candidate", type=Path, required=True,
                   help="output PDB to validate against template")
    p.add_argument("--anchor-residues", nargs="+", default=[],
                   help="anchor specs of form chain:resname:resseq:atomname (e.g. A:HIS:55:CA)")
    p.add_argument("--reactive-pair", action="append", default=[],
                   help="reactive distance: chain:resname:resseq:atomname,chain:resname:resseq:atomname,target_A "
                        "(may be repeated)")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    def parse_key(s: str) -> tuple:
        c, rn, rs, n = s.split(":")
        return (c, rn, int(rs), "", n)

    anchors = [parse_key(s) for s in args.anchor_residues]
    reactive = []
    for spec in args.reactive_pair:
        ka_s, kb_s, target_s = spec.rsplit(",", 2)
        reactive.append((parse_key(ka_s), parse_key(kb_s), float(target_s)))

    out = validate_or_die(
        args.template, args.candidate,
        anchor_keys=anchors or None,
        reactive_pairs=reactive or None,
    )
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
