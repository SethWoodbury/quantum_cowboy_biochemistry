"""±Imaginary-mode displacement test for TS verification.

Why this exists
---------------
A bona-fide first-order saddle, when displaced ±delta along its imaginary
eigenmode and then relaxed, decays into TWO different basins: the reactant
basin from the negative branch, the product basin from the positive branch
(or vice versa — the sign convention is arbitrary).

If both branches relax to the SAME basin, the imag mode is not the reaction
coordinate. If neither converges, the saddle was likely an artefact.

This is an inexpensive, **highly diagnostic** finishing step on top of
``refine-ts`` + ``validate-ts``. Cheaper than a full IRC.

Reaction-agnostic
-----------------
- The mode is supplied either as an XYZ (loaded via
  :func:`quantum_engine.qm.dimer.load_mode_from_xyz`) or as an .npy file
  containing the (N, 3) displacement vector.
- Pass / fail rules are heuristics: forward and backward branches must
  reach DIFFERENT basins, with energies BELOW the TS by at least
  ``--basin-min-drop-eV``. The basin "fingerprint" is the displacement
  vector relative to the TS, projected onto the imag mode — sign and
  magnitude define the basin direction.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from quantum_engine.units import EV_TO_KCAL

log = logging.getLogger("quantum_engine.ops.imag_mode_displace")


@dataclass
class IRCLikeBranch:
    sign: str  # "+" or "-"
    initial_displacement_A: float
    energy_eV: float
    fmax_final: float
    converged: bool
    delta_to_ts_kcal_mol: float
    projection_onto_mode: float
    output_pdb: str | None = None


@dataclass
class IRCLikeReport:
    overall_pass: bool
    ts_energy_eV: float
    branches: list[IRCLikeBranch] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)


def _normalize_mode(mode: np.ndarray, n_atoms: int) -> np.ndarray:
    v = np.asarray(mode, dtype=float)
    if v.ndim == 1:
        v = v.reshape(-1, 3)
    if v.shape != (n_atoms, 3):
        raise ValueError(f"mode shape {v.shape} != ({n_atoms}, 3)")
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("mode norm is ~zero")
    return v / norm


def verify_irc_like(
    ts_atoms: Atoms,
    *,
    imag_mode: np.ndarray,
    outdir: str | Path,
    displacement_A: float = 0.20,
    fmax: float = 0.05,
    max_steps: int = 200,
    optimizer: str = "lbfgs",
    basin_min_drop_eV: float = 0.005,
    write_pdbs: bool = True,
    bt_struct=None,
    charge: int | None = None,
) -> IRCLikeReport:
    """Push the TS atoms ±displacement along the imag mode and relax both branches.

    Args:
        ts_atoms: TS structure, calculator attached.
        imag_mode: ``(N, 3)`` or ``(3N,)`` displacement vector. Will be
            normalised.
        outdir: where to write per-branch outputs.
        displacement_A: |displacement| in Å; default 0.20 (gentle but past
            the saddle curvature). Bump to 0.30-0.50 for very stiff TSs.
        fmax, max_steps, optimizer: per-branch optimizer config.
        basin_min_drop_eV: each branch must drop by at least this much in
            energy from the TS to count as "decayed to a basin".
        write_pdbs: dump branch endpoints as PDBs (needs bt_struct for
            template).
        bt_struct: optional biotite template for PDB writing.
        charge: REMARK charge to write into output PDBs.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(ts_atoms)
    mode_unit = _normalize_mode(imag_mode, n)
    log.info("verify_irc_like: displacement=%.3f Å, fmax=%.4f, max_steps=%d",
             displacement_A, fmax, max_steps)

    ts_pos = ts_atoms.get_positions().copy()
    ts_energy = float(ts_atoms.get_potential_energy())
    log.info("verify_irc_like: TS energy = %.4f eV", ts_energy)

    branches: list[IRCLikeBranch] = []
    pdb_paths: list[str | None] = [None, None]

    from quantum_engine.io import write_pdb
    from quantum_engine.ops import opt as opt_op

    for sign_idx, sign in enumerate([+1, -1]):
        branch_dir = outdir / f"branch_{'+' if sign > 0 else '-'}"
        branch_dir.mkdir(parents=True, exist_ok=True)
        atoms = ts_atoms.copy()
        atoms.calc = ts_atoms.calc
        atoms.info = dict(ts_atoms.info)
        new_pos = ts_pos + sign * displacement_A * mode_unit
        atoms.set_positions(new_pos)

        try:
            res = opt_op.run(
                atoms, atoms.calc, branch_dir,
                constraint=None,
                optimizer=optimizer,
                fmax=fmax, max_steps=max_steps,
            )
            relaxed = res["atoms"]
            e_branch = float(res.get("energy_eV", relaxed.get_potential_energy()))
            fmax_branch = float(np.linalg.norm(relaxed.get_forces(), axis=1).max())
            converged = bool(res.get("converged", fmax_branch <= fmax * 1.5))
            # Projection of relaxed displacement onto imag mode (positive on the +
            # branch should remain positive)
            disp_relaxed = relaxed.get_positions() - ts_pos
            proj = float(np.sum(disp_relaxed * mode_unit))
            delta_kcal = (e_branch - ts_energy) * EV_TO_KCAL
            log.info(
                "verify_irc_like: branch %s — E=%.4f eV (Δ=%.2f kcal/mol), "
                "fmax_final=%.4f, projection=%.4f Å, converged=%s",
                "+" if sign > 0 else "-", e_branch, delta_kcal,
                fmax_branch, proj, converged,
            )

            pdb_path: str | None = None
            if write_pdbs:
                pdb_path = str(branch_dir / f"branch_{'plus' if sign > 0 else 'minus'}.pdb")
                write_pdb(relaxed, bt_struct, pdb_path,
                          total_charge=charge, energy_eV=e_branch)

            branches.append(IRCLikeBranch(
                sign="+" if sign > 0 else "-",
                initial_displacement_A=displacement_A,
                energy_eV=e_branch,
                fmax_final=fmax_branch,
                converged=converged,
                delta_to_ts_kcal_mol=delta_kcal,
                projection_onto_mode=proj,
                output_pdb=pdb_path,
            ))
            pdb_paths[sign_idx] = pdb_path
        except Exception as exc:
            log.error("verify_irc_like: branch %s failed: %s",
                      "+" if sign > 0 else "-", exc)
            branches.append(IRCLikeBranch(
                sign="+" if sign > 0 else "-",
                initial_displacement_A=displacement_A,
                energy_eV=float("nan"),
                fmax_final=float("nan"),
                converged=False,
                delta_to_ts_kcal_mol=float("nan"),
                projection_onto_mode=float("nan"),
            ))

    # Pass/fail logic
    notes: list[str] = []
    overall_pass = True
    if any(not b.converged for b in branches):
        notes.append("at least one branch did not converge (fmax > target)")
        overall_pass = False
    # Each branch must drop at least basin_min_drop_eV below the TS — that's
    # the minimum signal a true saddle gives when displaced along its imag
    # mode. NaN energies (failed relax) automatically fail.
    drops_ok = all(
        np.isfinite(b.energy_eV) and (ts_energy - b.energy_eV) >= basin_min_drop_eV
        for b in branches
    )
    if not drops_ok:
        notes.append(
            f"at least one branch did not drop {basin_min_drop_eV:.4f} eV below TS"
        )
        overall_pass = False
    # Branches must end in DIFFERENT basins (different sign of mode projection)
    if len(branches) == 2 and all(np.isfinite(b.projection_onto_mode) for b in branches):
        sgn_a = np.sign(branches[0].projection_onto_mode)
        sgn_b = np.sign(branches[1].projection_onto_mode)
        if sgn_a == sgn_b and sgn_a != 0:
            notes.append(
                f"both branches relax to the same side of the imag mode "
                f"(projections {branches[0].projection_onto_mode:.3f}, "
                f"{branches[1].projection_onto_mode:.3f}) — NOT a TS along this mode"
            )
            overall_pass = False

    summary_path = outdir / "verify_irc_like.json"
    summary_path.write_text(json.dumps({
        "overall_pass": overall_pass,
        "ts_energy_eV": ts_energy,
        "displacement_A": displacement_A,
        "fmax_target": fmax,
        "branches": [asdict(b) for b in branches],
        "notes": notes,
    }, indent=2, default=str))

    return IRCLikeReport(
        overall_pass=overall_pass,
        ts_energy_eV=ts_energy,
        branches=branches,
        notes=notes,
        outputs={"summary": str(summary_path),
                 **{f"branch_{i}": p for i, p in enumerate(pdb_paths) if p}},
    )


def load_mode(path: str | Path, n_atoms: int) -> np.ndarray:
    """Load an imag-mode vector from .npy or .xyz."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        v = np.load(p)
    else:
        from quantum_engine.qm.dimer import load_mode_from_xyz
        v = load_mode_from_xyz(p, n_atoms=n_atoms)
    return v


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    *,
    imag_mode_path: str | Path | None = None,
    imag_mode: np.ndarray | None = None,
    displacement_A: float = 0.20,
    fmax: float = 0.05,
    max_steps: int = 200,
    optimizer: str = "lbfgs",
    basin_min_drop_eV: float = 0.005,
    bt_struct=None,
    charge: int | None = None,
    **kwargs,
) -> dict:
    """``ops`` adapter signature for ``qcb verify-irc-like``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if atoms.calc is None:
        if calculator is None:
            raise ValueError("verify-irc-like: no calc and none supplied")
        atoms.calc = calculator
    if constraint is not None:
        atoms.set_constraint(constraint if isinstance(constraint, list) else [constraint])
    if imag_mode is None:
        if imag_mode_path is None:
            raise ValueError("Must supply imag_mode (array) or imag_mode_path (.npy/.xyz)")
        imag_mode = load_mode(imag_mode_path, n_atoms=len(atoms))

    report = verify_irc_like(
        atoms,
        imag_mode=imag_mode,
        outdir=outdir,
        displacement_A=displacement_A,
        fmax=fmax,
        max_steps=max_steps,
        optimizer=optimizer,
        basin_min_drop_eV=basin_min_drop_eV,
        bt_struct=bt_struct,
        charge=charge,
    )
    return {
        "status": "completed" if report.overall_pass else "warned",
        "overall_pass": report.overall_pass,
        "ts_energy_eV": report.ts_energy_eV,
        "branches": [asdict(b) for b in report.branches],
        "notes": report.notes,
        "outputs": report.outputs,
    }
