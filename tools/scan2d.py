#!/usr/bin/env python
"""scan2d.py — diagnostic 2-D relaxed scan around a 1-D scan TS guess.

Why this exists
---------------
The 1-D scan in :mod:`tools.scan_along_s` follows a constant-sum path
``d_A + d_B = sum_target`` swept in the asymmetry coordinate
``s = d_A − d_B``. That captures the saddle when the reaction coordinate
ALIGNS with the constant-sum line, but slices through it badly when the
true saddle has independent stretches in ``d_A`` and ``d_B``. This module
runs a small (3×3 / 5×5 / NxN) relaxed scan around the 1-D peak,
sweeping each bond independently, so the operator can confirm or refute
the 1-D path's saddle location.

Reaction-agnostic
-----------------
- Two reactive bonds are user-supplied (``--bond-a A,B``, ``--bond-b C,D``).
  They can be the forming + breaking bonds in an SN2-type reaction, or
  the two new C-C bonds in a Diels-Alder, etc.
- Tokens use the same grammar as :mod:`tools.endpoint_release`
  (``NAME.RESNAME[.RESID]``, integer 1-based serial, ``0:idx``).
- Defaults run a 3×3 grid with delta_d = 0.20 Å, but every knob is
  CLI-exposed.

Output
------
- ``scan2d.json`` with the energy grid, distances, and argmax indices.
- ``scan2d.png`` (matplotlib heatmap) if matplotlib is available.
- ``point_<i>_<j>.pdb`` for each grid point (relaxed geometry).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quantum_engine.io import load_structure, parse_constraints, build_fix_atoms, write_pdb
from quantum_engine.ops.charge_ledger import (
    ChargeLedger, append_remarks_to_pdb, inject_into_atoms, resolve_charge_and_spin,
)
from quantum_engine.select import preset_to_specs

log = logging.getLogger("tools.scan2d")


# ---------------------------------------------------------------------------
# Atom token resolution (shared idiom)
# ---------------------------------------------------------------------------
def _resolve_atom_token(tok: str, ase_atoms, bt_struct) -> int:
    s = tok.strip()
    if not s:
        raise ValueError("empty atom token")
    if s.startswith("0:"):
        return int(s[2:])
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s) - 1
    if "." in s:
        parts = s.split(".")
        if bt_struct is None:
            raise ValueError(f"atom token {tok!r} requires PDB template")
        if len(parts) == 2:
            name, resname = parts
            mask = (np.asarray(bt_struct.atom_name) == name) & \
                   (np.asarray(bt_struct.res_name) == resname)
        elif len(parts) == 3:
            name, resname, resid = parts
            mask = ((np.asarray(bt_struct.atom_name) == name)
                    & (np.asarray(bt_struct.res_name) == resname)
                    & (np.asarray(bt_struct.res_id) == int(resid)))
        else:
            raise ValueError(f"atom token {tok!r}: bad format")
        idxs = np.where(mask)[0]
        if not len(idxs):
            raise ValueError(f"atom token {tok!r}: no match in PDB template")
        return int(idxs[0])
    raise ValueError(
        f"atom token {tok!r}: must be NAME.RESNAME[.RESID], 1-based serial, or 0:idx"
    )


def _parse_grid(spec: str) -> tuple[int, int]:
    """Parse '3x3' or '5x7' into (n_a, n_b)."""
    s = spec.lower().strip().replace(" ", "")
    if "x" not in s:
        raise ValueError(f"--grid {spec!r} must be NxM (e.g. '3x3' or '5x5')")
    a, b = s.split("x", 1)
    try:
        return int(a), int(b)
    except ValueError as exc:
        raise ValueError(f"--grid {spec!r}: failed to parse integers") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class Scan2DResult:
    status: str
    grid_shape: tuple[int, int]
    bond_a: tuple[int, int]
    bond_b: tuple[int, int]
    d_a_grid: list[list[float]]  # actual distances achieved (post-relax)
    d_b_grid: list[list[float]]
    energy_grid_eV: list[list[float]]
    argmax_indices: tuple[int, int]
    argmin_indices: tuple[int, int]
    grid_pdbs: list[list[str]]
    summary_path: str
    plot_path: str | None
    warnings: list[str] = field(default_factory=list)


def scan_2d_around(
    input_pdb: str | Path,
    ts_guess_pdb: str | Path,
    *,
    out_dir: str | Path,
    bond_a: tuple[str, str],
    bond_b: tuple[str, str],
    grid: tuple[int, int] = (3, 3),
    delta_d: float = 0.20,
    delta_d_a: float | None = None,
    delta_d_b: float | None = None,
    boundary_fix_preset: str | None = None,
    fix_specs: Iterable[str] | None = None,
    free_specs: Iterable[str] | None = None,
    model: str = "mace-omol",
    head: str | None = None,
    device: str = "cuda",
    charge_ledger: ChargeLedger | None = None,
    cli_charge: int | None = None,
    cli_spin: int | None = None,
    fmax: float = 0.05,
    max_steps: int = 250,
    write_plot: bool = True,
) -> Scan2DResult:
    """Run a 2-D relaxed scan around a TS-guess geometry.

    Args:
        input_pdb: structure to load template + atoms from. Used for atom
            indexing — usually the SAME geometry as ``ts_guess_pdb``.
        ts_guess_pdb: the geometry the grid is centred on (the 1-D scan's
            argmax point, typically). May be the same as ``input_pdb``.
        out_dir: output directory.
        bond_a, bond_b: ``(token_a, token_b)`` pairs identifying the two
            bonds whose lengths are swept.
        grid: ``(n_a, n_b)`` — number of grid points in each direction.
            3×3 is the cheapest informative grid; 5×5 better resolves an
            off-axis saddle but costs ~3x more.
        delta_d: half-width of the sweep in BOTH directions, Å. The grid
            spans ``[d_center - delta_d, d_center + delta_d]`` for each bond.
        delta_d_a, delta_d_b: per-direction overrides (preferred over
            ``delta_d`` if set).
        boundary_fix_preset, fix_specs, free_specs: same constraint grammar
            as the rest of qcb.
        model, head, device, charge_ledger, cli_charge, cli_spin: standard.
        fmax, max_steps: per-grid-point optimizer convergence.
        write_plot: if True and matplotlib is available, save a heatmap PNG.

    Returns:
        :class:`Scan2DResult`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_pdb = Path(input_pdb)
    ts_guess_pdb = Path(ts_guess_pdb)

    n_a, n_b = grid
    if n_a < 2 or n_b < 2:
        raise ValueError(f"grid {grid} too small; need at least 2 points per axis")

    da_half = float(delta_d_a if delta_d_a is not None else delta_d)
    db_half = float(delta_d_b if delta_d_b is not None else delta_d)
    if da_half <= 0 or db_half <= 0:
        raise ValueError(f"delta_d must be > 0; got delta_d_a={da_half}, delta_d_b={db_half}")

    # Load template + TS guess (atoms identical, only positions differ)
    template_atoms, bt_struct, charge_hint = load_structure(input_pdb)
    ts_atoms, _, _ = load_structure(ts_guess_pdb)
    if len(ts_atoms) != len(template_atoms):
        raise ValueError(
            f"input_pdb and ts_guess_pdb have different atom counts "
            f"({len(template_atoms)} vs {len(ts_atoms)})"
        )

    # Resolve charge / spin
    charge, spin, ledger = resolve_charge_and_spin(
        ledger_path=None,
        cli_charge=cli_charge,
        cli_spin=cli_spin,
        pdb_charge_hint=charge_hint,
    )
    if charge_ledger is not None:
        ledger = charge_ledger
        charge = ledger.total
        spin = ledger.spin

    # Resolve bond indices
    a1 = _resolve_atom_token(bond_a[0], ts_atoms, bt_struct)
    a2 = _resolve_atom_token(bond_a[1], ts_atoms, bt_struct)
    b1 = _resolve_atom_token(bond_b[0], ts_atoms, bt_struct)
    b2 = _resolve_atom_token(bond_b[1], ts_atoms, bt_struct)
    log.info("scan2d: bond A = atoms %d-%d, bond B = atoms %d-%d", a1, a2, b1, b2)

    # Center distances
    pos_ts = ts_atoms.get_positions()
    d_a_center = float(np.linalg.norm(pos_ts[a1] - pos_ts[a2]))
    d_b_center = float(np.linalg.norm(pos_ts[b1] - pos_ts[b2]))
    log.info("scan2d: TS-guess distances: bond A = %.4f Å, bond B = %.4f Å",
             d_a_center, d_b_center)

    d_a_targets = np.linspace(d_a_center - da_half, d_a_center + da_half, n_a)
    d_b_targets = np.linspace(d_b_center - db_half, d_b_center + db_half, n_b)
    log.info("scan2d: bond A targets %s", d_a_targets.tolist())
    log.info("scan2d: bond B targets %s", d_b_targets.tolist())

    # Build boundary constraint
    fix_specs_list = list(fix_specs or [])
    free_specs_list = list(free_specs or [])
    excluded: set[str] = set()
    if boundary_fix_preset:
        preset_specs, preset_excluded = preset_to_specs(boundary_fix_preset, None)
        fix_specs_list = preset_specs + fix_specs_list
        excluded |= preset_excluded

    fix_mask = parse_constraints(ts_atoms, bt_struct, fix_specs_list, excluded) if fix_specs_list \
               else np.zeros(len(ts_atoms), dtype=bool)
    if free_specs_list:
        free_mask = parse_constraints(ts_atoms, bt_struct, free_specs_list, set())
        fix_mask &= ~free_mask
    boundary_constraint = build_fix_atoms(fix_mask)

    # Grid scan
    energy_grid = np.full((n_a, n_b), np.nan)
    d_a_grid = np.full((n_a, n_b), np.nan)
    d_b_grid = np.full((n_a, n_b), np.nan)
    pdb_paths: list[list[str]] = [["" for _ in range(n_b)] for _ in range(n_a)]

    from ase.constraints import FixBondLengths
    from quantum_engine.calc import make_calc
    from quantum_engine.ops import opt as opt_op

    for ia, da_target in enumerate(d_a_targets):
        for ib, db_target in enumerate(d_b_targets):
            # Start each grid point from the TS-guess geometry — independent
            # paths avoid drift accumulating along the scan
            atoms = ts_atoms.copy()
            atoms.info["charge"] = charge
            atoms.info["spin"] = spin
            if ledger is not None:
                inject_into_atoms(atoms, ledger)

            # Pull bonds toward target distances by pre-shifting positions.
            # When the two bonds share an atom (e.g. F-C and C-Cl share C),
            # accumulate that atom's contributions and apply ONCE so it
            # doesn't get double-shifted.
            pos = atoms.get_positions().copy()
            atom_shifts: dict[int, np.ndarray] = {}
            atom_shift_counts: dict[int, int] = {}
            cur_da = float(np.linalg.norm(pos[a1] - pos[a2]))
            shift_a = (da_target - cur_da) / 2.0
            v_a = (pos[a2] - pos[a1]) / max(cur_da, 1e-6)
            for atom_i, sign in ((a1, -1.0), (a2, +1.0)):
                atom_shifts[atom_i] = atom_shifts.get(atom_i, np.zeros(3)) + sign * shift_a * v_a
                atom_shift_counts[atom_i] = atom_shift_counts.get(atom_i, 0) + 1
            cur_db = float(np.linalg.norm(pos[b1] - pos[b2]))
            shift_b = (db_target - cur_db) / 2.0
            v_b = (pos[b2] - pos[b1]) / max(cur_db, 1e-6)
            for atom_i, sign in ((b1, -1.0), (b2, +1.0)):
                atom_shifts[atom_i] = atom_shifts.get(atom_i, np.zeros(3)) + sign * shift_b * v_b
                atom_shift_counts[atom_i] = atom_shift_counts.get(atom_i, 0) + 1
            # Average shift per atom (so a shared atom moves by the mean of
            # the two bond displacements rather than the sum)
            for atom_i, shift in atom_shifts.items():
                pos[atom_i] += shift / atom_shift_counts[atom_i]
            atoms.set_positions(pos)

            # Apply BOTH the boundary FixAtoms and a FixBondLengths pinning
            # the two reactive bonds at their target lengths
            constraints = []
            if boundary_constraint is not None:
                constraints.append(boundary_constraint)
            constraints.append(FixBondLengths(
                pairs=[[a1, a2], [b1, b2]],
                tolerance=1e-4,
            ))
            atoms.set_constraint(constraints)

            atoms.calc = make_calc(model=model, head=head, device=device, charge=charge)

            point_outdir = out_dir / f"point_{ia:02d}_{ib:02d}"
            try:
                res = opt_op.run(
                    atoms, atoms.calc, point_outdir,
                    constraint=None,  # already set on atoms
                    optimizer="lbfgs",
                    fmax=fmax,
                    max_steps=max_steps,
                )
                relaxed = res["atoms"]
                e_eV = float(res.get("energy_eV", relaxed.get_potential_energy()))
                pos_after = relaxed.get_positions()
                d_a_after = float(np.linalg.norm(pos_after[a1] - pos_after[a2]))
                d_b_after = float(np.linalg.norm(pos_after[b1] - pos_after[b2]))
                pdb_path = point_outdir / f"point_{ia:02d}_{ib:02d}.pdb"
                write_pdb(relaxed, bt_struct, pdb_path,
                          total_charge=charge, energy_eV=e_eV)
                if ledger is not None:
                    append_remarks_to_pdb(pdb_path, ledger)
                pdb_paths[ia][ib] = str(pdb_path)
                energy_grid[ia, ib] = e_eV
                d_a_grid[ia, ib] = d_a_after
                d_b_grid[ia, ib] = d_b_after
                log.info(
                    "scan2d: point (%d,%d) target=(%.3f,%.3f) actual=(%.3f,%.3f) E=%.4f eV",
                    ia, ib, da_target, db_target, d_a_after, d_b_after, e_eV,
                )
            except Exception as exc:
                log.error("scan2d: point (%d,%d) failed: %s", ia, ib, exc)

    # Argmax / argmin (treat NaN as -inf / +inf)
    finite = np.where(np.isfinite(energy_grid), energy_grid, -np.inf)
    am = np.unravel_index(np.argmax(finite), finite.shape)
    finite_min = np.where(np.isfinite(energy_grid), energy_grid, np.inf)
    amin = np.unravel_index(np.argmin(finite_min), finite_min.shape)

    # Plot
    plot_path: Path | None = None
    if write_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 5))
            cmap = ax.imshow(
                (energy_grid - np.nanmin(energy_grid)) * 23.0605,  # eV -> kcal
                origin="lower",
                aspect="auto",
                extent=[d_b_targets.min(), d_b_targets.max(),
                        d_a_targets.min(), d_a_targets.max()],
                cmap="viridis",
            )
            ax.set_xlabel("bond B (Å)")
            ax.set_ylabel("bond A (Å)")
            ax.set_title("scan2d: ΔE (kcal/mol) above min")
            plt.colorbar(cmap, ax=ax)
            ax.scatter([d_b_grid[am]], [d_a_grid[am]], marker="x", color="red",
                       s=100, label=f"argmax ({am[0]},{am[1]})")
            ax.scatter([d_b_grid[amin]], [d_a_grid[amin]], marker="o", color="white",
                       edgecolor="black", s=80, label=f"argmin ({amin[0]},{amin[1]})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            plot_path = out_dir / "scan2d.png"
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            log.info("scan2d: heatmap saved to %s", plot_path)
        except ImportError:
            log.info("matplotlib not installed; skipping heatmap")

    summary = {
        "input_pdb": str(input_pdb),
        "ts_guess_pdb": str(ts_guess_pdb),
        "model": model,
        "device": device,
        "charge": charge,
        "spin": spin,
        "bond_a_indices": [a1, a2],
        "bond_b_indices": [b1, b2],
        "bond_a_tokens": [bond_a[0], bond_a[1]],
        "bond_b_tokens": [bond_b[0], bond_b[1]],
        "d_a_center_A": d_a_center,
        "d_b_center_A": d_b_center,
        "delta_d_a": da_half,
        "delta_d_b": db_half,
        "grid": [n_a, n_b],
        "d_a_targets": d_a_targets.tolist(),
        "d_b_targets": d_b_targets.tolist(),
        "d_a_actual_grid": d_a_grid.tolist(),
        "d_b_actual_grid": d_b_grid.tolist(),
        "energy_grid_eV": energy_grid.tolist(),
        "argmax_indices": [int(am[0]), int(am[1])],
        "argmin_indices": [int(amin[0]), int(amin[1])],
        "fmax": fmax,
        "max_steps": max_steps,
        "boundary_fix_preset": boundary_fix_preset,
        "ledger": ledger.to_dict() if ledger is not None else None,
        "plot_path": str(plot_path) if plot_path else None,
    }
    summary_path = out_dir / "scan2d.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    warnings_out: list[str] = []
    # Saddle on edge?
    if am[0] in (0, n_a - 1) or am[1] in (0, n_b - 1):
        warnings_out.append(
            f"argmax at grid edge {am}; the saddle may sit OUTSIDE the scan window — "
            f"increase --delta-d (currently {da_half}/{db_half} Å)"
        )
    finite_mask = np.isfinite(energy_grid)
    if not finite_mask.all():
        n_failed = int((~finite_mask).sum())
        warnings_out.append(
            f"{n_failed}/{energy_grid.size} grid points failed (NaN energy)"
        )

    return Scan2DResult(
        status="completed",
        grid_shape=(n_a, n_b),
        bond_a=(a1, a2),
        bond_b=(b1, b2),
        d_a_grid=d_a_grid.tolist(),
        d_b_grid=d_b_grid.tolist(),
        energy_grid_eV=energy_grid.tolist(),
        argmax_indices=(int(am[0]), int(am[1])),
        argmin_indices=(int(amin[0]), int(amin[1])),
        grid_pdbs=pdb_paths,
        summary_path=str(summary_path),
        plot_path=str(plot_path) if plot_path else None,
        warnings=warnings_out,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scan2d",
        description=(
            "2-D relaxed scan around a TS guess. Sweeps two reactive bond "
            "lengths INDEPENDENTLY on a grid; the argmax tells you whether "
            "the 1-D scan path passes through the saddle or misses it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="PDB used for atom indexing (usually the same geometry "
                        "as --ts-guess).")
    p.add_argument("--ts-guess", required=True,
                   help="TS-guess PDB centred at the grid center.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--bond-a", required=True,
                   metavar="ATOMA,ATOMB",
                   help="First reactive bond (tokens: NAME.RESNAME[.RESID], "
                        "1-based serial, or 0:idx).")
    p.add_argument("--bond-b", required=True,
                   metavar="ATOMC,ATOMD",
                   help="Second reactive bond.")
    p.add_argument("--grid", default="3x3",
                   help="Grid shape NxM (default 3x3). Use 5x5 for a finer "
                        "scan when the 1-D path looks ambiguous.")
    p.add_argument("--delta-d", type=float, default=0.20,
                   help="Half-width of the sweep in each direction (Å). "
                        "Default 0.20. Bump to 0.30-0.40 if argmax lands on "
                        "the edge.")
    p.add_argument("--delta-d-a", type=float, default=None,
                   help="Per-axis override for bond A (defaults to --delta-d).")
    p.add_argument("--delta-d-b", type=float, default=None,
                   help="Per-axis override for bond B.")
    p.add_argument("--boundary-fix-preset", default=None,
                   choices=["ca-only", "backbone", "backbone-water", "none"])
    p.add_argument("--fix", nargs="+", default=None)
    p.add_argument("--free", nargs="+", default=None)
    p.add_argument("--model", default="mace-omol")
    p.add_argument("--head", default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--charge-ledger", default=None,
                   help="Optional ledger.yaml.")
    p.add_argument("--charge", type=int, default=None)
    p.add_argument("--spin", type=int, default=None)
    p.add_argument("--fmax", type=float, default=0.05,
                   help="Per-grid-point optimizer convergence. Looser than the "
                        "release stage on purpose — the FixBondLengths "
                        "constraint is what holds the geometry.")
    p.add_argument("--max-steps", type=int, default=250,
                   help="Per-grid-point optimizer iteration cap.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the heatmap PNG (matplotlib must still install).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    bond_a = tuple(s.strip() for s in args.bond_a.split(",", 1))
    bond_b = tuple(s.strip() for s in args.bond_b.split(",", 1))
    if len(bond_a) != 2 or len(bond_b) != 2:
        raise SystemExit("--bond-a / --bond-b must be 'A,B'")

    grid = _parse_grid(args.grid)

    ledger = None
    if args.charge_ledger:
        from quantum_engine.ops.charge_ledger import load_ledger
        ledger = load_ledger(args.charge_ledger)

    res = scan_2d_around(
        args.input,
        args.ts_guess,
        out_dir=args.out,
        bond_a=bond_a,
        bond_b=bond_b,
        grid=grid,
        delta_d=args.delta_d,
        delta_d_a=args.delta_d_a,
        delta_d_b=args.delta_d_b,
        boundary_fix_preset=args.boundary_fix_preset,
        fix_specs=args.fix,
        free_specs=args.free,
        model=args.model,
        head=args.head,
        device=args.device,
        charge_ledger=ledger,
        cli_charge=args.charge,
        cli_spin=args.spin,
        fmax=args.fmax,
        max_steps=args.max_steps,
        write_plot=not args.no_plot,
    )
    print(json.dumps({
        "status": res.status,
        "argmax": res.argmax_indices,
        "argmin": res.argmin_indices,
        "summary": res.summary_path,
        "plot": res.plot_path,
        "warnings": res.warnings,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
