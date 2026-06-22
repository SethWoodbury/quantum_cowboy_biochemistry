"""Higher-level reaction-coordinate scans (bond-difference / two-sided / auto).

``ops/scan.py`` drives ONE internal coordinate. Multi-bond reactions (SN2, most
enzymatic chemistry) need a coordinate that couples a forming and a breaking
bond. This module adds three modes on top of ``scan`` + ``cv_spring`` and the
same constrained-relaxation idea:

- ``single``          — one bond/angle/dihedral (delegates to ``ops.scan.run``).
- ``two-sided``       — drive a forming bond and a breaking bond SYNCHRONOUSLY
                        (paired targets, two ``FixBondLength`` constraints).
- ``bond-difference`` — drive ``s = d(P,LG) - d(P,nuc)`` (two bonds sharing a
                        central atom) via the ``cv_spring`` restraint stepped
                        across a range — robust where a hard ``FixInternals``
                        combo fails on the near-collinear SN2 geometry.
- ``auto``            — pick the mode from a resolved ``ReactionSpec``: a CV
                        bond-difference → ``bond-difference``; one changing bond
                        → ``single``; two changing bonds → ``two-sided``.

These are endpoint-generation + 1D-PES diagnostics (a TS *guess* lives at the
profile peak), NOT the acceptance authority — the saddle+Hessian+IRC gate is.
Reaction-agnostic: all atom indices come from the caller / the ReactionSpec.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixBondLength

from quantum_engine.units import EV_TO_KCAL

log = logging.getLogger("quantum_engine.ops.scan_modes")

_MODES = ("single", "two-sided", "bond-difference", "auto")


def _relax(atoms, *, fmax, max_opt_steps, backend, logfile) -> int:
    """Relax with the configured backend; return n_steps. Mirrors ops.scan."""
    if backend == "ase-lbfgs":
        from ase.optimize import LBFGS  # noqa: PLC0415
        opt = LBFGS(atoms, logfile=str(logfile))
        opt.run(fmax=fmax, steps=max_opt_steps)
        return int(getattr(opt, "nsteps", -1))
    from quantum_engine.opt import make_optimizer  # noqa: PLC0415
    res = make_optimizer(backend, fmax=fmax, max_steps=max_opt_steps,
                         logfile=logfile).run(atoms)
    return int(res.n_steps)


def _pick_ts_guess(energies):
    """Pick the TS-guess frame from a relaxed reactant->product scan profile.

    Returns ``(ts_idx, reactant_basin_idx, barrierless)``.

    In a *relaxed* scan the two endpoints are basins by construction (the scan
    drives the reaction coordinate while everything else minimizes), so the
    global energy maximum can land on a strained endpoint when the scan range
    overshoots a basin. The transition state is therefore the highest *interior*
    local maximum (a frame at least as high as both neighbours); the forward
    barrier is measured from the lowest frame on its reactant side.

    If there is no interior local maximum the profile is monotonic — barrierless
    along this coordinate within the scanned range — and we fall back to the
    global argmax with ``barrierless=True`` so callers can warn / widen the range.
    """
    e = np.asarray(energies, dtype=float)
    n = e.size
    if n < 3:
        return int(np.argmax(e)), int(np.argmin(e)), True
    interior = [i for i in range(1, n - 1) if e[i] >= e[i - 1] and e[i] >= e[i + 1]]
    if not interior:
        gi = int(np.argmax(e))
        return gi, (int(np.argmin(e[: gi + 1])) if gi > 0 else 0), True
    ts = max(interior, key=lambda i: e[i])
    return ts, int(np.argmin(e[: ts + 1])), False


def _assemble(values, energies, relax_steps, frames, outdir, *, label, meta):
    """Write csv/xyz/plot/summary and build the standard scan result dict."""
    from ase.io import write as ase_write  # noqa: PLC0415
    outdir = Path(outdir)
    energies = np.asarray(energies, dtype=float)
    rel_kcal = (energies - energies.min()) * EV_TO_KCAL

    csv_path = outdir / f"{label}.csv"
    with open(csv_path, "w") as f:
        f.write("step,coord_value,energy_eV,energy_rel_kcal,relax_steps\n")
        for i, (v, e, rel, n) in enumerate(zip(values, energies, rel_kcal, relax_steps)):
            f.write(f"{i},{v:.5f},{e:.8f},{rel:.4f},{n}\n")

    xyz_path = outdir / f"{label}-trajectory.xyz"
    ase_write(str(xyz_path), frames, format="extxyz")

    png_path = None
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(values, rel_kcal, "o-", markersize=6)
        ax.set_xlabel(meta.get("xlabel", "coordinate"))
        ax.set_ylabel("Relative energy (kcal/mol)")
        ax.set_title(meta.get("title", label))
        ax.grid(alpha=0.3)
        png_path = outdir / f"{label}.png"
        fig.tight_layout(); fig.savefig(png_path, dpi=150); plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        log.warning("  plot failed: %s", exc)

    ts_idx, reactant_basin_idx, barrierless = _pick_ts_guess(energies)
    global_max_idx = int(np.argmax(energies))
    barrier_kcal = float((energies[ts_idx] - energies[reactant_basin_idx]) * EV_TO_KCAL)
    result = {
        "status": "completed",
        "mode": meta.get("mode"),
        "atoms": frames[ts_idx].copy(),          # TS *guess* = highest interior maximum
        "ts_guess": frames[ts_idx].copy(),
        "ts_guess_idx": ts_idx,
        "reactant_basin_idx": reactant_basin_idx,
        "barrierless": barrierless,
        "coord_values": [float(v) for v in values],
        "energies_eV": energies.tolist(),
        "relative_energies_kcal": rel_kcal.tolist(),
        "max_energy_idx": global_max_idx,        # global argmax (may be an endpoint/basin)
        "max_coord_value": float(values[global_max_idx]),
        "ts_coord": float(values[ts_idx]),
        "barrier_kcal": barrier_kcal,            # forward: E(TS guess) - E(reactant basin)
        "energy_span_kcal": float(rel_kcal[global_max_idx]),
        "frames": frames,
        "outputs": {"csv": str(csv_path), "trajectory": str(xyz_path),
                    "plot": str(png_path) if png_path else None},
    }
    summary = {**meta, "barrier_kcal": barrier_kcal, "ts_coord": result["ts_coord"],
               "ts_guess_idx": ts_idx, "reactant_basin_idx": reactant_basin_idx,
               "barrierless": barrierless, "max_coord_value": result["max_coord_value"],
               "max_energy_idx": global_max_idx, "energy_span_kcal": result["energy_span_kcal"],
               "n_steps": len(values)}
    (outdir / f"{label}-summary.json").write_text(json.dumps(summary, indent=2))
    result["outputs"]["summary"] = str(outdir / f"{label}-summary.json")
    if barrierless:
        log.info("scan-%s done: NO interior barrier in range (monotonic) — span %.2f kcal/mol; "
                 "widen/shift the range or inspect the profile",
                 meta.get("mode"), result["energy_span_kcal"])
    else:
        log.info("scan-%s done: forward barrier = %.2f kcal/mol ; TS guess at coord = %.3f (frame %d)",
                 meta.get("mode"), barrier_kcal, result["ts_coord"], ts_idx)
    return result


def run(
    atoms: Atoms,
    calculator=None,
    *,
    mode: str = "auto",
    outdir: str | Path = ".",
    constraint=None,
    # single
    coord_type: str = "bond",
    atom_indices: list[int] | None = None,
    start: float | None = None,
    end: float | None = None,
    # two-sided  (bond_a forming, bond_b breaking)
    bond_a: tuple[int, int] | None = None,
    bond_b: tuple[int, int] | None = None,
    a_start: float | None = None, a_end: float | None = None,
    b_start: float | None = None, b_end: float | None = None,
    # bond-difference  (s = d(diff_bonds[0]) - d(diff_bonds[1]); bonds share a centre)
    diff_bonds: tuple[tuple[int, int], tuple[int, int]] | None = None,
    s_start: float | None = None, s_end: float | None = None,
    spring_k: float = 5.0,
    # auto
    reaction=None,
    # shared
    n_steps: int = 15,
    relax_other: bool = True,
    fmax: float = 0.05,
    max_opt_steps: int = 300,
    backend: str = "ase-lbfgs",
    **kwargs,
) -> dict:
    """Run a reaction-coordinate scan in the chosen ``mode`` (see module docstring).

    ``auto`` requires ``reaction`` (a ``ResolvedReaction`` from
    ``ReactionSpec.resolve(...)``) and resolves the mode + atom indices, while
    the scan RANGE stays the caller's (reaction-specific, not derivable). Returns
    the standard scan result dict plus ``ts_guess``/``ts_guess_idx`` (profile peak).
    """
    if mode not in _MODES:
        raise ValueError(f"mode={mode!r} not in {_MODES}")
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    if atoms.calc is None:
        if calculator is None:
            raise ValueError("Need a calculator")
        atoms.calc = calculator
    base = (constraint if isinstance(constraint, list)
            else [constraint] if constraint is not None else [])

    if mode == "auto":
        mode, auto_kw = _resolve_auto(reaction)
        log.info("scan_modes auto → mode=%s", mode)
        bond_a = bond_a or auto_kw.get("bond_a")
        bond_b = bond_b or auto_kw.get("bond_b")
        diff_bonds = diff_bonds or auto_kw.get("diff_bonds")
        atom_indices = atom_indices or auto_kw.get("atom_indices")

    if mode == "single":
        from quantum_engine.ops import scan  # noqa: PLC0415
        if atom_indices is None or start is None or end is None:
            raise ValueError("single mode needs atom_indices + start + end")
        res = scan.run(atoms, calculator, outdir=outdir, constraint=constraint,
                       coord_type=coord_type, atom_indices=list(atom_indices),
                       start=start, end=end, n_steps=n_steps,
                       relax_other=relax_other, fmax=fmax,
                       max_opt_steps=max_opt_steps, backend=backend)
        res["mode"] = "single"
        res.setdefault("ts_guess_idx", res.get("max_energy_idx"))
        res.setdefault("ts_guess", res.get("atoms"))
        return res

    if mode == "two-sided":
        if not (bond_a and bond_b):
            raise ValueError("two-sided mode needs bond_a and bond_b")
        bond_a = tuple(bond_a); bond_b = tuple(bond_b)
        a_lo = a_start if a_start is not None else atoms.get_distance(*bond_a)
        b_lo = b_start if b_start is not None else atoms.get_distance(*bond_b)
        a_hi = a_end if a_end is not None else a_lo
        b_hi = b_end if b_end is not None else b_lo
        paired = list(zip(np.linspace(a_lo, a_hi, n_steps),
                          np.linspace(b_lo, b_hi, n_steps)))
        meta = {"mode": "two-sided", "bond_a": list(bond_a), "bond_b": list(bond_b),
                "xlabel": f"d{tuple(bond_a)} (Å)", "title": "two-sided scan"}
        return _two_sided_scan(atoms, outdir, base, paired, bond_a, bond_b,
                               relax_other, fmax, max_opt_steps, backend, meta)

    if mode == "bond-difference":
        if not diff_bonds or s_start is None or s_end is None:
            raise ValueError("bond-difference mode needs diff_bonds + s_start + s_end")
        b0, b1 = tuple(diff_bonds[0]), tuple(diff_bonds[1])
        shared = set(b0) & set(b1)
        if len(shared) != 1:
            raise ValueError(
                "bond-difference needs the two bonds to share exactly one atom "
                f"(the central atom); got {b0} and {b1}")
        center = shared.pop()
        # s = d(center,breaking) - d(center,forming): breaking is the first bond's
        # non-shared atom, forming the second's. (diff_bonds order sets which is which.)
        breaking = b0[1] if b0[0] == center else b0[0]
        forming = b1[1] if b1[0] == center else b1[0]
        return _bond_difference_scan(
            atoms, outdir, base, center, breaking, forming,
            np.linspace(s_start, s_end, n_steps), spring_k,
            relax_other, fmax, max_opt_steps, backend)

    raise ValueError(f"unhandled mode {mode!r}")


def _two_sided_scan(atoms, outdir, base, paired, bond_a, bond_b, relax_other,
                    fmax, max_opt_steps, backend, meta):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    frames, energies, relax_steps, xvals = [], [], [], []
    for i, (ta, tb) in enumerate(paired):
        atoms.set_distance(bond_a[0], bond_a[1], ta, fix=0.5)
        atoms.set_distance(bond_b[0], bond_b[1], tb, fix=0.5)
        atoms.set_constraint(base + [FixBondLength(*bond_a), FixBondLength(*bond_b)])
        n_opt = _relax(atoms, fmax=fmax, max_opt_steps=max_opt_steps, backend=backend,
                       logfile=outdir / f"scan-twosided-opt{i:03d}.log") if relax_other else 0
        e = float(atoms.get_potential_energy())   # hard constraints do no work
        fr = atoms.copy(); fr.info["scan_value"] = float(ta)
        fr.info["energy_eV"] = e; fr.info["bond_b_value"] = float(tb)
        frames.append(fr); energies.append(e); relax_steps.append(n_opt); xvals.append(float(ta))
        log.info("  step %d/%d: d_a=%.3f d_b=%.3f  E=%.6f eV  relax_steps=%d",
                 i + 1, len(paired), ta, tb, e, n_opt)
    atoms.set_constraint(base)
    return _assemble(xvals, energies, relax_steps, frames, outdir,
                     label="scan-twosided", meta=meta)


def _bond_difference_scan(atoms, outdir, base, center, breaking, forming, s_vals,
                          spring_k, relax_other, fmax, max_opt_steps, backend):
    """Drive s = d(center,breaking) - d(center,forming) with the cv_spring restraint, stepped.

    The energy recorded is the BARE PES energy (the soft restraint is removed
    before each measurement), and the x-axis is the ACHIEVED s at the relaxed
    geometry — so the profile is honest even where the restraint slightly
    undershoots the target.
    """
    from quantum_engine.mlff.cv_spring import BondDifferenceCVSpring  # noqa: PLC0415
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    frames, energies, relax_steps, achieved = [], [], [], []
    for i, s_t in enumerate(s_vals):
        spring = BondDifferenceCVSpring(center_idx=center, breaking_idx=breaking,
                                        forming_idx=forming, k=spring_k,
                                        s_target=float(s_t), fmax=None, mode="both")
        atoms.set_constraint(base + [spring])
        n_opt = _relax(atoms, fmax=fmax, max_opt_steps=max_opt_steps, backend=backend,
                       logfile=outdir / f"scan-bonddiff-opt{i:03d}.log") if relax_other else 0
        # bare PES energy (exclude the restraint contribution) + achieved s
        atoms.set_constraint(base)
        e = float(atoms.get_potential_energy())
        s_now = float(atoms.get_distance(center, breaking) - atoms.get_distance(center, forming))
        fr = atoms.copy(); fr.info["scan_value"] = s_now; fr.info["energy_eV"] = e
        fr.info["s_target"] = float(s_t)
        frames.append(fr); energies.append(e); relax_steps.append(n_opt); achieved.append(s_now)
        log.info("  step %d/%d: s_target=%.3f s_achieved=%.3f  E=%.6f eV  relax_steps=%d",
                 i + 1, len(s_vals), float(s_t), s_now, e, n_opt)
    atoms.set_constraint(base)
    meta = {"mode": "bond-difference", "center": center, "breaking": breaking,
            "forming": forming, "spring_k": spring_k,
            "xlabel": "s = d(center,breaking) − d(center,forming) (Å)",
            "title": "bond-difference scan"}
    return _assemble(achieved, energies, relax_steps, frames, outdir,
                     label="scan-bonddiff", meta=meta)


def _resolve_auto(reaction):
    """Pick a concrete mode + atom indices from a ResolvedReaction."""
    if reaction is None:
        raise ValueError("mode='auto' needs reaction= (a ResolvedReaction)")
    cv = getattr(reaction, "cv_bond_difference", None)
    if cv is not None:
        p, nuc, lg = cv
        # s = d(P,LG) - d(P,nuc): diff_bonds share the central atom P
        return "bond-difference", {"diff_bonds": ((p, lg), (p, nuc))}
    forming = list(getattr(reaction, "forming", []))
    breaking = list(getattr(reaction, "breaking", []))
    changing = forming + breaking
    if len(changing) == 1:
        return "single", {"atom_indices": list(changing[0])}
    if forming and breaking:
        return "two-sided", {"bond_a": tuple(forming[0]), "bond_b": tuple(breaking[0])}
    if len(changing) >= 2:
        return "two-sided", {"bond_a": tuple(changing[0]), "bond_b": tuple(changing[1])}
    raise ValueError("auto: reaction has no changing bonds / CV to scan")


__all__ = ["run"]
