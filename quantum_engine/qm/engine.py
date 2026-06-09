"""QM-native engine gateway — the third plug-and-play axis (whole-step QM).

Where an energy function is a *calculator* (per-step forces driven by OUR
optimizers), an **engine** routes the whole TS step to the QC package's OWN
optimizer — the right move for DFT, where per-step ASE driving wastes the SCF
restart and the package's native NEB-TS/OptTS is far more efficient.

Each engine is registered with ``register_engine(name, runner)`` and implements
one contract:

    runner(reaction, ctx, *, entry, outdir, reactant=, product=, ts_guess=, ...)
        -> result dict (ts / energy_eV_ts / imag_freq_cm / n_imag / gates / outputs)

ORCA ships today (native NEB-TS for R+P, OptTS+Freq for a guess). Turbomole /
Gaussian slot in later via the same one-liner. ORCA is NOT in the cowboy-qc container,
so the runner shells out where ORCA is installed (a host/node); the input-gen +
output parser (``qm/orca.py``) are unit-tested without it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from quantum_engine.logging_utils import get_logger
from quantum_engine.ops.gates import Gate, GateReport
from quantum_engine.registry import Registry

log = get_logger("qm.engine")

ENGINES: Registry = Registry("engine")


def register_engine(name: str, runner=None, *, aliases: tuple[str, ...] = (),
                    overwrite: bool = False):
    """Register a QM-native engine runner (decorator or imperative)."""
    return ENGINES.register(name, runner, aliases=aliases, overwrite=overwrite)


def make_engine(name: str):
    """Return the engine runner registered under ``name`` (name/alias)."""
    if name not in ENGINES:
        raise ValueError(f"Unknown engine {name!r}. Choices: {ENGINES.names()}")
    return ENGINES.get(name)


def _write_xyz(atoms, path: Path) -> Path:
    from ase.io import write as ase_write  # noqa: PLC0415
    path = Path(path)
    ase_write(str(path), atoms, format="xyz")
    return path


def _atoms_from_geometry(geometry) -> Any:
    """(symbols, positions) -> ASE Atoms (or None)."""
    if not geometry:
        return None
    from ase import Atoms  # noqa: PLC0415
    syms, pos = geometry
    return Atoms(symbols=syms, positions=pos)


def _qm_method_basis(ctx, method, basis):
    """Resolve method/basis: explicit args, else ctx.extra, else a 'method/basis'
    model alias, else defaults."""
    m = method or ctx.extra.get("method")
    b = basis or ctx.extra.get("basis")
    if (m is None or b is None) and ctx.model and "/" in ctx.model:
        mm, bb = ctx.model.split("/", 1)
        m = m or mm
        b = b or bb
    return (m or "B3LYP"), (b or "def2-SVP")


def run_orca_engine(
    reaction, ctx, *, entry: str, outdir: str | Path,
    reactant=None, product=None, ts_guess=None,
    method: str | None = None, basis: str | None = None,
    n_images: int = 8, nproc: int = 8, maxcore: int = 4000,
    solvent: str | None = None, orca_bin: str | None = None,
    execute: bool = True, **kwargs,
) -> dict:
    """ORCA native TS engine: NEB-TS (R+P) or OptTS+Freq (guess).

    ``execute=False`` writes the inputs and returns without running ORCA (useful
    for generating a job to ``sbatch``, and for testing). Returns a ts_entry-style
    result dict plus ``outputs`` (input/out paths) and a GateReport (gates.json).
    """
    from quantum_engine.qm import orca  # noqa: PLC0415
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    meth, bas = _qm_method_basis(ctx, method, basis)
    report = GateReport(step="orca_engine")

    if entry == "reactant-only":
        raise ValueError(
            "ORCA native engine needs both endpoints. For reactant-only, generate "
            "a product first (MLFF ts_entry reactant-only / scan_modes), then run "
            "engine='orca' with entry='reactant-product'.")

    if entry == "reactant-product":
        if reactant is None or product is None:
            raise ValueError("engine=orca reactant-product needs reactant= and product=")
        r_xyz = _write_xyz(reactant, outdir / "reactant.xyz")
        p_xyz = _write_xyz(product, outdir / "product.xyz")
        inp = orca.write_orca_nebts_input(
            r_xyz, p_xyz, outdir / "nebts.inp", charge=ctx.charge,
            multiplicity=ctx.multiplicity, method=meth, basis=bas, n_images=n_images,
            freq=True, nproc=nproc, maxcore=maxcore, solvent=solvent)
        job = "neb-ts"
    elif entry == "ts-guess":
        if ts_guess is None:
            raise ValueError("engine=orca ts-guess needs ts_guess=")
        g_xyz = _write_xyz(ts_guess, outdir / "ts_guess.xyz")
        inp = orca.write_orca_input(
            g_xyz, outdir / "optts.inp", charge=ctx.charge,
            multiplicity=ctx.multiplicity, method=meth, basis=bas, job_type="ts+freq",
            nproc=nproc, maxcore=maxcore, solvent=solvent)
        job = "ts+freq"
    else:
        raise ValueError(f"engine=orca: unknown entry {entry!r}")

    log.info("ORCA engine: entry=%s job=%s method=%s basis=%s -> %s",
             entry, job, meth, bas, inp.name)

    if not execute:
        report.add(Gate("orca_input_written", "PASS",
                        detail=f"input ready ({inp.name}); execute=False"))
        report.emit(outdir)
        return {"status": "prepared", "entry": entry, "engine": "orca",
                "ts": None, "energy_eV_ts": None, "imag_freq_cm": None,
                "n_imag": None, "gates_overall": report.overall,
                "outputs": {"input": str(inp), "outdir": str(outdir),
                            "gates_json": str(outdir / "gates.json")}}

    res = orca.run_orca(inp, orca_bin=orca_bin)
    ts_atoms = _atoms_from_geometry(res.get("geometry"))
    if entry == "reactant-product":
        ts_xyz = orca.find_nebts_ts_xyz(outdir, inp.stem)
        if ts_xyz is not None:
            from ase.io import read as ase_read  # noqa: PLC0415
            ts_atoms = ase_read(str(ts_xyz))

    n_imag = res.get("n_imag")
    imag = res.get("imag_freq_cm")
    report.add(Gate("orca_returncode",
                    "PASS" if res.get("returncode") == 0 else "FAIL",
                    detail="ORCA exit status", value=res.get("returncode")))
    report.add(Gate("n_imag", "PASS" if n_imag == 1 else "FAIL",
                    detail="exactly one imaginary mode", value=n_imag, threshold=1))
    report.add(Gate("imag_freq_cm",
                    "PASS" if (imag is not None and imag < -50.0) else "FAIL",
                    detail="imag freq < -50 cm^-1", value=imag, threshold=-50.0))
    report.emit(outdir)

    return {
        "status": "failed" if report.critical_fail else "converged",
        "entry": entry, "engine": "orca",
        "ts": ts_atoms,
        "energy_eV_ts": res.get("energy_eV"),
        "imag_freq_cm": imag, "n_imag": n_imag,
        "frequencies_cm": res.get("frequencies_cm"),
        "gates_overall": report.overall, "critical_fail": report.critical_fail,
        "outputs": {"input": str(inp), "out": res.get("out_path"),
                    "outdir": str(outdir), "gates_json": str(outdir / "gates.json")},
    }


register_engine("orca", run_orca_engine)


__all__ = ["ENGINES", "register_engine", "make_engine", "run_orca_engine"]
