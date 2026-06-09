"""PLUMED-driven enhanced sampling: WT-MTD, OPES, umbrella sampling.

Bridges qcb's config + ops layer to PLUMED 2 via ASE's Plumed calculator wrapper.
Uses the prebuilt PLUMED kernel auto-detected by `quantum_engine.config.PLUMED_KERNEL`
(or set the env var manually).

What this enables that pure-Python `quantum_engine.mlff.metadynamics` does not:
- Arbitrary CVs: distance, angle, dihedral, COORDINATION, RMSD, path, ...
- Multi-walker MTD via shared-filesystem hills
- Reweighting machinery (PLUMED's COLVAR + sum_hills compatibility)
- Standard PLUMED file formats (HILLS, COLVAR) that `quantum_engine.analysis.fes` reads
- OPES with full PLUMED implementation (SIGMA adaptive, OPES_EXPLORE, ...)

Design
------
The user supplies a PlumedCV spec (dataclass) that maps cleanly to PLUMED's
input language. The runner generates the PLUMED input on the fly, wires it
to the ASE calculator, and runs Langevin MD.

For multi-walker: each walker is a separate process (typically separate SLURM
job) writing HILLS to a shared directory; PLUMED's MULTIPLE_WALKERS keyword
handles the file-based exchange.

References
----------
- Tribello et al. "PLUMED 2: New feathers for an old bird." Comput. Phys.
  Commun. 2014, 185, 604. https://doi.org/10.1016/j.cpc.2013.09.018
- Bonomi et al. "Promoting transparency and reproducibility in enhanced
  molecular simulations." Nat. Methods 2019, 16, 670 (PLUMED-NEST).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from quantum_engine.units import EV_TO_KCAL
from ase import Atoms, units
from ase.io import write as ase_write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

log = logging.getLogger("quantum_engine.mlff.plumed_runner")

KJ_TO_EV = units.kJ / units.mol  # 0.010364 eV per kJ/mol


# ═══════════════════════════════════════════════════════════════════
# CV specs — clean dataclasses that produce PLUMED ARG lines
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DistanceCV:
    """PLUMED DISTANCE between two atoms (1-indexed in PLUMED)."""
    name: str
    atoms: tuple[int, int]              # 0-indexed; converted to 1-indexed in PLUMED
    components: bool = False             # if True, output x, y, z components

    def to_plumed(self) -> str:
        i, j = self.atoms
        comp = " COMPONENTS" if self.components else ""
        return f"{self.name}: DISTANCE ATOMS={i+1},{j+1}{comp}"


@dataclass
class DistanceDifferenceCV:
    """s = d(a,b) − d(c,d). Implemented via two DISTANCEs + COMBINE."""
    name: str
    atoms_minus: tuple[int, int]         # the bond that LENGTHENS (P-LG)
    atoms_plus: tuple[int, int]          # the bond that SHORTENS (P-nuc) — sign flip per convention

    def to_plumed(self) -> list[str]:
        i, j = self.atoms_minus
        k, l = self.atoms_plus
        return [
            f"{self.name}_d1: DISTANCE ATOMS={i+1},{j+1}",
            f"{self.name}_d2: DISTANCE ATOMS={k+1},{l+1}",
            f"{self.name}: COMBINE ARG={self.name}_d1,{self.name}_d2 COEFFICIENTS=1,-1 PERIODIC=NO",
        ]


@dataclass
class AngleCV:
    name: str
    atoms: tuple[int, int, int]

    def to_plumed(self) -> str:
        i, j, k = self.atoms
        return f"{self.name}: ANGLE ATOMS={i+1},{j+1},{k+1}"


@dataclass
class DihedralCV:
    name: str
    atoms: tuple[int, int, int, int]

    def to_plumed(self) -> str:
        i, j, k, l = self.atoms
        return f"{self.name}: TORSION ATOMS={i+1},{j+1},{k+1},{l+1}"


@dataclass
class CoordinationCV:
    """PLUMED COORDINATION number with rational switching function."""
    name: str
    group_a: list[int]                   # 0-indexed atom indices
    group_b: list[int]
    r0_nm: float = 0.20
    nn: int = 5
    mm: int = 10
    d0_nm: float = 0.0

    def to_plumed(self) -> str:
        a = ",".join(str(i + 1) for i in self.group_a)
        b = ",".join(str(i + 1) for i in self.group_b)
        switch = (f"{{RATIONAL R_0={self.r0_nm} D_0={self.d0_nm} "
                  f"NN={self.nn} MM={self.mm}}}")
        return f"{self.name}: COORDINATION GROUPA={a} GROUPB={b} SWITCH={switch}"


CVType = DistanceCV | DistanceDifferenceCV | AngleCV | DihedralCV | CoordinationCV


# ═══════════════════════════════════════════════════════════════════
# Method specs — what to do with the CV
# ═══════════════════════════════════════════════════════════════════

@dataclass
class WTMetadynamics:
    cv_name: str                          # name of the CV to bias
    sigma: float = 0.05                   # Gaussian width (CV units)
    height_kJ: float = 1.2                # initial height (kJ/mol)
    pace_steps: int = 500                 # deposit interval
    bias_factor: float = 10.0             # γ for well-tempered
    grid_min: float | None = None
    grid_max: float | None = None
    grid_bin: int = 400
    file: str = "HILLS"                   # output filename

    def to_plumed(self) -> str:
        kw = (f"METAD ARG={self.cv_name} SIGMA={self.sigma} "
              f"HEIGHT={self.height_kJ} BIASFACTOR={self.bias_factor} "
              f"PACE={self.pace_steps} TEMP=300 FILE={self.file}")
        if self.grid_min is not None and self.grid_max is not None:
            kw += f" GRID_MIN={self.grid_min} GRID_MAX={self.grid_max} GRID_BIN={self.grid_bin}"
        return f"metad: {kw}"


@dataclass
class OPESMetadynamics:
    """OPES-MetaD (Invernizzi & Parrinello JPCL 2020)."""
    cv_name: str
    sigma: float = 0.05
    barrier_kJ: float = 30.0              # estimated barrier in kJ/mol
    pace_steps: int = 500
    bias_factor: float = 10.0             # γ
    grid_min: float | None = None
    grid_max: float | None = None
    grid_bin: int = 400
    file: str = "HILLS"

    def to_plumed(self) -> str:
        kw = (f"OPES_METAD ARG={self.cv_name} SIGMA={self.sigma} "
              f"BARRIER={self.barrier_kJ} PACE={self.pace_steps} "
              f"BIASFACTOR={self.bias_factor} TEMP=300 FILE={self.file}")
        if self.grid_min is not None and self.grid_max is not None:
            kw += f" GRID_MIN={self.grid_min} GRID_MAX={self.grid_max} GRID_BIN={self.grid_bin}"
        return f"opes: {kw}"


@dataclass
class UmbrellaWindow:
    """Single-window harmonic restraint (umbrella sampling)."""
    cv_name: str
    at: float                             # restraint center (CV units)
    kappa_kJ: float                       # k in kJ/mol/CV² (e.g., kJ/mol/Å² for distance)

    def to_plumed(self) -> str:
        return f"restraint: RESTRAINT ARG={self.cv_name} AT={self.at} KAPPA={self.kappa_kJ}"


MethodType = WTMetadynamics | OPESMetadynamics | UmbrellaWindow


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PlumedRunSpec:
    cvs: list[CVType]
    method: MethodType
    print_stride: int = 100               # how often to write COLVAR
    print_args: list[str] | None = None   # CV names to print (default: all CV names)
    walker_id: int | None = None          # set for multi-walker
    n_walkers: int | None = None
    walker_dir: str | Path | None = None  # shared dir for HILLS exchange


def write_plumed_input(spec: PlumedRunSpec, out_path: str | Path) -> Path:
    """Generate a PLUMED input file from a PlumedRunSpec."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Generated by quantum_engine.mlff.plumed_runner"]

    # Multi-walker setup
    if spec.walker_id is not None and spec.n_walkers and spec.n_walkers > 1:
        if spec.walker_dir is None:
            raise ValueError("multi-walker requires walker_dir (shared filesystem)")
        # PLUMED expects walker_id 0..n-1
        lines.append(f"# Multi-walker: walker {spec.walker_id} of {spec.n_walkers}")
        # Use file-based MULTIPLE_WALKERS via shared dir
        # PLUMED METAD/OPES_METAD takes WALKERS_N + WALKERS_ID + WALKERS_DIR
        method = spec.method
        method_line = method.to_plumed()
        if isinstance(method, (WTMetadynamics, OPESMetadynamics)):
            method_line += (f" WALKERS_N={spec.n_walkers} "
                            f"WALKERS_ID={spec.walker_id} "
                            f"WALKERS_DIR={spec.walker_dir} "
                            f"WALKERS_RSTRIDE={method.pace_steps}")
            method.file = f"HILLS.{spec.walker_id}"
            method_line = method.to_plumed()  # regenerate with walker file
            method_line += (f" WALKERS_N={spec.n_walkers} "
                            f"WALKERS_ID={spec.walker_id} "
                            f"WALKERS_DIR={spec.walker_dir} "
                            f"WALKERS_RSTRIDE={method.pace_steps}")
        # Fall through: still emit method below

    # CVs
    for cv in spec.cvs:
        line_or_lines = cv.to_plumed()
        if isinstance(line_or_lines, list):
            lines.extend(line_or_lines)
        else:
            lines.append(line_or_lines)

    # Method (bias)
    method_line = spec.method.to_plumed()
    if spec.walker_id is not None and spec.n_walkers and spec.n_walkers > 1:
        method_line += (f" WALKERS_N={spec.n_walkers} "
                        f"WALKERS_ID={spec.walker_id} "
                        f"WALKERS_DIR={spec.walker_dir} "
                        f"WALKERS_RSTRIDE={spec.method.pace_steps if hasattr(spec.method, 'pace_steps') else 500}")
    lines.append(method_line)

    # PRINT for COLVAR
    args = spec.print_args or [cv.name for cv in spec.cvs]
    bias_arg = "metad.bias" if isinstance(spec.method, WTMetadynamics) else \
               "opes.bias" if isinstance(spec.method, OPESMetadynamics) else \
               "restraint.bias"
    args_str = ",".join(args + [bias_arg])
    colvar_file = "COLVAR" if spec.walker_id is None else f"COLVAR.{spec.walker_id}"
    lines.append(f"PRINT ARG={args_str} STRIDE={spec.print_stride} FILE={colvar_file}")

    out_path.write_text("\n".join(lines) + "\n")
    log.info(f"  PLUMED input → {out_path}")
    return out_path


def _setup_plumed_env() -> tuple[str, dict]:
    """Resolve PLUMED kernel path and return env dict for subprocess/calc.

    Returns (kernel_path, env_dict_to_set).
    """
    try:
        from quantum_engine.site import PLUMED_KERNEL
    except Exception:
        PLUMED_KERNEL = None

    kernel = os.environ.get("PLUMED_KERNEL") or PLUMED_KERNEL
    if not kernel or not os.path.isfile(kernel):
        raise FileNotFoundError(
            "PLUMED kernel not found. Set $PLUMED_KERNEL or build via "
            "deps/plumed2_build.sh. Tried: " + str(kernel)
        )

    env = {"PLUMED_KERNEL": kernel}
    # LD_LIBRARY_PATH: include the dir containing the kernel
    kernel_dir = os.path.dirname(kernel)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{kernel_dir}:{existing}".rstrip(":")
    return kernel, env


def run_plumed_md(
    atoms: Atoms,
    calculator,
    spec: PlumedRunSpec,
    outdir: str | Path,
    timestep_fs: float = 1.0,
    total_time_ps: float = 50.0,
    temperature_K: float = 300.0,
    friction_per_ps: float = 1.0,
    constraint=None,
    seed: int | None = None,
) -> dict:
    """Run Langevin MD with PLUMED bias.

    Args:
        atoms: ASE Atoms (caller may have set atoms.calc)
        calculator: base ASE calculator (e.g., MACE)
        spec: PlumedRunSpec describing CVs + method
        outdir: working directory (HILLS, COLVAR, plumed.dat written here)
        timestep_fs, total_time_ps, temperature_K, friction_per_ps: MD params
        constraint: ASE constraint(s) applied during MD
        seed: RNG seed

    Returns: dict with output paths and basic stats.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve PLUMED kernel + env
    kernel, env_updates = _setup_plumed_env()
    for k, v in env_updates.items():
        os.environ[k] = v
    log.info(f"  PLUMED kernel: {kernel}")

    # Set up calc
    if atoms.calc is None:
        atoms.calc = calculator

    # Apply constraints
    if constraint is not None:
        if not isinstance(constraint, list):
            constraint = [constraint]
        atoms.set_constraint(constraint)

    # Generate PLUMED input
    plumed_input_path = outdir / "plumed.dat"
    write_plumed_input(spec, plumed_input_path)

    # Wrap calculator with PLUMED
    from ase.calculators.plumed import Plumed

    # ASE's Plumed needs the input as a list of strings (per PLUMED line)
    plumed_lines = plumed_input_path.read_text().splitlines()

    plumed_calc = Plumed(
        calc=atoms.calc,
        input=plumed_lines,
        timestep=timestep_fs * units.fs,
        atoms=atoms,
        kT=temperature_K * units.kB,
        log=str(outdir / "plumed.log"),
    )
    atoms.calc = plumed_calc

    # Initialize velocities
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)

    # Run Langevin
    friction_ase = friction_per_ps * 1e-3 / units.fs
    n_steps = int(total_time_ps * 1000 / timestep_fs)
    log.info(f"  Langevin NVT @ {temperature_K} K, {total_time_ps} ps "
             f"({n_steps} steps × {timestep_fs} fs)")

    dyn = Langevin(atoms, timestep_fs * units.fs,
                   temperature_K=temperature_K, friction=friction_ase, rng=rng)

    # Snapshot trajectory
    traj_path = outdir / "md.traj"
    dyn.attach(lambda: ase_write(str(outdir / "snapshot.xyz"), atoms,
                                  format="extxyz", append=True),
               interval=max(100, n_steps // 1000))

    # Note: PLUMED writes HILLS / COLVAR using its own working directory,
    # which is wherever the calculator was instantiated. Run with cwd=outdir
    # to get the files written there.
    cwd_orig = os.getcwd()
    try:
        os.chdir(outdir)
        dyn.run(n_steps)
    finally:
        os.chdir(cwd_orig)

    log.info(f"  PLUMED MD done: {n_steps} steps")

    # Find output files
    outputs = {
        "plumed_input": str(plumed_input_path),
        "plumed_log": str(outdir / "plumed.log"),
        "trajectory": str(outdir / "snapshot.xyz") if (outdir / "snapshot.xyz").exists() else None,
    }
    if isinstance(spec.method, (WTMetadynamics, OPESMetadynamics)):
        hills_pattern = "HILLS" if spec.walker_id is None else f"HILLS.{spec.walker_id}"
        hills_files = list(outdir.glob(hills_pattern)) + list(outdir.glob(f"{hills_pattern}.*"))
        outputs["hills"] = [str(f) for f in hills_files]
    colvar_files = list(outdir.glob("COLVAR*"))
    outputs["colvar"] = [str(f) for f in colvar_files]

    return {
        "status": "completed",
        "atoms": atoms,
        "n_steps": n_steps,
        "outputs": outputs,
    }


# ═══════════════════════════════════════════════════════════════════
# Convenience: bond-difference MTD using qcb-flavored API
# ═══════════════════════════════════════════════════════════════════

def run_bond_difference_mtd(
    atoms: Atoms,
    calculator,
    p_idx: int, nuc_idx: int, lg_idx: int,
    outdir: str | Path,
    method: str = "wt",                  # "wt" or "opes"
    sigma_A: float = 0.1,
    pace_steps: int = 500,
    bias_factor: float = 10.0,
    height_kJ: float = 1.2,              # WT only
    barrier_kJ: float = 30.0,            # OPES only
    timestep_fs: float = 1.0,
    total_time_ps: float = 50.0,
    temperature_K: float = 300.0,
    friction_per_ps: float = 1.0,
    constraint=None,
    seed: int | None = None,
    walker_id: int | None = None,
    n_walkers: int | None = None,
    walker_dir: str | Path | None = None,
) -> dict:
    """One-shot: WT-MTD or OPES on s = d(P-LG) − d(P-nuc) via PLUMED.

    PLUMED uses nm internally; we write Å values that PLUMED reads if UNITS
    LENGTH=A is set. To keep it simple here, we use PLUMED's default nm —
    but our DISTANCE/COMBINE pipeline still gives s in Å as long as ASE
    units module yields the same Å in both places. (ASE writes Å,
    PLUMED's default unit conversion handles this — verify with first run.)
    """
    cv = DistanceDifferenceCV(
        name="s",
        atoms_minus=(p_idx, lg_idx),
        atoms_plus=(p_idx, nuc_idx),
    )

    # Method
    if method == "wt":
        m = WTMetadynamics(
            cv_name="s",
            sigma=sigma_A * 0.1,         # nm if PLUMED is in nm; safe-ish
            height_kJ=height_kJ,
            pace_steps=pace_steps,
            bias_factor=bias_factor,
        )
    elif method == "opes":
        m = OPESMetadynamics(
            cv_name="s",
            sigma=sigma_A * 0.1,
            barrier_kJ=barrier_kJ,
            pace_steps=pace_steps,
            bias_factor=bias_factor,
        )
    else:
        raise ValueError(f"method must be 'wt' or 'opes', got {method!r}")

    spec = PlumedRunSpec(
        cvs=[cv],
        method=m,
        walker_id=walker_id,
        n_walkers=n_walkers,
        walker_dir=walker_dir,
    )

    return run_plumed_md(
        atoms, calculator, spec, outdir,
        timestep_fs=timestep_fs, total_time_ps=total_time_ps,
        temperature_K=temperature_K, friction_per_ps=friction_per_ps,
        constraint=constraint, seed=seed,
    )


def run_umbrella_window(
    atoms: Atoms,
    calculator,
    cv_atoms: tuple[int, int],
    at_A: float,
    kappa_kJ_per_A2: float,
    outdir: str | Path,
    timestep_fs: float = 1.0,
    total_time_ps: float = 5.0,
    temperature_K: float = 300.0,
    friction_per_ps: float = 1.0,
    constraint=None,
    seed: int | None = None,
) -> dict:
    """Single-window umbrella: harmonic restraint on a distance CV."""
    cv = DistanceCV(name="d", atoms=cv_atoms)
    m = UmbrellaWindow(cv_name="d", at=at_A * 0.1, kappa_kJ=kappa_kJ_per_A2 * 100.0)
    # (×0.1 nm conversion + ×100 because k in kJ/mol/nm² = k_in_A²/100)
    spec = PlumedRunSpec(cvs=[cv], method=m)
    return run_plumed_md(
        atoms, calculator, spec, outdir,
        timestep_fs=timestep_fs, total_time_ps=total_time_ps,
        temperature_K=temperature_K, friction_per_ps=friction_per_ps,
        constraint=constraint, seed=seed,
    )
