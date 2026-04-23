"""Molecular dynamics (Langevin NVT, Velocity Verlet NVE).

Supports simple annealing protocols (linear ramp to peak temperature, then
back down). For enzyme systems, Langevin NVT at 300 K is the typical choice.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import write as ase_write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

log = logging.getLogger("qcb.ops.md")

EV_TO_KCAL = 23.0609


def run(
    atoms: Atoms,
    calculator=None,
    outdir: str | Path = ".",
    constraint=None,
    ensemble: str = "langevin_nvt",
    timestep_fs: float = 1.0,
    total_time_ps: float = 10.0,
    temperature_K: float = 300.0,
    friction_per_ps: float = 1.0,
    dump_every_fs: float = 10.0,
    anneal_peak_K: float | None = None,
    anneal_fraction: float = 0.5,
    seed: int | None = None,
    **kwargs,
) -> dict:
    """Molecular dynamics.

    Args:
        atoms, calculator, outdir, constraint: standard
        ensemble: "langevin_nvt" or "verlet_nve"
        timestep_fs: integration timestep in femtoseconds
        total_time_ps: total simulation time in picoseconds
        temperature_K: target temperature (Langevin) or initial temperature (NVE)
        friction_per_ps: Langevin friction in ps⁻¹
        dump_every_fs: trajectory dump interval
        anneal_peak_K: if not None, linearly ramp temperature from
                       initial → peak (over first anneal_fraction of run)
                       then peak → initial (remaining time)
        anneal_fraction: fraction of total time for the heat-up ramp (default 0.5)
        seed: RNG seed for velocities
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if atoms.calc is None:
        if calculator is None:
            raise ValueError("Need a calculator")
        atoms.calc = calculator

    if constraint is not None:
        if not isinstance(constraint, list):
            constraint = [constraint]
        atoms.set_constraint(constraint)

    # Initialize velocities
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)

    n_steps = int(total_time_ps * 1000 / timestep_fs)
    dump_interval = max(1, int(dump_every_fs / timestep_fs))

    # Set up integrator
    if ensemble == "langevin_nvt":
        friction_ase = friction_per_ps * 1e-3 / units.fs  # convert ps⁻¹ → ASE time units
        dyn = Langevin(atoms, timestep_fs * units.fs, temperature_K=temperature_K,
                       friction=friction_ase, rng=rng)
    elif ensemble == "verlet_nve":
        dyn = VelocityVerlet(atoms, timestep_fs * units.fs)
    else:
        raise ValueError(f"Unknown ensemble '{ensemble}'. Choices: langevin_nvt, verlet_nve")

    # Attach trajectory and log writers
    traj_path = outdir / "md.traj"
    log_path = outdir / "md.log"
    xyz_path = outdir / "md-traj.xyz"

    frames = []
    energies = []
    temperatures = []

    with open(log_path, "w") as lf:
        lf.write("# step time_ps E_pot_eV E_kin_eV T_K\n")

        def snapshot():
            e_pot = atoms.get_potential_energy()
            e_kin = atoms.get_kinetic_energy()
            T = atoms.get_temperature()
            t_ps = dyn.nsteps * timestep_fs / 1000
            lf.write(f"{dyn.nsteps:>8d} {t_ps:.4f} {e_pot:.6f} {e_kin:.6f} {T:.2f}\n")
            lf.flush()
            energies.append(float(e_pot))
            temperatures.append(float(T))
            frames.append(atoms.copy())

        dyn.attach(snapshot, interval=dump_interval)

        # Annealing schedule (Langevin only — NVE can't change T)
        if anneal_peak_K is not None and ensemble == "langevin_nvt":
            heat_steps = int(n_steps * anneal_fraction)
            cool_steps = n_steps - heat_steps
            # Heat up
            for i in range(heat_steps):
                alpha = (i + 1) / heat_steps
                T_target = temperature_K + alpha * (anneal_peak_K - temperature_K)
                dyn.set_temperature(T_target)
                dyn.run(1)
            # Cool down
            for i in range(cool_steps):
                alpha = (i + 1) / cool_steps
                T_target = anneal_peak_K + alpha * (temperature_K - anneal_peak_K)
                dyn.set_temperature(T_target)
                dyn.run(1)
        else:
            log.info(f"MD: {ensemble} @ {temperature_K} K, {total_time_ps} ps "
                     f"({n_steps} steps of {timestep_fs} fs)")
            dyn.run(n_steps)

    # Write extxyz trajectory for visualization
    if frames:
        for frame, e in zip(frames, energies):
            frame.info["energy_eV"] = e
        ase_write(str(xyz_path), frames, format="extxyz")

    e_final = atoms.get_potential_energy()
    T_final = atoms.get_temperature()

    result = {
        "status": "completed",
        "atoms": atoms,
        "energy_eV": float(e_final),
        "energy_kcal_mol": float(e_final * EV_TO_KCAL),
        "temperature_final_K": float(T_final),
        "energies_eV": energies,
        "temperatures_K": temperatures,
        "n_steps": n_steps,
        "n_frames_saved": len(frames),
        "outputs": {
            "trajectory": str(traj_path),
            "log": str(log_path),
            "xyz": str(xyz_path),
        },
    }

    (outdir / "md-summary.json").write_text(json.dumps({
        "ensemble": ensemble, "timestep_fs": timestep_fs,
        "total_time_ps": total_time_ps, "temperature_K": temperature_K,
        "n_steps": n_steps, "n_frames_saved": len(frames),
        "energy_final_eV": result["energy_eV"],
        "temperature_final_K": result["temperature_final_K"],
        "energy_mean_eV": float(np.mean(energies)) if energies else None,
        "energy_std_eV": float(np.std(energies)) if energies else None,
    }, indent=2))
    result["outputs"]["summary"] = str(outdir / "md-summary.json")

    log.info(f"MD done: {n_steps} steps, final E={e_final:.4f} eV, T={T_final:.1f} K")
    return result
