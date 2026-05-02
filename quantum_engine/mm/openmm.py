"""OpenMM wrapper — broadest MD engine in the codebase.

Why OpenMM:
  * Active upstream (8.4 / 8.5 currently vendored), big community.
  * CUDA-accelerated kernels — orders of magnitude faster than ASE
    for classical MD.
  * Force-field-agnostic by design: AMBER, CHARMM, OPLS for classical;
    MACE/ANI/NequIP via ``openmm-ml``; arbitrary PyTorch models via
    ``openmm-torch``; xtb via custom-force interfacing.
  * PLUMED biasing in-process via ``openmm-plumed`` (we already vendor
    PLUMED; this just binds it to OpenMM's MD loop instead of ASE's).
  * The MD layer that fits underneath the same Pipeline ``Step``
    adapters as :mod:`quantum_engine.ops.md`.

Install:
    bash deps/build_openmm.sh   # conda-forge → qcb-xtb env

Use:
    >>> from quantum_engine.mm.openmm import OpenMMSession
    >>> sess = OpenMMSession.from_pdb("system.pdb",
    ...                                forcefield="amber14-all.xml")
    >>> sess.run_md(total_time_ps=10.0, temperature_K=300)
    >>> sess.atoms                  # final ASE Atoms

For ML force fields see :func:`OpenMMSession.attach_mace`. For QM/MM
with xtb see :func:`OpenMMSession.attach_xtb_qm_region`.

Status: thin scaffold. The high-level workflow API is sketched; the
MACE-via-openmmml attachment + the xtb-as-custom-force bridge are
TODOs that need a labmate's real workflow to test against. The goal
here is the package layout + import-resolves story so the rest of
qcb can already point at it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("quantum_engine.mm.openmm")


def openmm_available() -> bool:
    """Is the openmm Python package importable?"""
    try:
        import openmm  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class OpenMMSession:
    """An OpenMM ``System`` + ``Simulation`` pair, with our extras
    (PLUMED bias, ML force, xtb QM region) attached as needed.

    Construct via :meth:`from_pdb` or :meth:`from_atoms`; methods like
    ``attach_mace`` / ``attach_plumed`` / ``attach_xtb_qm_region``
    modify the System in place and rebuild the Simulation.
    ``run_md`` advances; ``atoms`` materialises the current state as
    an ASE ``Atoms`` object.
    """
    system: Any = None              # openmm.System
    topology: Any = None            # openmm.app.Topology
    simulation: Any = None          # openmm.app.Simulation
    forcefield_name: str = "amber14-all.xml"
    integrator: str = "Langevin"
    timestep_fs: float = 1.0
    temperature_K: float = 300.0
    friction_per_ps: float = 1.0
    platform: str = "auto"          # "auto" | "CUDA" | "CPU" | "OpenCL"

    # ---------- constructors ----------
    @classmethod
    def from_pdb(cls, pdb_path: str | Path, *,
                 forcefield: str = "amber14-all.xml",
                 water: str | None = "amber14/tip3p.xml") -> "OpenMMSession":
        if not openmm_available():
            raise ImportError(
                "OpenMM not installed. Run `bash deps/build_openmm.sh` "
                "(conda-forge into qcb-xtb env, ~3 min)."
            )
        from openmm import app
        pdb = app.PDBFile(str(pdb_path))
        ff_files = [forcefield] + ([water] if water else [])
        ff = app.ForceField(*ff_files)
        system = ff.createSystem(pdb.topology, nonbondedMethod=app.PME,
                                 constraints=app.HBonds)
        sess = cls(
            system=system,
            topology=pdb.topology,
            forcefield_name=forcefield,
        )
        return sess._build_simulation(initial_positions=pdb.positions)

    @classmethod
    def from_atoms(cls, atoms: Any, *, forcefield: str = "amber14-all.xml") -> "OpenMMSession":
        """Build from ASE Atoms via a temp PDB (OpenMM's ForceField
        wants a topology object, which ASE doesn't natively carry)."""
        if not openmm_available():
            raise ImportError("OpenMM not installed; see deps/build_openmm.sh")
        import tempfile
        from ase.io import write as ase_write
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
            ase_write(tf.name, atoms)
            return cls.from_pdb(tf.name, forcefield=forcefield)

    def _build_simulation(self, initial_positions=None) -> "OpenMMSession":
        from openmm import LangevinIntegrator, Platform, unit
        from openmm.app import Simulation
        T = self.temperature_K * unit.kelvin
        gamma = self.friction_per_ps / unit.picosecond
        dt = (self.timestep_fs / 1000.0) * unit.picosecond
        integrator = LangevinIntegrator(T, gamma, dt)
        platform = None
        if self.platform != "auto":
            platform = Platform.getPlatformByName(self.platform)
        self.simulation = Simulation(self.topology, self.system, integrator,
                                     platform=platform)
        if initial_positions is not None:
            self.simulation.context.setPositions(initial_positions)
        return self

    # ---------- ML / QM force attachments ----------
    def attach_mace(self, model_path: str | Path) -> "OpenMMSession":
        """Add a MACE potential as an OpenMM Force (via openmm-ml)."""
        try:
            from openmmml import MLPotential
        except ImportError as e:
            raise ImportError(
                "openmm-ml not installed. Re-run "
                "`bash deps/build_openmm.sh` to pull it from conda-forge."
            ) from e
        potential = MLPotential("mace", modelPath=str(model_path))
        self.system = potential.createSystem(self.topology)
        return self._build_simulation()

    def attach_plumed(self, plumed_input: str) -> "OpenMMSession":
        """Add a PLUMED bias to the System (via openmm-plumed)."""
        try:
            from openmmplumed import PlumedForce
        except ImportError as e:
            raise ImportError(
                "openmm-plumed not installed; re-run deps/build_openmm.sh"
            ) from e
        plumed_force = PlumedForce(plumed_input)
        self.system.addForce(plumed_force)
        return self._build_simulation()

    def attach_xtb_qm_region(self, qm_indices: list[int]) -> "OpenMMSession":
        """STUB — OpenMM-xtb QM/MM via custom force.

        The hook lives here so callers can target this method; the
        implementation needs an ASE-xtb-as-OpenMM-CustomExternalForce
        bridge. Sketch in :mod:`enz_qc_pipelines.active_site_ts` once
        we wire stage 7 to the in-pocket QM/MM step.
        """
        raise NotImplementedError(
            "OpenMM-xtb QM/MM not wired yet. The hook is here so callers "
            "can target this method; implementation needs an ASE-xtb-as-"
            "OpenMM-CustomExternalForce bridge. Until that lands, fall "
            "back to the ASE-driven QM/MM pattern in "
            ":mod:`quantum_engine.ops.md` with a FixAtoms constraint on "
            "the MM region."
        )

    # ---------- run ----------
    def run_md(self, total_time_ps: float = 10.0, *,
               equilibration_steps: int = 1000) -> "OpenMMSession":
        from openmm import unit
        if self.simulation is None:
            self._build_simulation()
        self.simulation.minimizeEnergy()
        self.simulation.context.setVelocitiesToTemperature(
            self.temperature_K * unit.kelvin)
        if equilibration_steps:
            log.info(f"  equilibration: {equilibration_steps} steps")
            self.simulation.step(equilibration_steps)
        n_steps = int(total_time_ps * 1000 / self.timestep_fs)
        log.info(f"  production: {n_steps} steps ({total_time_ps} ps)")
        self.simulation.step(n_steps)
        return self

    @property
    def atoms(self) -> Any:
        """Materialise the current state as ASE Atoms."""
        from ase import Atoms
        from openmm import unit
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(
            unit.angstrom)
        symbols = [atom.element.symbol for atom in self.topology.atoms()]
        return Atoms(symbols=symbols, positions=positions)
