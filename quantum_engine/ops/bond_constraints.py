"""Per-bond ASE constraints for ``qcb opt`` (and reusable elsewhere).

Two ways to hold a forming/breaking bond during a relax:

* **Hard pin** — ``ase.constraints.FixBondLength``. It pins the bond at the
  distance present *when the constraint is created*, so to pin at a chosen
  length we ``set_distance`` first (mirrors ``ops/scan.py``).
* **Soft restraint** — :class:`HarmonicDistance`, a *two-sided* harmonic spring
  ``E = ½·k·(d−r0)²``. (ASE's ``Hookean`` is one-sided — it only acts when the
  bond exceeds ``rt`` — so it can't pull a bond toward a target from both sides.)

Indices are **0-based ASE** indices, matching ``qcb scan --indices``.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("quantum_engine.ops.bond_constraints")


class HarmonicDistance:
    """Two-sided harmonic distance restraint between atoms ``i`` and ``j``.

    ``E   = ½·k·(d_ij − r0)²``
    ``F_i = +k·(d − r0)·(r_j − r_i)/d`` ;  ``F_j = −F_i``
    """

    def __init__(self, i: int, j: int, k: float, r0: float):
        self.i = int(i)
        self.j = int(j)
        self.k = float(k)
        self.r0 = float(r0)

    def adjust_positions(self, atoms, new):  # soft restraint — never moves atoms
        return

    def adjust_forces(self, atoms, forces):
        r = atoms.positions[self.j] - atoms.positions[self.i]
        d = float(np.linalg.norm(r))
        if d < 1e-9:
            return
        unit = r / d
        f_mag = self.k * (d - self.r0)
        forces[self.i] += f_mag * unit
        forces[self.j] -= f_mag * unit

    def get_indices(self):
        return [self.i, self.j]

    def todict(self):
        return {"name": "HarmonicDistance",
                "kwargs": {"i": self.i, "j": self.j, "k": self.k, "r0": self.r0}}


def build_bond_constraints(atoms, fix_bond=None, restrain_bond=None) -> list:
    """Build ASE bond constraints from CLI specs.

    Parameters
    ----------
    atoms : ase.Atoms
        Mutated in place when a ``fix_bond`` target length is given
        (``set_distance`` before pinning).
    fix_bond : list of sequences
        Each is ``[i, j]`` (pin at current length) or ``[i, j, r0]`` (set the
        bond to ``r0`` Å, then pin). 0-based ASE indices.
    restrain_bond : list of sequences
        Each is ``[i, j, k, r0]`` → :class:`HarmonicDistance`.

    Returns
    -------
    list of ASE constraints (append AFTER any FixAtoms scaffold).
    """
    from ase.constraints import FixBondLength

    cons: list = []
    for spec in (fix_bond or []):
        if len(spec) not in (2, 3):
            raise SystemExit(f"--fix-bond expects 'I J' or 'I J R0', got {list(spec)!r}")
        i, j = int(spec[0]), int(spec[1])
        if len(spec) == 3:
            r0 = float(spec[2])
            atoms.set_distance(i, j, r0, fix=0.5)
            log.info(f"  fix-bond {i}-{j}: set to {r0:.3f} A, then pinned")
        else:
            d = float(atoms.get_distance(i, j))
            log.info(f"  fix-bond {i}-{j}: pinned at current length {d:.3f} A")
        cons.append(FixBondLength(i, j))

    for spec in (restrain_bond or []):
        if len(spec) != 4:
            raise SystemExit(f"--restrain-bond expects 'I J K R0', got {list(spec)!r}")
        i, j, k, r0 = int(spec[0]), int(spec[1]), float(spec[2]), float(spec[3])
        log.info(f"  restrain-bond {i}-{j}: harmonic k={k:g} eV/A^2 toward {r0:.3f} A")
        cons.append(HarmonicDistance(i, j, k, r0))

    return cons
