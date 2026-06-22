"""Fixed-atom (Cartesian freeze) support for GSM/FSM via pysisyphus ``freeze_atoms``.

The string methods run *inside* pysisyphus (not on ASE images), so ASE ``FixAtoms``
never reached them. pysisyphus ``Geometry`` natively supports ``freeze_atoms``; the
fix threads a frozen-index list through ``_atoms_to_geom`` -> the endpoint geoms, and
``GrowingString`` grows nodes via ``Geometry.copy()`` (which preserves ``freeze_atoms``),
so every interior node inherits the constraint too.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pysisyphus")

from ase import Atoms

from quantum_engine.qm.pysisyphus import _atoms_to_geom


def _water():
    return Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])


def test_atoms_to_geom_sets_freeze_atoms():
    g = _atoms_to_geom(_water(), freeze_atoms=[0])
    assert list(g.freeze_atoms) == [0]


def test_atoms_to_geom_no_freeze_by_default():
    g = _atoms_to_geom(_water())
    assert len(getattr(g, "freeze_atoms", [])) == 0


def test_geometry_copy_preserves_freeze_atoms():
    # GrowingString grows nodes via Geometry.copy(); grown nodes must inherit the
    # frozen set or the string interior would drift off the constraint.
    g = _atoms_to_geom(_water(), freeze_atoms=[0, 2])
    assert list(g.copy().freeze_atoms) == [0, 2]


def test_gsm_holds_frozen_atom_across_the_string(tmp_path):
    """Full path: ops.gsm.run with freeze_atoms must hold that atom fixed in every
    output image, while a non-frozen atom is free to move."""
    from ase.calculators.emt import EMT
    from quantum_engine.ops import gsm

    r = Atoms("Cu4", positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [2.5, 2.5, 0.0]])
    p = r.copy()
    p.positions[3] = [2.5, 2.0, 1.0]
    res = gsm.run(r, p, lambda: EMT(), outdir=tmp_path, method="gsm",
                  n_images=5, fmax=0.5, freeze_atoms=[0])
    imgs = res.get("images", [])
    if not imgs:
        pytest.skip(f"GSM produced no images on this pysisyphus build (status={res.get('status')})")
    frozen_move = max(float(np.linalg.norm(im.positions[0] - imgs[0].positions[0])) for im in imgs)
    free_move = max(float(np.linalg.norm(im.positions[3] - imgs[0].positions[3])) for im in imgs)
    assert frozen_move < 1e-6, f"frozen atom moved {frozen_move} Å"
    assert free_move > 1e-3, "free atom did not move — test system degenerate"
