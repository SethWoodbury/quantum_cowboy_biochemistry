"""Tests for ops.bond_monitor — non-constraining bond/metal report."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ase import Atoms

from quantum_engine.ops.bond_monitor import bonds_from_reaction, monitor_bonds


def test_bond_states_bonded_stretched_broken():
    # C-Cl: rcov(C)+rcov(Cl) ~ 1.78 Å. Place at ratios ~1.0 / ~1.45 / ~1.95.
    rcov = None
    from ase.data import covalent_radii, atomic_numbers
    rcov = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["Cl"]]
    for d, expect in [(rcov * 1.0, "bonded"), (rcov * 1.45, "stretched"),
                      (rcov * 1.95, "broken")]:
        atoms = Atoms("CCl", positions=[[0, 0, 0], [0, 0, d]])
        rep = monitor_bonds(atoms, bonds=[(0, 1)], metals=None)
        b = rep["bonds"][0]
        assert b["state"] == expect, (d, b["ratio"], b["state"])
        assert b["symbols"] == "C-Cl" and b["i"] == 0 and b["j"] == 1


def test_delta_vs_reference_atoms_and_dict():
    ref = Atoms("CCl", positions=[[0, 0, 0], [0, 0, 1.8]])
    cur = Atoms("CCl", positions=[[0, 0, 0], [0, 0, 2.3]])
    # reference as Atoms
    rep = monitor_bonds(cur, bonds=[(0, 1)], metals=None, reference=ref)
    assert abs(rep["bonds"][0]["delta_A"] - 0.5) < 1e-6
    # reference as dict
    rep2 = monitor_bonds(cur, bonds=[(0, 1)], metals=None, reference={(0, 1): 1.8})
    assert abs(rep2["bonds"][0]["delta_A"] - 0.5) < 1e-6
    # no reference -> None
    rep3 = monitor_bonds(cur, bonds=[(0, 1)], metals=None)
    assert rep3["bonds"][0]["delta_A"] is None


def test_metal_auto_detect_and_coordination():
    # Zn at origin; 3 O at 2.0 Å (coordinated), 1 O at 4.0 Å (not), 1 C far away.
    atoms = Atoms(
        "ZnOOOOC",
        positions=[
            [0, 0, 0],
            [2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0],   # 3 coordinated O
            [4.0, 0, 0],                              # far O
            [10, 0, 0],                               # far C
        ],
    )
    rep = monitor_bonds(atoms, metals="auto")
    assert len(rep["metals"]) == 1
    m = rep["metals"][0]
    assert m["symbol"] == "Zn" and m["index"] == 0
    assert m["coordination_number"] == 3
    # ligands sorted by distance, all the near O's
    assert [l["symbol"] for l in m["ligands"]] == ["O", "O", "O"]
    assert all(abs(l["distance_A"] - 2.0) < 1e-6 for l in m["ligands"])


def test_explicit_metal_indices_override_auto():
    atoms = Atoms("OO", positions=[[0, 0, 0], [0, 0, 1.4]])
    # force atom 0 to be treated as a "metal" center even though it's O
    rep = monitor_bonds(atoms, metals=[0])
    assert rep["metals"][0]["index"] == 0
    assert rep["metals"][0]["coordination_number"] == 1


def test_non_constraining_does_not_mutate_atoms():
    atoms = Atoms("CCl", positions=[[0, 0, 0], [0, 0, 1.8]])
    pos_before = atoms.get_positions().copy()
    n_constraints_before = len(atoms.constraints)
    monitor_bonds(atoms, bonds=[(0, 1)], metals="auto")
    assert np.allclose(atoms.get_positions(), pos_before)
    assert len(atoms.constraints) == n_constraints_before  # never adds constraints


def test_json_emission(tmp_path):
    atoms = Atoms("CCl", positions=[[0, 0, 0], [0, 0, 1.8]])
    rep = monitor_bonds(atoms, bonds=[(0, 1)], metals=None,
                        label="post_relax", outdir=tmp_path, write_json=True)
    out = tmp_path / "post_relax.json"
    assert out.exists() and rep["_json"] == str(out)
    import json
    loaded = json.loads(out.read_text())
    assert loaded["label"] == "post_relax" and loaded["bonds"][0]["i"] == 0


def test_bonds_from_reaction_helper():
    resolved = SimpleNamespace(forming=[(1, 2)], breaking=[(3, 4), (5, 6)])
    assert bonds_from_reaction(resolved) == [(1, 2), (3, 4), (5, 6)]
