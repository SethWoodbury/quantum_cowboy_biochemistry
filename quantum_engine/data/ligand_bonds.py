"""Default bond-breaking definitions per ligand 3-letter code.

For each known ligand in the Baker-lab phosphoesterase / hydrolase
pipelines, this dict gives a list of ``(atom_a, atom_b, target_distance_A,
direction)`` tuples that drive endpoint-generation for NEB / TS searches.

* ``direction = "attractive"`` — bond shortens (nucleophile forms bond)
* ``direction = "repulsive"`` — bond lengthens (leaving group dissociates)

These are *default* targets. ``get_smart_bond_targets()`` in
:mod:`quantum_engine.qm.endpoints` (or the legacy
``tools/legacy/run_neb_ts.py``) overrides them with covalent-radii-based
values when ``--endpoint-method auto`` is used.

Adding a new ligand: drop a new key here. Convention: 3-letter PDB
ligand code, atom names match the ligand's HETATM entries. Tests
should reference the ligand by code, not by repeating the bond list.
"""

BOND_BREAKING_DEFS: dict[str, list[tuple[str, str, float, str]]] = {
    # ─── PTE phosphoester substrates ───
    "YYL": [
        ("P1", "O1", 1.4, "attractive"),   # nucleophile forms bond
        ("P1", "O5", 3.5, "repulsive"),    # leaving group fully dissociates
    ],
    "YYE": [
        ("P1", "O3", 1.4, "attractive"),
        ("P1", "O7", 3.5, "repulsive"),
    ],
    "YYF": [
        ("P1", "O3", 1.4, "attractive"),
        ("P1", "O7", 3.5, "repulsive"),
    ],
    "XUW": [
        ("P1", "O3", 1.4, "attractive"),
        ("P1", "O7", 3.5, "repulsive"),
    ],
    "YZW": [
        ("P1", "O1", 1.4, "attractive"),
        ("P1", "O5", 3.5, "repulsive"),
    ],
    "SUB": [
        ("P1", "O3", 1.4, "attractive"),
        ("P1", "O7", 3.5, "repulsive"),
    ],
    # ─── Non-PTE ───
    "PT4": [
        ("C7", "C8", 2.5, "repulsive"),
        ("C7", "C5", 1.4, "attractive"),
    ],
}
