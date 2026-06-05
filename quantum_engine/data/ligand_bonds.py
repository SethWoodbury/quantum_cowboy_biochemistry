"""OPTIONAL reference bond-breaking definitions, keyed by ligand 3-letter code.

This is a CONVENIENCE lookup, NOT a required part of the pipeline. It lets the
``qcb ts`` CLI auto-fill the CV atoms for a handful of curated ligands so you
don't have to type ``--p-idx/--nuc-idx/--lg-idx``. For ANY other reaction, the
pipeline does not consult this file — you specify the reactive atoms directly
(CLI flags or a ``ReactionSpec``), and ``ts_entry`` / ``scan_modes`` /
``cv_spring`` work entirely from those. Nothing here is hardcoded into the TS
search; it's reference data you can ignore, extend, or delete.

Each entry is a list of ``(atom_a, atom_b, target_distance_A, direction)``:
* ``direction = "attractive"`` — bond shortens (nucleophile forms bond)
* ``direction = "repulsive"`` — bond lengthens (leaving group dissociates)

These are rough *default* targets; covalent-radii-based auto-targets override
them when an ``auto`` endpoint method is used.

Adding a ligand: drop a new key here. Convention: 3-letter PDB code; atom names
match the ligand's HETATM entries. The set spans more than the OPAA/PTE test
case — see the non-PTE examples below — and is meant to grow as a shared,
reaction-agnostic reference, not to encode any single system.
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
    # ─── Non-PTE examples (this table is reaction-agnostic reference data) ───
    "PT4": [
        ("C7", "C8", 2.5, "repulsive"),
        ("C7", "C5", 1.4, "attractive"),
    ],
    # Generic SN2-at-carbon, e.g. an alkyl halide: nucleophile forms C-Nu while
    # the C-X leaving-group bond breaks. Atom names are placeholders — adjust to
    # your residue's HETATM names.
    "MCL": [
        ("C1", "NU", 1.5, "attractive"),   # nucleophile attacks carbon
        ("C1", "CL", 2.6, "repulsive"),    # halide leaves
    ],
    # Generic carboxylic-ester hydrolysis (acyl C attacked by water O, alkoxy
    # leaving group departs).
    "EST": [
        ("C1", "OW", 1.4, "attractive"),   # water oxygen forms bond to carbonyl C
        ("C1", "O2", 2.4, "repulsive"),    # alkoxy/leaving oxygen departs
    ],
}
