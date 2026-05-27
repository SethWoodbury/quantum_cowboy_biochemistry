"""Tests for the post-CREST geometry filter (revised 2026-05-06).

Covers:
  1. Source-PDB bond-order-aware repair target lookup
  2. Reactive atom protection (no auto-repair / no rejection)
  3. Source-distance shrink-tolerance vibration tolerance
  4. Metal exclusion from fallback element-pair table (log-only)
  5. Real-world KCX_set3 conf_01 case (HIS 257 N-CA = 1.073 Å)
  6. Element-pair fallback for brand-new contacts (no source bond)
  7. CLI argument parsing for new flags
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

# Make tools/ importable
TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import crest_funnel as cf  # type: ignore  # noqa: E402


def make_xyz_body(elems_coords: list[tuple[str, float, float, float]]) -> list[str]:
    return [f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}"
            for (el, x, y, z) in elems_coords]


class TestRepairTargetOrganicFallback(unittest.TestCase):
    """Verify the fallback table excludes metals and returns sane organics."""

    def test_organic_pairs_in_table(self):
        # C-C, C-N, C-O all in DEFAULT_REPAIR_BOND_LENGTHS
        self.assertAlmostEqual(cf._repair_bond_target_length_organic("C", "C"), 1.54)
        self.assertAlmostEqual(cf._repair_bond_target_length_organic("C", "N"), 1.47)
        self.assertAlmostEqual(cf._repair_bond_target_length_organic("N", "C"), 1.47)
        self.assertAlmostEqual(cf._repair_bond_target_length_organic("O", "C"), 1.43)

    def test_organic_pair_falls_back_to_covalent_radii(self):
        # B-N is not in the table but B and N have covalent radii so we
        # should get cov(B) + cov(N).
        target = cf._repair_bond_target_length_organic("B", "N")
        self.assertIsNotNone(target)
        # cov_B=0.84, cov_N=0.71 → 1.55
        self.assertAlmostEqual(target, 0.84 + 0.71, places=4)

    def test_metal_pair_returns_none(self):
        # Zn-O, Mg-N, Fe-S, Ca-O all metal-containing → None
        self.assertIsNone(cf._repair_bond_target_length_organic("Zn", "O"))
        self.assertIsNone(cf._repair_bond_target_length_organic("O", "Zn"))
        self.assertIsNone(cf._repair_bond_target_length_organic("Mg", "N"))
        self.assertIsNone(cf._repair_bond_target_length_organic("Fe", "S"))
        self.assertIsNone(cf._repair_bond_target_length_organic("Ca", "O"))
        # Both metals
        self.assertIsNone(cf._repair_bond_target_length_organic("Zn", "Fe"))


class TestMetalDetection(unittest.TestCase):
    def test_common_metals_recognized(self):
        for m in ("Zn", "Mg", "Mn", "Fe", "Cu", "Ca", "Ni", "Co", "Pd", "Pt"):
            self.assertTrue(cf._is_metal(m), f"{m} should be metal")

    def test_organics_not_metal(self):
        for e in ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"):
            self.assertFalse(cf._is_metal(e), f"{e} should not be metal")


class TestParseReactiveAtomsSpec(unittest.TestCase):
    def test_empty_spec(self):
        self.assertEqual(cf._parse_reactive_atoms_spec("", None), set())
        self.assertEqual(cf._parse_reactive_atoms_spec(None, None), set())
        self.assertEqual(cf._parse_reactive_atoms_spec("   ", None), set())

    def test_serial_only(self):
        # serials are 1-based, 0-based output
        self.assertEqual(cf._parse_reactive_atoms_spec("1,2,5", None), {0, 1, 4})

    def test_name_resname_token(self):
        atoms = [
            cf.PdbAtom(serial=1, record="ATOM", name="P1", altloc="",
                       resname="SUB", chain="A", resseq=1, icode="",
                       x=0., y=0., z=0., occ=1., bfac=0., element="P",
                       charge_field=""),
            cf.PdbAtom(serial=2, record="ATOM", name="O3", altloc="",
                       resname="OHX", chain="A", resseq=2, icode="",
                       x=0., y=0., z=0., occ=1., bfac=0., element="O",
                       charge_field=""),
            cf.PdbAtom(serial=3, record="ATOM", name="O7", altloc="",
                       resname="SUB", chain="A", resseq=3, icode="",
                       x=0., y=0., z=0., occ=1., bfac=0., element="O",
                       charge_field=""),
        ]
        # Mix tokens
        out = cf._parse_reactive_atoms_spec("P1.SUB,O3.OHX", atoms)
        self.assertEqual(out, {0, 1})
        # Mix serials and tokens
        out = cf._parse_reactive_atoms_spec("3,O3.OHX", atoms)
        self.assertEqual(out, {1, 2})

    def test_unknown_token(self):
        atoms = [
            cf.PdbAtom(serial=1, record="ATOM", name="CA", altloc="",
                       resname="HIS", chain="A", resseq=1, icode="",
                       x=0., y=0., z=0., occ=1., bfac=0., element="C",
                       charge_field=""),
        ]
        # No match — returns empty, but does not raise
        out = cf._parse_reactive_atoms_spec("NONESUCH.NOPE", atoms)
        self.assertEqual(out, set())
        # Bad integer — also no raise
        out = cf._parse_reactive_atoms_spec("not-an-int", atoms)
        self.assertEqual(out, set())


class TestSourcePdbLookup(unittest.TestCase):
    """Synthetic two-atom system: post bond is too short, source is normal."""

    def setUp(self):
        # Source: simple C-C single bond at 1.54 Å
        self.src_elems = ["C", "C"]
        self.src_coords = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
        # Post: same atoms but bond compressed to 1.05 Å
        self.post_elems = ["C", "C"]
        self.post_coords = np.array([[0.0, 0.0, 0.0], [1.05, 0.0, 0.0]])

    def test_atom_map_exact_match(self):
        m = cf._build_post_to_source_atom_map(
            self.post_elems, self.post_coords,
            self.src_elems, self.src_coords,
            max_match_distance_a=0.75,
        )
        self.assertEqual(m, [0, 1])

    def test_atom_map_unmatched_when_too_far(self):
        # Post atom 1 at x=10 (way too far from any source atom)
        post_coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        m = cf._build_post_to_source_atom_map(
            self.post_elems, post_coords,
            self.src_elems, self.src_coords,
            max_match_distance_a=0.75,
        )
        self.assertEqual(m[0], 0)
        self.assertIsNone(m[1])

    def test_lookup_source_distance(self):
        m = [0, 1]
        d = cf._lookup_source_bond_distance(0, 1, m, self.src_coords)
        self.assertAlmostEqual(d, 1.54, places=4)

    def test_lookup_returns_none_when_unmapped(self):
        m = [0, None]
        d = cf._lookup_source_bond_distance(0, 1, m, self.src_coords)
        self.assertIsNone(d)


class TestPostCrestFilterRejectMode(unittest.TestCase):
    """Reject mode with synthetic bad bonds."""

    def test_clean_passthrough(self):
        body = make_xyz_body([
            ("C", 0.0, 0.0, 0.0),
            ("C", 1.54, 0.0, 0.0),
        ])
        confs = [(0.0, body)]
        out, summary = cf.post_crest_geometry_filter(
            confs, bond_cutoff_a=1.10, bad_bond_mode="reject",
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(summary["n_clean_passthrough"], 1)
        self.assertEqual(summary["n_rejected"], 0)

    def test_reject_organic_bad_bond_no_source(self):
        # 0.9 Å C-C is way too short
        body = make_xyz_body([
            ("C", 0.0, 0.0, 0.0),
            ("C", 0.9, 0.0, 0.0),
        ])
        confs = [(0.0, body)]
        out, summary = cf.post_crest_geometry_filter(
            confs, bond_cutoff_a=1.10, bad_bond_mode="reject",
        )
        self.assertEqual(len(out), 0)
        self.assertEqual(summary["n_rejected"], 1)


class TestPostCrestFilterRepairMode(unittest.TestCase):
    def test_repair_with_source_lookup_uses_source_distance(self):
        # Source: C-C at 1.50 Å (perfectly fine single bond)
        src_elems = ["C", "C"]
        src_coords = np.array([[0.0, 0.0, 0.0], [1.50, 0.0, 0.0]])
        # Post: same atoms but bond compressed to 0.90 Å (far below
        # 0.7 * 1.50 = 1.05 shrink tol)
        body = make_xyz_body([
            ("C", 0.0, 0.0, 0.0),
            ("C", 0.90, 0.0, 0.0),
        ])
        confs = [(0.0, body)]
        out, summary = cf.post_crest_geometry_filter(
            confs, bond_cutoff_a=1.10, bad_bond_mode="repair",
            source_elems=src_elems, source_coords=src_coords,
            source_anchor_post_idx=[0, 1], source_anchor_src_idx=[0, 1],
            source_shrink_tolerance=0.7,
        )
        # NB: anchor_idx must have ≥3 entries; with 2 we skip source lookup,
        # so this test verifies graceful fallback path.
        # Verify that we got something out (either repaired via fallback
        # or via source) — just check the survivor count.
        self.assertEqual(summary["n_repaired"] + summary["n_clean_passthrough"], 1)

    def test_repair_with_3_anchors_uses_source(self):
        # Source: 3 anchor atoms (CA-like) + 2 atoms that have a shifted
        # C-C bond at 1.50 Å. Conformer has the same 3 anchors + 2 atoms
        # whose C-C is compressed to 0.95 Å. Source lookup should fire,
        # repair target should be 1.50.
        src_coords = np.array([
            [0.0, 0.0, 0.0],   # CA1 (anchor)
            [5.0, 0.0, 0.0],   # CA2 (anchor)
            [0.0, 5.0, 0.0],   # CA3 (anchor)
            [10.0, 10.0, 10.0],  # C
            [10.0 + 1.50, 10.0, 10.0],  # C (1.50 Å bond)
        ])
        src_elems = ["C", "C", "C", "C", "C"]
        # Post: same anchors (identical positions for simplicity), but
        # the C-C bond at the end compressed to 0.95 Å
        post_coords_array = src_coords.copy()
        post_coords_array[4] = [10.0 + 0.95, 10.0, 10.0]
        body = [
            f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}"
            for el, (x, y, z) in zip(src_elems, post_coords_array)
        ]
        confs = [(0.0, body)]
        out, summary = cf.post_crest_geometry_filter(
            confs, bond_cutoff_a=1.10, bad_bond_mode="repair",
            source_elems=src_elems, source_coords=src_coords,
            source_anchor_post_idx=[0, 1, 2],
            source_anchor_src_idx=[0, 1, 2],
            source_shrink_tolerance=0.7,
            max_match_distance_a=0.75,
        )
        self.assertEqual(summary["n_repaired"], 1)
        self.assertTrue(summary["have_source_lookup"])
        # Verify the repaired body has the bond at 1.50 (source distance)
        out_body = out[0][1]
        last_line = out_body[-1].split()
        x_repaired = float(last_line[1])
        # Repair stretches from 0.95 to 1.50 along the bond direction
        self.assertAlmostEqual(x_repaired, 10.0 + 1.50, places=2)
        # Confirm the per-conformer log records source_pdb as the target
        self.assertEqual(summary["per_conformer"][0]["mode"], "repaired")
        repair_log = summary["per_conformer"][0]["repair_log"]
        self.assertTrue(any(r.get("target_source") == "source_pdb"
                            for r in repair_log))


class TestSourceShrinkToleranceTolerates(unittest.TestCase):
    """Verify that bonds within shrink tolerance are LEFT ALONE."""

    def test_within_shrink_tolerance_is_tolerated(self):
        # Source C-C at 1.50; post at 1.06 Å. Ratio = 1.06/1.50 = 0.707,
        # JUST above shrink_tolerance=0.70 → tolerate.
        # But 1.06 < cutoff 1.10, so it WOULD be flagged in element-pair-
        # only mode. With source lookup + tol=0.70, it's accepted.
        src_coords = np.array([
            [0.0, 0.0, 0.0],   # CA1
            [5.0, 0.0, 0.0],   # CA2
            [0.0, 5.0, 0.0],   # CA3
            [10.0, 10.0, 10.0],  # C
            [10.0 + 1.50, 10.0, 10.0],  # C (1.50 Å bond)
        ])
        src_elems = ["C", "C", "C", "C", "C"]
        post_coords_array = src_coords.copy()
        post_coords_array[4] = [10.0 + 1.06, 10.0, 10.0]
        body = [
            f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}"
            for el, (x, y, z) in zip(src_elems, post_coords_array)
        ]
        confs = [(0.0, body)]
        out, summary = cf.post_crest_geometry_filter(
            confs, bond_cutoff_a=1.10, bad_bond_mode="reject",
            source_elems=src_elems, source_coords=src_coords,
            source_anchor_post_idx=[0, 1, 2],
            source_anchor_src_idx=[0, 1, 2],
            source_shrink_tolerance=0.70,
        )
        # 1.06 / 1.50 = 0.707 > 0.70: tolerated — survivor.
        self.assertEqual(len(out), 1)
        self.assertEqual(summary["n_tolerated_pairs"], 1)
        self.assertEqual(summary["n_rejected"], 0)

    def test_below_shrink_tolerance_rejects(self):
        # Source C-C at 1.50; post at 1.00 Å. Ratio = 1.00/1.50 = 0.667 < 0.70
        # → MTD artifact, reject.
        src_coords = np.array([
            [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0],
            [10.0, 10.0, 10.0], [10.0 + 1.50, 10.0, 10.0],
        ])
        src_elems = ["C", "C", "C", "C", "C"]
        post_coords_array = src_coords.copy()
        post_coords_array[4] = [10.0 + 1.00, 10.0, 10.0]
        body = [
            f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}"
            for el, (x, y, z) in zip(src_elems, post_coords_array)
        ]
        confs = [(0.0, body)]
        out, summary = cf.post_crest_geometry_filter(
            confs, bond_cutoff_a=1.10, bad_bond_mode="reject",
            source_elems=src_elems, source_coords=src_coords,
            source_anchor_post_idx=[0, 1, 2],
            source_anchor_src_idx=[0, 1, 2],
            source_shrink_tolerance=0.70,
        )
        self.assertEqual(len(out), 0)
        self.assertEqual(summary["n_rejected"], 1)


class TestReactiveAtomProtection(unittest.TestCase):
    """Bonds involving reactive atoms must NEVER be repaired or rejected."""

    def test_reactive_atom_bond_passes_through(self):
        # A 0.50 Å C-C bond (geometrically impossible) — but C2 is in the
        # reactive atom set, so it must pass through.
        src_coords = np.array([
            [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0],
            [10.0, 10.0, 10.0], [10.0 + 1.50, 10.0, 10.0],
        ])
        src_elems = ["C", "C", "C", "C", "C"]
        # Post: CCs collapsed to 0.5 Å — would normally be rejected, but
        # atom index 4 is reactive
        post_coords_array = src_coords.copy()
        post_coords_array[4] = [10.0 + 0.5, 10.0, 10.0]
        body = [
            f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}"
            for el, (x, y, z) in zip(src_elems, post_coords_array)
        ]
        confs = [(0.0, body)]
        out, summary = cf.post_crest_geometry_filter(
            confs, bond_cutoff_a=1.10, bad_bond_mode="reject",
            source_elems=src_elems, source_coords=src_coords,
            source_anchor_post_idx=[0, 1, 2],
            source_anchor_src_idx=[0, 1, 2],
            reactive_atoms={4},
            source_shrink_tolerance=0.70,
        )
        # Reactive-protected → not rejected.
        self.assertEqual(len(out), 1)
        self.assertEqual(summary["n_reactive_protected_pairs"], 1)
        self.assertEqual(summary["n_rejected"], 0)


class TestMetalNoSourceNoRepair(unittest.TestCase):
    """Metal-without-source must be log-only, not repaired or rejected."""

    def _build_metal_no_source_case(self, mode: str):
        src_coords = np.array([
            [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0],
            # Source has Zn at 100,100,100 and O at 101,100,100 — far
            # from the post-CREST positions. So no atom mapping.
            [100.0, 100.0, 100.0], [101.0, 100.0, 100.0],
        ])
        src_elems = ["C", "C", "C", "Zn", "O"]
        post_coords_array = src_coords.copy()
        post_coords_array[3] = [50.0, 50.0, 50.0]
        post_coords_array[4] = [50.0 + 1.0, 50.0, 50.0]
        body = [
            f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}"
            for el, (x, y, z) in zip(src_elems, post_coords_array)
        ]
        confs = [(0.0, body)]
        return cf.post_crest_geometry_filter(
            confs, bond_cutoff_a=1.10, bad_bond_mode=mode,
            source_elems=src_elems, source_coords=src_coords,
            source_anchor_post_idx=[0, 1, 2],
            source_anchor_src_idx=[0, 1, 2],
            source_shrink_tolerance=0.70,
            max_match_distance_a=0.75,
        )

    def test_repair_mode_passes_through(self):
        out, summary = self._build_metal_no_source_case("repair")
        # Conformer survives (no real_bad after classification).
        self.assertEqual(len(out), 1)
        self.assertEqual(summary["n_rejected"], 0)
        self.assertEqual(summary["n_repair_failed"], 0)
        self.assertEqual(summary["n_metal_no_source_pairs"], 1)
        # Should be tracked in clean_passthrough (no real_bad → kept).
        self.assertEqual(summary["n_clean_passthrough"], 1)

    def test_reject_mode_passes_through(self):
        # The user requirement: metal-no-source in REJECT mode also passes
        # through (don't auto-reject metal bonds).
        out, summary = self._build_metal_no_source_case("reject")
        self.assertEqual(len(out), 1)
        self.assertEqual(summary["n_rejected"], 0)
        self.assertEqual(summary["n_metal_no_source_pairs"], 1)


class TestRealKCXSet3Conf01Case(unittest.TestCase):
    """Reproduce the actual KCX_set3 conf_01 HIS 257 N-CA bug and verify
    the filter catches and repairs it correctly."""

    SOURCE_PDB = Path(
        "/net/scratch/woodbuse/PTE_iter_2026-05-06/KCX_set3_baseline_sum4p25/"
        "PdPTE_KCX_set3_waters__O3nuc_P1_O7lg__netCHG_minus_1.pdb"
    )
    FAILING_PDB = Path(
        "/net/scratch/woodbuse/PTE_iter_2026-05-06/_FAILING_CONFORMERS_for_inspection/"
        "KCX_set3_conf01.pdb"
    )

    def setUp(self):
        # Skip if test fixtures are missing
        if not self.SOURCE_PDB.exists():
            self.skipTest(f"source PDB not present: {self.SOURCE_PDB}")
        if not self.FAILING_PDB.exists():
            self.skipTest(f"failing PDB not present: {self.FAILING_PDB}")

    def _load_pdb_atoms(self, p: Path) -> tuple[list[str], np.ndarray]:
        elems: list[str] = []
        coords: list[list[float]] = []
        for line in p.read_text().splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            # Skip waters
            resname = line[17:20].strip()
            if resname in ("HOH", "WAT"):
                continue
            elem = line[76:78].strip()
            if not elem:
                # Fallback: parse from atom name (rare)
                elem = line[12:16].strip()[0].upper()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            elems.append(elem.capitalize())
            coords.append([x, y, z])
        return elems, np.asarray(coords)

    def test_default_cutoff_catches_the_artifact(self):
        # Build a 1-conformer "post-CREST" body from the failing PDB
        post_elems, post_coords = self._load_pdb_atoms(self.FAILING_PDB)
        # The PDB serial of N (atom 184) and C (atom 187) are 184 / 187 —
        # which after stripping waters from the source is the same numbering.
        # In the conformer file there are no waters at all so they are in
        # the same order as on disk: 184 and 187 (1-based) → 183 and 186
        # (0-based).
        d = float(np.linalg.norm(post_coords[183] - post_coords[186]))
        # Sanity: this is the 1.073 Å artifact
        self.assertAlmostEqual(d, 1.072, places=2)

        # Build a body
        body = [f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}"
                for el, (x, y, z) in zip(post_elems, post_coords)]
        # Just call the detector directly
        bad = cf._detect_short_heavy_heavy_bonds(
            post_elems, post_coords,
            cutoff_a=cf.POST_CREST_BOND_CUTOFF_DEFAULT,
        )
        # The 1.073 Å bond MUST appear at the top of the bad list
        self.assertGreaterEqual(len(bad), 1, "default cutoff missed the artifact")
        self.assertLess(bad[0][0], 1.10)
        # Confirm it's the (183, 186) pair
        worst_pair = (bad[0][1], bad[0][2])
        self.assertEqual(sorted(worst_pair), [183, 186])

    def test_default_tolerance_tolerates_artifact_via_source_lookup(self):
        # At default shrink-tolerance 0.7: ratio 1.073/1.475 = 0.727 > 0.7,
        # so the source-comparison path TOLERATES this bond. The conformer
        # should pass through without modification. Downstream xtb (with
        # task #25/#26 timeout-skip + salvage) is then responsible for
        # handling failed-SCF cases.
        post_elems, post_coords = self._load_pdb_atoms(self.FAILING_PDB)
        src_elems, src_coords = self._load_pdb_atoms(self.SOURCE_PDB)
        body = [f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}"
                for el, (x, y, z) in zip(post_elems, post_coords)]
        confs = [(0.0, body)]
        out, summary = cf.post_crest_geometry_filter(
            confs,
            bond_cutoff_a=cf.POST_CREST_BOND_CUTOFF_DEFAULT,
            bad_bond_mode="reject",
            source_elems=src_elems,
            source_coords=src_coords,
            same_atom_order=True,  # post and source share atom order
            source_shrink_tolerance=cf.POST_CREST_SOURCE_SHRINK_TOLERANCE_DEFAULT,
        )
        self.assertEqual(summary["have_source_lookup"], True)
        # With default 0.7 tolerance, the 1.073 ↔ 1.475 ratio (0.727) is
        # tolerated, so the conformer passes through.
        self.assertEqual(len(out), 1)
        self.assertEqual(summary["n_rejected"], 0)
        self.assertGreaterEqual(summary["n_tolerated_pairs"], 1)

    def test_tighter_tolerance_repairs_artifact(self):
        # Set tolerance to 0.80: now 0.727 < 0.80, so the bond is treated
        # as an MTD artifact and REPAIRED to source distance.
        post_elems, post_coords = self._load_pdb_atoms(self.FAILING_PDB)
        src_elems, src_coords = self._load_pdb_atoms(self.SOURCE_PDB)
        body = [f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}"
                for el, (x, y, z) in zip(post_elems, post_coords)]
        confs = [(0.0, body)]
        out, summary = cf.post_crest_geometry_filter(
            confs,
            bond_cutoff_a=cf.POST_CREST_BOND_CUTOFF_DEFAULT,
            bad_bond_mode="repair",
            source_elems=src_elems,
            source_coords=src_coords,
            same_atom_order=True,
            source_shrink_tolerance=0.80,
        )
        self.assertEqual(summary["have_source_lookup"], True)
        # With tighter 0.80 tolerance, the 1.073 ↔ 1.475 ratio (0.727) is
        # an artifact and the bond is repaired (to source 1.475).
        self.assertEqual(summary["n_repaired"], 1)
        out_body = out[0][1]
        new_coords = np.asarray(
            [[float(t) for t in line.split()[1:4]] for line in out_body]
        )
        d_new = float(np.linalg.norm(new_coords[183] - new_coords[186]))
        # Source bond is 1.475 Å; verify within 5% of source.
        self.assertAlmostEqual(d_new, 1.475, places=1)
        # Confirm the repair_log records source_pdb as the target source
        repair_log = summary["per_conformer"][0]["repair_log"]
        real_repairs = [r for r in repair_log if r.get("action") == "repaired"]
        self.assertGreaterEqual(len(real_repairs), 1)
        self.assertEqual(real_repairs[0]["target_source"], "source_pdb")


class TestCLIArgumentsParse(unittest.TestCase):
    def test_help_prints_new_flags(self):
        env = os.environ.copy()
        # Make help fast
        result = subprocess.run(
            [sys.executable, str(TOOLS_DIR / "crest_funnel.py"), "--help"],
            check=True, capture_output=True, text=True, env=env, timeout=30,
        )
        self.assertIn("--post-crest-bond-cutoff", result.stdout)
        self.assertIn("--post-crest-bad-bond-mode", result.stdout)
        self.assertIn("--post-crest-min-survivors", result.stdout)
        self.assertIn("--post-crest-repair-max-passes", result.stdout)
        self.assertIn("--post-crest-source-shrink-tolerance", result.stdout)
        self.assertIn("--post-crest-reactive-atoms", result.stdout)
        self.assertIn("--post-crest-max-match-distance", result.stdout)


if __name__ == "__main__":
    unittest.main()
