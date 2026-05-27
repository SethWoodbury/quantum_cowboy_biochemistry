"""graft_back.py — re-add pruned atoms at the end of the pipeline.

When the pipeline prunes residues for cheaper QM (e.g. drop W131 backbone or
keep only the LYS169 warhead), the polished output PDB is missing those
atoms. RFdiffusion3 needs the FULL atom set in its input even though it
will infer/refine positions during inference — so the final, final PDB
must be a complete drop-in replacement for the input.

This module performs the inverse of `prune_utils.apply_prune_with_caps`:

  1. Load the ORIGINAL input PDB (full atom set).
  2. Identify which atoms were pruned vs kept (via the
     ``old_to_new_index`` mapping that ``apply_prune_with_caps`` stashed
     on the polished lineage).
  3. Drop all HP* cap-H atoms from the polished lineage (their parent atoms
     are about to be re-bonded to the restored heavy atoms).
  4. Build a merged lineage: kept atoms at their POLISHED coordinates,
     pruned atoms at their ORIGINAL coordinates (so the user's request
     "same angle, distance, dihedral to unpruned parts" is satisfied
     automatically — the original input had those geometries).
  5. Run a fast xTB-GFN2 partial relax with the polished atoms
     FixAtoms-pinned and only the grafted (re-added) atoms free, to
     resolve any clashes from the merge. Loose convergence; stops after
     N steps even if not converged. User accepts residual clashes per
     spec.

Public API:
    apply_graft_back(polished_lineage, polished_coords, original_pdb,
                     *, do_xtb_relax=True, xtb_max_steps=50,
                     xtb_charge=0, xtb_solvent=None, log=None)
        → (merged_lineage, merged_coords, info_dict)

`info_dict` contains: n_original, n_polished, n_grafted, n_caps_removed,
n_clashes (heavy-atom pairs < 1.6 Å), max_clash_distance, runtime_sec.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from structure_io import (  # noqa: E402
    PdbAtom,
    StructureLineage,
    _guess_element,
    parse_pdb_lineage,
)


def apply_graft_back(
    polished_lineage: StructureLineage,
    polished_coords: np.ndarray,
    original_pdb: Path,
    *,
    do_xtb_relax: bool = True,
    xtb_max_steps: int = 50,
    xtb_charge: int = 0,
    xtb_solvent: str | None = None,
    log: logging.Logger | None = None,
) -> tuple[StructureLineage, np.ndarray, dict]:
    """Re-add pruned atoms using original PDB positions; xTB-relax the grafted
    atoms while keeping all polished atoms Cartesian-fixed.
    """
    log = log or logging.getLogger("graft_back")
    t0 = time.monotonic()

    info: dict = {
        "n_original": 0, "n_polished": 0, "n_grafted": 0,
        "n_caps_removed": 0, "n_clashes": 0,
        "max_clash_distance": None, "runtime_sec": 0.0,
        "xtb_relax_attempted": False, "xtb_relax_succeeded": False,
    }

    # --- 1. Load original PDB ---
    if not original_pdb.exists():
        raise FileNotFoundError(f"graft_back: original PDB not found: {original_pdb}")
    original_lineage = parse_pdb_lineage(original_pdb)
    n_orig = len(original_lineage.atoms)
    info["n_original"] = n_orig
    log.info(f"  graft-back: original PDB has {n_orig} atoms")

    # --- 2. Resolve old→new index mapping (stashed by apply_prune_with_caps) ---
    old_to_new = getattr(polished_lineage, "old_to_new_index", None)
    if old_to_new is None:
        log.warning(
            "  graft-back: polished lineage has no old_to_new_index "
            "(no pruning happened?). Returning polished lineage unchanged."
        )
        info["runtime_sec"] = time.monotonic() - t0
        return polished_lineage, polished_coords, info

    # Identify HP* caps in the polished lineage; these go away in graft-back
    cap_indices = [i for i, a in enumerate(polished_lineage.atoms)
                    if a.name.strip().startswith("HP")]
    info["n_caps_removed"] = len(cap_indices)
    log.info(f"  graft-back: removing {len(cap_indices)} cap H atoms (HP*)")

    # Build new index map that excludes caps
    cap_set = set(cap_indices)
    polished_keep_indices = [i for i in range(len(polished_lineage.atoms))
                              if i not in cap_set]
    info["n_polished"] = len(polished_keep_indices)

    # Reverse map: new_polished_idx → old_idx (for non-cap polished atoms)
    new_to_old: dict[int, int] = {}
    for old_idx, new_idx in old_to_new.items():
        if new_idx in cap_set:
            continue
        new_to_old[new_idx] = old_idx

    # --- 3. Build merged lineage in ORIGINAL atom order ---
    #     Each original atom either:
    #       (a) was kept → use its polished position
    #       (b) was pruned → use its original position (so geometry preserved)
    merged_atoms: list[PdbAtom] = []
    merged_coords_list: list[np.ndarray] = []
    grafted_indices: list[int] = []   # 0-based indices in the MERGED output

    original_coords = np.array(
        [[a.x, a.y, a.z] for a in original_lineage.atoms]
    )

    # For polished atoms, build inverse {old_idx: polished_idx}
    old_to_polished_idx = {old_idx: new_idx
                           for old_idx, new_idx in old_to_new.items()
                           if new_idx not in cap_set}

    for old_idx, orig_atom in enumerate(original_lineage.atoms):
        if old_idx in old_to_polished_idx:
            # Kept atom: use polished coords (has whatever the optimizer did)
            polished_idx = old_to_polished_idx[old_idx]
            xyz = polished_coords[polished_idx]
            polished_atom = polished_lineage.atoms[polished_idx]
            # Use polished atom (preserves any name/charge updates) but with
            # original residue context (resname/resseq/chain unchanged).
            merged_atoms.append(polished_atom)
            merged_coords_list.append(xyz.copy())
        else:
            # Pruned atom: graft back from original
            merged_atoms.append(orig_atom)
            merged_coords_list.append(original_coords[old_idx].copy())
            grafted_indices.append(len(merged_atoms) - 1)

    info["n_grafted"] = len(grafted_indices)
    log.info(f"  graft-back: merged {info['n_polished']} polished + "
             f"{info['n_grafted']} grafted atoms = {len(merged_atoms)} total "
             f"(original had {n_orig})")

    if len(merged_atoms) != n_orig:
        log.warning(
            f"  graft-back: merged atom count {len(merged_atoms)} != original "
            f"{n_orig}; this is unexpected. Check old_to_new_index integrity."
        )

    merged_coords = np.array(merged_coords_list)

    # --- 3a. Build merged lineage object preserving original REMARKs ---
    merged_lineage = StructureLineage(
        atoms=merged_atoms,
        raw_remarks=original_lineage.raw_remarks,
        matcher_remarks=original_lineage.matcher_remarks,
        qcb_lines=polished_lineage.qcb_lines,    # keep polished QCB lineage
        other_remarks=original_lineage.other_remarks,
    )

    # --- 4. Detect clashes (heavy-atom pairs < 1.6 Å) ---
    n_clashes = 0
    max_clash = 0.0
    if grafted_indices:
        polished_indices = [i for i in range(len(merged_atoms))
                             if i not in grafted_indices]
        for gi in grafted_indices:
            ga = merged_atoms[gi]
            if ga.element == "H" or _guess_element(ga.name) == "H":
                continue
            for pi in polished_indices:
                pa = merged_atoms[pi]
                if pa.element == "H" or _guess_element(pa.name) == "H":
                    continue
                d = float(np.linalg.norm(merged_coords[gi] - merged_coords[pi]))
                if d < 1.6:
                    n_clashes += 1
                    max_clash = max(max_clash, 1.6 - d)
        info["n_clashes"] = n_clashes
        info["max_clash_distance"] = round(max_clash, 4) if n_clashes else None
        log.info(f"  graft-back: detected {n_clashes} heavy-atom clashes "
                 f"(< 1.6 Å) before relax")

    # --- 5. xTB partial relax: free=grafted, fixed=polished ---
    if do_xtb_relax and grafted_indices:
        info["xtb_relax_attempted"] = True
        try:
            merged_coords = _xtb_relax_subset(
                merged_lineage, merged_coords,
                free_indices=grafted_indices,
                xtb_charge=xtb_charge,
                xtb_solvent=xtb_solvent,
                xtb_max_steps=xtb_max_steps,
                log=log,
            )
            info["xtb_relax_succeeded"] = True
            log.info("  graft-back: xTB partial relax complete")

            # Re-detect clashes after relax
            n_clashes_after = 0; max_clash_after = 0.0
            for gi in grafted_indices:
                if merged_atoms[gi].element == "H": continue
                for pi in range(len(merged_atoms)):
                    if pi in grafted_indices: continue
                    if merged_atoms[pi].element == "H": continue
                    d = float(np.linalg.norm(
                        merged_coords[gi] - merged_coords[pi]))
                    if d < 1.6:
                        n_clashes_after += 1
                        max_clash_after = max(max_clash_after, 1.6 - d)
            info["n_clashes_after_relax"] = n_clashes_after
            info["max_clash_distance_after_relax"] = (
                round(max_clash_after, 4) if n_clashes_after else None
            )
            log.info(f"  graft-back: post-relax clashes = {n_clashes_after}")
        except Exception as e:
            log.warning(f"  graft-back: xTB relax failed ({e}); keeping merged "
                        "geometry as-is. Residual clashes acceptable per spec.")
            info["xtb_relax_succeeded"] = False

    info["runtime_sec"] = round(time.monotonic() - t0, 2)
    log.info(f"  graft-back: done in {info['runtime_sec']:.2f}s")
    return merged_lineage, merged_coords, info


def _xtb_relax_subset(
    lineage: StructureLineage,
    coords: np.ndarray,
    free_indices: list[int],
    *,
    xtb_charge: int = 0,
    xtb_solvent: str | None = None,
    xtb_max_steps: int = 50,
    log: logging.Logger | None = None,
) -> np.ndarray:
    """Run xtb-GFN2 opt with all atoms NOT in free_indices Cartesian-fixed.

    Returns updated coords (or original on failure). Uses xtb's $fix block
    via constraints.inp; loose convergence; capped at xtb_max_steps.
    """
    log = log or logging.getLogger("graft_back")

    XTB_BIN = REPO / "deps/xtb/install/bin/xtb"
    if not XTB_BIN.exists():
        raise FileNotFoundError(f"xtb binary not found at {XTB_BIN}")

    n = len(coords)
    free_set = set(free_indices)
    fix_atoms_1based = [i + 1 for i in range(n) if i not in free_set]
    if not fix_atoms_1based:
        log.info("  xtb relax: nothing to fix; skipping")
        return coords
    if not free_indices:
        log.info("  xtb relax: nothing to free; skipping")
        return coords

    elements = [a.element if a.element else _guess_element(a.name)
                for a in lineage.atoms]
    work = Path(tempfile.mkdtemp(prefix="graft_xtb_"))
    try:
        # Write input.xyz
        xyz_path = work / "input.xyz"
        with xyz_path.open("w") as fh:
            fh.write(f"{n}\ngraft_back charge={xtb_charge}\n")
            for el, (x, y, z) in zip(elements, coords):
                fh.write(f"{el:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")
        # Constraints
        with (work / "constraints.inp").open("w") as fh:
            fh.write("$fix\n  atoms: ")
            ranges = []
            start = prev = fix_atoms_1based[0]
            for x in fix_atoms_1based[1:]:
                if x == prev + 1:
                    prev = x; continue
                ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
                start = prev = x
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            fh.write(",".join(ranges) + "\n")
            fh.write("$end\n")
        cmd = [str(XTB_BIN), "input.xyz",
               "--gfn", "2", "--opt", "loose",
               "--cycles", str(xtb_max_steps),
               "--chrg", str(xtb_charge),
               "--input", "constraints.inp"]
        if xtb_solvent:
            cmd += ["--alpb", xtb_solvent]
        log.info(f"  xtb partial-relax: free={len(free_indices)} "
                 f"fixed={len(fix_atoms_1based)} max_steps={xtb_max_steps} "
                 f"charge={xtb_charge}")
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=str(work),
                              capture_output=True, text=True, timeout=600)
        wall = time.monotonic() - t0
        log.info(f"  xtb partial-relax wall: {wall:.2f}s rc={proc.returncode}")
        if proc.returncode != 0:
            log.warning(f"  xtb returncode={proc.returncode} stderr_tail="
                        f"{proc.stderr[-400:] if proc.stderr else '(empty)'}")
            return coords
        opt_xyz = work / "xtbopt.xyz"
        if not opt_xyz.exists():
            log.warning("  xtbopt.xyz not produced; returning unrelaxed coords")
            return coords
        # Strict parse: check element + atom count
        lines = opt_xyz.read_text().splitlines()
        if len(lines) < 2 + n:
            log.warning(f"  xtbopt.xyz line count {len(lines)} < expected "
                         f"{2+n}; discarding")
            return coords
        new_coords = np.empty_like(coords)
        try:
            for i in range(n):
                parts = lines[2 + i].split()
                el_xyz = parts[0]
                if el_xyz != elements[i]:
                    log.warning(
                        f"  xtbopt.xyz row {i} element {el_xyz!r} != "
                        f"expected {elements[i]!r}; discarding relax result")
                    return coords
                new_coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
        except (IndexError, ValueError) as e:
            log.warning(f"  xtbopt.xyz parse error: {e}; discarding relax")
            return coords
        return new_coords
    finally:
        # Keep tempdir on failure for debugging? For now: clean up.
        import shutil as _sh
        _sh.rmtree(work, ignore_errors=True)
