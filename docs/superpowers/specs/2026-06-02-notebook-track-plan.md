# Notebook track — plan for the 5 fixes

**Date:** 2026-06-02  ·  **Goal:** get the OPAA theozyme TS notebook running end-to-end on `qcb`.
**Status:** draft for codex + subagent review, then user approval.

## What the investigation changed

Three of the five items are smaller or differently-shaped than first thought:

- **#1 UMA is mostly already done.** `calc/factory.py::_family_of` already routes `uma-*` →
  `_make_uma`, which raises a clear "run inside the sidecar" error (commit f30aa47). The
  `uma-20260527.sif` sidecar contains `ase` + `quantum_engine` + `fairchem.core` — verified — so
  `apptainer exec uma.sif qcb … --model uma-s-1p1` works today. No silent MACE-fallback bug exists.
- **#2 does NOT depend on #5.** Bond-pinning can be CLI flags on `qcb opt` (ops/opt.py already
  accepts a constraint *list*; `qcb neb` already has the `--key-bond` pattern). No YAML/pydantic.
- **#3 is decided by container reality:** `graph_electrostatics` is absent → `mace-polar-m` can't
  load; `mace-mh-1 --head omol` **loads + evals + is charge-aware** (verified, H2O = −2079.85 eV).

So the notebook's critical path is **#2 + #3 + #4**; **#1** is verify+document; **#5** is decoupled.

## Item A — Bond constraints on `qcb opt`  (#2)  [DO]   (review-corrected)
- Add to the `opt` subparser (cli.py ~1240):
  - `--fix-bond I J [R0]` (repeatable) → hard pin via ASE `FixBondLength`. **Semantics fix (review):**
    `FixBondLength` pins at the distance *at constraint-creation time*. So if optional `R0` is given,
    call `atoms.set_distance(I, J, R0, fix=0.5)` FIRST, then construct the constraint (mirrors
    `ops/scan.py:92`). If `R0` omitted, pin the current length. Always **log the pinned length**.
  - `--restrain-bond I J K R0` (repeatable) → soft harmonic spring toward R0. **Semantics fix
    (review):** ASE `Hookean` is *one-sided* (only acts when r>rt), so it won't pull toward R0
    symmetrically. Add a tiny two-sided `HarmonicDistance` ASE constraint (precedent:
    `enz_qc_pipelines/active_site_refine/refine.py:993`) in a small `quantum_engine/mlff` or
    `ops` helper, and use it here.
- Indices are **0-based ASE** to match `qcb scan --indices` (confirmed in ops/scan.py:96 + the
  notebook). Document this in `--help`; note separately that `scan`(0-based) vs
  `refine-ts --reactive-atoms`(1-based PDB) is an inconsistency to reconcile later.
- Parse in `_cmd_opt`; **append bond constraints AFTER** the FixAtoms scaffold (match scan's
  `base_constraints + [bond_con]`, ops/scan.py:81). **No `ops/opt.py` change** (it normalizes a
  single constraint or list via `atoms.set_constraint`).
- Test (TDD): `--fix-bond i j` keeps |r_ij| fixed across relax (±1e-3 Å); `--fix-bond i j R0` pins at
  R0; `--restrain-bond` pulls toward R0 from BOTH sides; all compose with `--fix-preset ca-only`.

## Item B — Charge-aware model default + clear polar error  (#3)  [DO]
- In `calc/factory.py::_make_mace`, when `model.startswith("mace-polar")` (covers bare `mace-polar`
  too, site.py:176) — BEFORE the local-path `MACECalculator` construction (factory.py:~118) — attempt
  the `graph_electrostatics` import and raise a **clear** error ("mace-polar needs graph_electrostatics,
  absent in quantum_chem-*.sif; use `mace-mh-1 --head omol` (charge-aware, loads in-container) or
  `mace-omol` for higher accuracy on a big GPU") instead of a cryptic load failure.
- Update the OPAA notebook: primary model `mace-mh-1 --head omol` (was `mace-polar-m`); keep
  `mace-omol` noted as the high-accuracy A6000/H200 option. (Notebook is the user's file — propose
  the diff; don't edit without sign-off.)
- Defer: building `graph_electrostatics` into the container (a rebuild project).

## Item C — `--ligand-charge` on the protonator  (#4)  [DO]   (review-corrected)
- Add `--ligand-charge RESNAME=Q` (repeatable) to `protonator.main`'s argparse; parse with the
  existing `_parse_kv`.
- **Counting fix (review):** total = `compute_net_charge()` protein total + Σ over HETATM **residue
  instances** (group via `group_residues()` on `record=="HETATM"`, key `(chain,resid,icode,resname)`),
  adding the resname's Q **once per residue instance** — so two `ZN` residues at +2 each → +4, not +2,
  and a multi-atom ligand counts once. **Exclude waters** (`WATER_RESNAMES`, already defined). **Warn**
  when: a `--ligand-charge` resname isn't present in the structure; OR a non-water HETATM resname is
  present but has no `--ligand-charge` given (defaults 0 with a warning, so the user notices).
- Report `TOTAL_SYSTEM_CHARGE` in the REMARK 999 block and `total_system_charge` in the
  `--output-info-file` JSON, alongside `net_protein_charge`. Reporting-only — HETATM geometry untouched.
  (Extend `_qcb_remarks` + `_write_info_json` signatures to carry it.)
- Test against the OPAA input
  (`/home/woodbuse/for/antonia/opaa_theozyme/opaa_3l7g_optimal_maximal_theozyme_pxn_unprotonated.pdb`):
  with the di-Zn (`ZN=+2` each) + substrate charge, confirm the reported total matches the expected
  cluster charge.

## Item D — UMA: verify + document  (#1)  [VERIFY + DOC]
- Verify end-to-end: `apptainer exec --nv … uma-20260527.sif qcb sp <xyz> --model uma-s-1p1` returns
  an energy (the sidecar has everything; confirm the FAIRChemCalculator path actually evaluates).
- Document the sidecar-wrap pattern in README + the notebook (a `UMA=apptainer exec … uma-*.sif qcb`
  prefix), mirroring the existing `$QCB` prefix.
- FORK: optionally add a small `--container <name>` convenience to `cli.py` that auto-wraps a `qcb`
  call in the named sidecar (~40 LOC). Defer the full in-process subprocess/socket bridge (~600 LOC,
  not needed while the notebook uses MACE).

## Item E — pydantic / `qcb run`  (#5)  [DECOUPLED]
- Not on the notebook's path (notebook uses `qcb opt/scan/refine-ts`, not `qcb run`).
- **Rationale fix (review):** `deps/quantum_chem.def` pins `pydantic<2` as part of *global dependency
  hard-pinning / fairchem-conflict control* (the explicit SCINE comment is about numpy). Whether SCINE
  strictly needs pydantic v1 is unconfirmed — so pinning v2 + rebuild is *plausibly* safe but unverified
  and a heavy rebuild. The lower-risk real fix is a **v1-compatible `config/schema.py`** (rewrite the 3
  `field_validator` + 3 `model_validator` + discriminated-union + `ConfigDict` + `.model_validate`
  to v1 `validator`/`root_validator`/`class Config`/`parse_obj` + manual union dispatch). ~1–2 days.
- NOW (10 min): in `quantum_engine/config/__init__.py`, wrap the `schema import *` in a **narrow**
  `try/except ImportError` (review: NOT broad `except Exception`, which would hide real schema bugs);
  on failure set an explicit `CONFIG_UNAVAILABLE` marker. This clears the confusing
  `test_subpackage_imports` baseline failure but does NOT make `qcb run` work (it imports schema
  directly in run_config.py) — the v1 rewrite is scheduled separately.

## Sequencing & verification
1. A + B + C together (the notebook core), each TDD'd and verified in `quantum_chem-20260506.sif`.
2. D verify + doc.
3. E: defensive import now; v1 rewrite deferred to its own task.
4. Independent subagent review after A–C (as in the cleanup). codex review of this plan up front.
5. Then propose the OPAA notebook edits for the user to apply, and do an end-to-end dry-run of the
   notebook's printed commands.

## Open forks for the user
1. **UMA convenience** — add `--container` auto-wrap now, or document-only + defer?
2. **pydantic** — defensive-import now + defer the v1 rewrite, or schedule the full v1 rewrite as
   part of this track?
3. **Notebook edits** — want me to edit the OPAA `.ipynb` directly (mace-mh-1 default, qcb protonate,
   bond flags), or propose diffs for you to paste?
