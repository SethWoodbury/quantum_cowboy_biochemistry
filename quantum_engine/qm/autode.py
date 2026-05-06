"""autodE adapter (duartegroup/autodE, MIT).

autodE drives a SMILES → reaction-profile workflow: parse the reactant
and product SMILES, build vacuum complexes, run a low-level NEB / saddle
search to localise a transition state, then (optionally) refine with a
hcode (DFT) and run a thermo + single-point sweep.

For QCB we use it strictly as a SMILES → vacuum-TS shortcut — the
cheapest reproducible way to land at "this is roughly the TS atoms in
free space" before docking into an enzyme. Per Seth's "no DFT in the
default loop" rule (see ``docs/plans/enzyme_ts_design.md``), we monkey
out the DFT hcode dispatch and force every internal call to use the
vendored xTB / g-xTB binary.

Two gotchas that motivated this dedicated module:

  1. autodE's stock h_methods are all DFT (ORCA, Gaussian, QChem, NWChem).
     Setting ``Config.hcode = None`` is not enough — many submodules do
     ``from autode.methods import get_hmethod`` at import time, which
     captures the original function reference. We have to monkey-patch
     **every** ``autode.*`` module that has already imported it.

  2. autodE rejects a TS at xTB-Hessian level when the imaginary mode is
     too shallow (common for cycloadditions: Diels-Alder TSs are
     genuinely flat at GFN2). When that happens, ``rxn.ts`` is None but
     a usable geometry is still on disk under
     ``transition_states/TS_*_optts_xtb.xyz``. We fall back to the
     highest-quality on-disk TS guess so downstream stages have a
     starting geometry; Stage 8 (HighResTSPolish) re-optimises with
     MACE-POLAR-1M + Sella anyway.

Public API:

  * :func:`autode_available` — bool import probe.
  * :func:`vacuum_ts_from_smiles` — the single SMILES → TS entry point
    that ``VacuumTSSearch`` calls.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("quantum_engine.qm.autode")


def autode_available() -> bool:
    """Is the ``autode`` package importable in this env?"""
    try:
        import autode  # noqa: F401
        return True
    except ImportError:
        return False


def _require() -> None:
    if not autode_available():
        raise ImportError(
            "autodE not installed. autodE is GitHub-only (NOT on PyPI). "
            "Install via:\n"
            "  pip install --target deps/.local_pkgs "
            "git+https://github.com/duartegroup/autodE.git\n"
            "Verified with v1.4.5 (2026-05-05). Note: autodE is for organic "
            "small-molecule reactions; charged species in vacuum and "
            "metalloenzyme active sites (binuclear Zn etc.) are out of scope."
        )


# ─────────────────────────────────────────────────────────────────────
# Configuration: point autodE at the vendored xTB binary and force
# every internal hmethod dispatch to return XTB() instead of DFT.
# ─────────────────────────────────────────────────────────────────────

def _configure_autode_for_xtb(qm_method: str = "g-xtb") -> str:
    """Wire autodE.Config to the vendored xTB / g-xTB binary and
    monkey-patch ``autode.methods.get_hmethod`` so every code path inside
    autodE picks XTB instead of an h_method DFT engine.

    Idempotent: safe to call multiple times in a single process. Returns
    the resolved xTB binary path (useful for logging / sanity assertion).
    """
    _require()

    import autode as ade
    import autode.methods as _ade_methods

    from quantum_engine.site import GXTB_BIN, XTB_BIN, XTB_LIB_DIRS

    # Resolve which xtb to point at. g-xTB is a strict superset of
    # upstream xtb that adds the `--gxtb` flag for Grimme's
    # wB97M-V/def2-TZVPPD-trained method. autodE's XTB wrapper drives
    # the binary as a stock xtb regardless of qm_method — choosing
    # GXTB_BIN here just routes commands through the g-xTB-aware fork.
    if qm_method.lower() in ("g-xtb", "gxtb"):
        xtb_path = GXTB_BIN if os.path.isfile(GXTB_BIN) else XTB_BIN
    else:
        xtb_path = XTB_BIN

    if not os.path.isfile(xtb_path):
        raise FileNotFoundError(
            f"_configure_autode_for_xtb: xtb binary not found at {xtb_path}. "
            "Build via `bash deps/build_xtb.sh` or `bash deps/bump_gxtb.sh`."
        )

    # Make sure the vendored libxtb / libquadmath are reachable when
    # autodE shells out to xtb.
    ld_path_existing = os.environ.get("LD_LIBRARY_PATH", "")
    ld_extra = ":".join(p for p in XTB_LIB_DIRS if p)
    if ld_extra and ld_extra not in ld_path_existing:
        os.environ["LD_LIBRARY_PATH"] = (
            ld_extra + (":" + ld_path_existing if ld_path_existing else "")
        )

    ade.Config.XTB.path = xtb_path
    ade.Config.lcode = "xtb"   # low-level (opt / freq)
    ade.Config.hcode = None    # disable DFT

    # autodE's stock h_methods are all DFT (ORCA / Gaussian / QChem /
    # NWChem). Force every dispatch to return XTB instead. This works
    # in two passes:
    #   1. Replace the function on the source module.
    #   2. Walk every already-imported autode.* submodule and patch
    #      its captured reference (``from autode.methods import
    #      get_hmethod`` at import time keeps a stale binding).
    import sys as _sys

    def _xtb_as_hmethod():
        return _ade_methods.XTB()

    _ade_methods.get_hmethod = _xtb_as_hmethod
    for _modname, _mod in list(_sys.modules.items()):
        if not _modname.startswith("autode") or _mod is None:
            continue
        if getattr(_mod, "get_hmethod", None) is not None:
            _mod.get_hmethod = _xtb_as_hmethod

    # Cap parallelism + sphere/rotation grids so a single SMILES → TS
    # call doesn't run for hours. These match the values that worked on
    # the L40 GPU runs in the Diels-Alder smoke test.
    ade.Config.n_cores = max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))
    ade.Config.num_complex_sphere_points = 4
    ade.Config.num_complex_random_rotations = 4

    return xtb_path


# ─────────────────────────────────────────────────────────────────────
# Public entry point: SMILES → TS
# ─────────────────────────────────────────────────────────────────────

def vacuum_ts_from_smiles(
    reactant_smiles: str,
    product_smiles: str,
    *,
    qm_method: str = "g-xtb",
    net_charge: int = 0,
    formal_charges_per_fragment_r: list[int] | None = None,
    formal_charges_per_fragment_p: list[int] | None = None,
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """Drive an autodE SMILES → vacuum-TS search using xTB / g-xTB.

    Parameters
    ----------
    reactant_smiles, product_smiles : str
        Joint SMILES strings for reactants / products. Multi-fragment
        reactions are split on ``.`` and a ``ade.Reactant``/``ade.Product``
        is created per fragment (autodE's Reactant SMILES parser does
        not handle ``.`` itself).
    qm_method : str, default "g-xtb"
        Which low-level method to point autodE at. Currently a label —
        both "g-xtb" and "xtb" route to the vendored binary; the wrapper
        treats them identically. Reserved for future use when MACE-POLAR
        becomes a Calculator-based hmethod.
    net_charge : int, default 0
        System net charge. Stored in ``ts_atoms.info["charge"]`` for
        downstream stages. Per-fragment charges are computed from RDKit
        formal charges on each fragment unless overridden.
    formal_charges_per_fragment_r, formal_charges_per_fragment_p :
        Optional explicit per-fragment charge overrides. Length must
        match the number of ``.``-separated fragments in the
        corresponding SMILES. If None, charges are inferred from RDKit
        ``GetFormalCharge`` per fragment.
    workdir : Path or str, optional
        Directory to run autodE in. Created if missing. autodE
        scatters subdirs everywhere, so we ``chdir`` into ``workdir``
        for the duration of the call.

    Returns
    -------
    dict
        Keys::

            ts_atoms        : ase.Atoms       — the TS geometry
            ts_xyz_path     : pathlib.Path    — written XYZ
            barrier_kcal    : float | None    — ΔE‡ in kcal/mol
            energy_eV       : float | None    — TS electronic energy
            imag_freqs      : list[float]     — imaginary modes (cm-1)
            ts_source       : str             — "autode-validated" |
                                                "autode-disk-fallback"
            xtb_path        : str             — binary path used
    """
    _require()

    if not reactant_smiles or not product_smiles:
        raise ValueError(
            "vacuum_ts_from_smiles: both reactant_smiles and product_smiles "
            "must be non-empty SMILES strings."
        )

    import autode as ade
    from ase import Atoms as ASEAtoms
    from ase.io import read as ase_read
    from rdkit import Chem as _Chem

    # ── 1. Configure autodE (Config + monkey-patch). Idempotent. ─────
    xtb_path = _configure_autode_for_xtb(qm_method)

    # ── 2. Set up workdir; autodE wants its own cwd. ─────────────────
    workdir = Path(workdir).resolve() if workdir is not None else Path.cwd().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    log.info(
        f"vacuum_ts_from_smiles(qm={qm_method}, charge={net_charge}) → {workdir}"
    )

    def _split_with_charges(
        smi: str, override: list[int] | None,
    ) -> list[tuple[str, int]]:
        """Split joint SMILES into (fragment_smiles, charge) pairs.

        If ``override`` is given, use those charges directly; otherwise
        sum the formal charges of each fragment via RDKit.
        """
        frags = [s.strip() for s in smi.split(".") if s.strip()]
        if override is not None:
            if len(override) != len(frags):
                raise ValueError(
                    f"per-fragment charge override has {len(override)} entries "
                    f"but SMILES has {len(frags)} fragments: {smi!r}"
                )
            return list(zip(frags, override))
        out: list[tuple[str, int]] = []
        for frag_smi in frags:
            m = _Chem.MolFromSmiles(frag_smi)
            q = sum(a.GetFormalCharge() for a in m.GetAtoms()) if m else 0
            out.append((frag_smi, q))
        return out

    r_frags = _split_with_charges(reactant_smiles, formal_charges_per_fragment_r)
    p_frags = _split_with_charges(product_smiles, formal_charges_per_fragment_p)
    if not r_frags or not p_frags:
        raise ValueError(
            "vacuum_ts_from_smiles: empty fragment list after SMILES split"
        )

    # Sanity: total per-fragment charges should match net_charge for
    # reactant side, and reactant ↔ product totals should agree.
    sum_r = sum(q for _, q in r_frags)
    sum_p = sum(q for _, q in p_frags)
    if sum_r != sum_p:
        log.warning(
            f"  per-fragment charge sums differ: r={sum_r}, p={sum_p}"
        )

    prev_cwd = Path.cwd()
    try:
        os.chdir(workdir)

        # ── 3. Build per-fragment Reactant/Product ──────────────────
        reactants = [
            ade.Reactant(name=f"R{i}", smiles=s, charge=q)
            for i, (s, q) in enumerate(r_frags)
        ]
        products = [
            ade.Product(name=f"P{i}", smiles=s, charge=q)
            for i, (s, q) in enumerate(p_frags)
        ]

        rxn = ade.Reaction(
            *reactants, *products,
            name="vacuum_ts",
            solvent_name=None,  # gas phase
        )

        # ── 4. locate_transition_state — cheapest (no thermo sweep) ──
        try:
            rxn.locate_transition_state()
        except Exception as e:
            # autodE often raises after partial success — capture
            # whatever TS it did find, otherwise re-raise.
            ts = getattr(rxn, "ts", None)
            if ts is None:
                raise RuntimeError(
                    f"autodE locate_transition_state failed: "
                    f"{type(e).__name__}: {e}"
                ) from e
            log.warning(
                f"  autodE raised {type(e).__name__}; partial TS recovered: {e}"
            )

        ts = getattr(rxn, "ts", None)
        ts_source = "autode-validated"
        ts_xyz_disk: Path | None = None

        # ── 5. Disk-fallback for shallow-saddle xTB rejections ───────
        # If autodE rejected the optimised TS (e.g. wrong # imag modes
        # at xTB Hessian level — common for cycloadditions where the
        # saddle is shallow), recover the highest-quality TS guess
        # autodE already wrote to disk. Stage 8 will re-optimise to a
        # proper saddle.
        if ts is None or getattr(ts, "atoms", None) is None:
            ts_dir = workdir / "transition_states"
            candidates = (
                sorted(ts_dir.glob("TS_*_optts_xtb.xyz"))
                if ts_dir.is_dir() else []
            )
            if not candidates:
                candidates = (
                    sorted(ts_dir.rglob("*_optts_xtb.xyz"))
                    if ts_dir.is_dir() else []
                )
            if candidates:
                ts_xyz_disk = candidates[0]
                log.warning(
                    f"  autodE rejected its TS (no/wrong imag modes); "
                    f"falling back to on-disk guess {ts_xyz_disk.name}. "
                    "Stage 8 polish will re-optimise."
                )
                ts_source = "autode-disk-fallback"
            else:
                raise RuntimeError(
                    "autodE finished but produced no TS geometry "
                    "(rxn.ts is None and no TS_*_optts_xtb.xyz on "
                    f"disk). Inspect logs in {workdir} for diagnostic hints."
                )

        # ── 6. Convert TS → ASE Atoms + write XYZ ───────────────────
        if ts is not None and getattr(ts, "atoms", None) is not None:
            symbols = [str(a.label) for a in ts.atoms]
            positions = [(float(a.coord[0]), float(a.coord[1]), float(a.coord[2]))
                         for a in ts.atoms]
            ts_ase = ASEAtoms(symbols=symbols, positions=positions)
        else:
            ts_ase = ase_read(str(ts_xyz_disk))  # type: ignore[arg-type]
            symbols = ts_ase.get_chemical_symbols()
            positions = ts_ase.get_positions().tolist()
        ts_ase.info["charge"] = net_charge
        ts_ase.info["ts_source"] = ts_source

        ts_xyz_path = workdir / "vacuum_ts.xyz"
        with ts_xyz_path.open("w") as fh:
            fh.write(f"{len(ts_ase)}\n")
            fh.write(f"vacuum TS guess from autodE (source={ts_source})\n")
            for sym, (x, y, z) in zip(symbols, positions):
                fh.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

        # PDB sidecar for PyMOL — bond connectivity from the reactant
        # SMILES (the TS atom order matches the reactant after autodE's
        # internal embedding). If this fails (size mismatch, etc.) the
        # XYZ above is still valid; we just lose bond visualisation.
        ts_pdb_path = workdir / "vacuum_ts.pdb"
        try:
            from quantum_engine.io.smiles_pdb import xyz_to_pdb
            xyz_to_pdb(ts_xyz_path, ts_pdb_path,
                       reference_smiles=reactant_smiles,
                       residue_prefix="TS ")
        except Exception as e:
            log.warning(
                f"  vacuum_ts.pdb sidecar failed ({type(e).__name__}: {e}); "
                "XYZ output is still valid."
            )
            ts_pdb_path = None  # signal to caller

        # ── 7. Barrier in kcal/mol (best-effort) ────────────────────
        barrier_kcal: float | None = None
        try:
            de_ts = rxn.delta("E‡")
            if de_ts is not None:
                if hasattr(de_ts, "to"):
                    barrier_kcal = float(de_ts.to("kcal mol-1"))
                else:
                    barrier_kcal = float(de_ts) * 627.5094740631
        except Exception as e:
            log.warning(f"  could not extract autodE barrier: {e}")

        # Energy in eV for downstream consumers
        energy_eV: float | None = None
        try:
            if ts is not None and ts.energy is not None:
                if hasattr(ts.energy, "to"):
                    energy_eV = float(ts.energy.to("eV"))
                else:
                    energy_eV = float(ts.energy) * 27.211386245988
        except Exception:
            pass

        # Imaginary frequencies, if available
        imag_freqs: list[float] = []
        try:
            if ts is not None and ts.imaginary_frequencies is not None:
                imag_freqs = [float(f) for f in ts.imaginary_frequencies]
        except Exception:
            pass

    finally:
        os.chdir(prev_cwd)

    return {
        "ts_atoms": ts_ase,
        "ts_xyz_path": ts_xyz_path,
        "ts_pdb_path": ts_pdb_path,
        "barrier_kcal": barrier_kcal,
        "energy_eV": energy_eV,
        "imag_freqs": imag_freqs,
        "ts_source": ts_source,
        "xtb_path": xtb_path,
    }
