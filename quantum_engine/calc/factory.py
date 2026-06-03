"""MLFF calculator factory + model registry.

Originally a MACE-only factory; extended 2026-05-27 to dispatch to multiple
ML calculator backends based on the model alias:

    MACE        — mace-mp / mace-off / mace-omol / mace-mh / mace-polar
    ORB         — orb-mol / orb-mol-conservative
    AIMNet2     — aimnet2-rxn
    UMA         — uma-s-1p1 / uma-s-1p2 / uma-m-1p1 / uma-sm  (FairChem)

All ops (opt, md, freq, neb, ts, ...) call ``make_calc`` and get an
ASE-compatible ``Calculator`` back. Charge/spin handling differs by family:

* MACE      — charge-aware variants (mace-omol, mace-mh head=omol, mace-polar)
              read ``atoms.info["charge"]`` set by the caller.
* ORB       — charge/spin are conditioned on the conservative model via the
              ``SystemConfig`` passed to ``ORBCalculator``.
* AIMNet2   — charge/spin baked into ``AIMNet2ASE(charge=, mult=)`` at
              construction time.
* UMA       — system charge/spin set on the FAIRChemCalculator after build.

Example
-------
>>> from quantum_engine.calc import make_calc
>>> atoms.calc = make_calc("mace-omol", charge=0)
>>> atoms.calc = make_calc("orb-mol", charge=-2, spin=1, device="cuda")
>>> atoms.calc = make_calc("uma-s-1p1", charge=-2, spin=1, device="cuda")
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("quantum_engine.calc")


# Single source of truth for model paths: quantum_engine.site.MACE_MODELS.
# Edit that file (or override via env vars per the layered-config story) to
# add/move models; this factory just reads.
try:
    from quantum_engine.site import MACE_MODELS
except Exception:
    MACE_MODELS = {}


def list_models() -> dict[str, str]:
    """Return dict of available model aliases and their file paths."""
    return dict(MACE_MODELS)


# Alias-name → family. ``mace`` is the catch-all default (also covers any
# absolute .model path the user passes through).
def _family_of(model: str) -> str:
    m = model.lower()
    if m.startswith("uma-") or "fairchem" in m:
        return "uma"
    if m.startswith("orb"):
        return "orb"
    if m.startswith("aimnet"):
        return "aimnet"
    return "mace"


@dataclass
class CalcSpec:
    """Specification for a calculator (can be persisted to YAML/JSON)."""
    model: str = "mace-omol"
    head: str | None = None          # multi-head MACE (mace-mh head="omol")
    device: str = "cuda"
    default_dtype: str = "float64"
    charge: int | None = None
    spin: int | None = None          # 2S+1 / multiplicity (ORB, UMA, AIMNet2)


# ---------------------------------------------------------------------------
# Per-family builders.  Each returns an ASE-compatible Calculator.  Each is
# only imported lazily so the factory itself is import-cheap.
# ---------------------------------------------------------------------------
def _make_mace(model: str, model_path: str | None, head: str | None,
               device: str, default_dtype: str, charge: int | None
               ) -> "Calculator":
    from mace.calculators import MACECalculator  # noqa: PLC0415

    # MACE-POLAR (polarizable / long-range electrostatics) needs the
    # `graph_electrostatics` package, which is NOT in the standard
    # quantum_chem-*.sif. Detect early and fail with an actionable message
    # rather than a cryptic load error (covers both the local-path and the
    # auto-download branches, and the bare `mace-polar` alias).
    if "polar" in model.lower():
        try:
            import graph_electrostatics  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ImportError(
                f"{model!r} is a MACE-POLAR model and needs the "
                f"'graph_electrostatics' package, which is not installed in "
                f"this environment (e.g. the standard quantum_chem-*.sif). "
                f"Use a charge-aware model that loads here instead: "
                f"'mace-mh-1 --head omol' (recommended) or 'mace-omol' "
                f"(higher accuracy, needs a large GPU)."
            ) from exc

    # No local path → try MACE's HuggingFace auto-download by family.
    if model_path is None:
        try:
            if "omol" in model:
                from mace.calculators import mace_omol
                log.info("Loading MACE-OMOL via HuggingFace auto-download")
                return mace_omol(model="extra_large", device=device,
                                 default_dtype=default_dtype)
            if "polar" in model:
                from mace.calculators import mace_polar
                size = ("polar-1-s" if "-s" in model
                        else "polar-1-l" if "-l" in model else "polar-1-m")
                log.info(f"Loading MACE-POLAR via auto-download ({size})")
                return mace_polar(model=size, device=device,
                                  default_dtype=default_dtype)
            if "off" in model:
                from mace.calculators import mace_off
                size = ("small" if "small" in model
                        else "medium" if "medium" in model else "large")
                log.info(f"Loading MACE-OFF via auto-download ({size})")
                return mace_off(model=size, device=device,
                                default_dtype=default_dtype)
            if "mp" in model:
                from mace.calculators import mace_mp
                log.info("Loading MACE-MP via auto-download")
                return mace_mp(device=device, default_dtype=default_dtype)
        except Exception as exc:                # noqa: BLE001
            log.warning(f"MACE auto-download failed for {model}: {exc}")
        raise FileNotFoundError(
            f"MACE model {model!r} not available locally and auto-download "
            f"did not match any family ('mp', 'off', 'omol', 'polar').")

    log.info(f"Loading {model!r} from {model_path}"
             + (f" (head={head})" if head else ""))
    kwargs = dict(model_paths=model_path, device=device,
                  default_dtype=default_dtype)
    if head:
        kwargs["head"] = head
    calc = MACECalculator(**kwargs)
    if charge is not None:
        log.debug(f"  set atoms.info['charge'] = {charge} for charge-aware MACE")
    return calc


def _make_orb(model: str, model_path: str | None, device: str,
              charge: int | None, spin: int | None) -> "Calculator":
    """ORB-mol — orbital-materials.orb_models.

    The conservative orb-mol checkpoint is what's registered (forces are true
    energy gradients, which matters for saddle search). Charge/spin are read
    from ``atoms.info['charge']`` / ``atoms.info['spin']`` at force time; we
    stamp them onto the Atoms via a fresh ASE constraint-attribute mechanism
    isn't available here, so the caller must set ``atoms.info`` before any
    force call. For convenience the values supplied here are remembered on
    the returned calculator (``calc.qcb_charge`` / ``calc.qcb_spin``) and a
    helper wrapper stamps them onto atoms automatically."""
    try:
        from orb_models.forcefield import pretrained                  # noqa: PLC0415
        from orb_models.forcefield.atomic_system import SystemConfig  # noqa: PLC0415
        from orb_models.forcefield.calculator import ORBCalculator    # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            f"ORB requested ({model!r}) but `orb_models` is not installed "
            f"in this environment — run inside the quantum_chem container."
        ) from exc

    # alias → orb pretrained loader. We only ship orb-mol-conservative locally.
    loader = pretrained.orb_v3_conservative_omol
    weights = (model_path if model_path and os.path.isfile(model_path)
               else None)
    log.info(f"Loading ORB {model!r}"
             + (f" from {weights}" if weights
                else " via HuggingFace auto-download"))
    net = (loader(weights_path=weights, device=device) if weights
           else loader(device=device))

    sysconf = SystemConfig(radius=6.0, max_num_neighbors=20)
    calc = ORBCalculator(net, system_config=sysconf, device=device)
    # Stash charge/spin on the calculator; the caller is responsible for
    # copying them onto atoms.info before each force evaluation (the standard
    # ASE pattern for charge-aware MLFFs). Stage 12's charge accounting + the
    # CLI's _setup_atoms_and_calc already do this for the MACE family; the
    # same code path works here.
    if charge is not None:
        calc.qcb_charge = int(charge)
    if spin is not None:
        calc.qcb_spin = int(spin)
    return calc


def _make_aimnet(model: str, model_path: str | None, device: str,
                 charge: int | None, spin: int | None) -> "Calculator":
    """AIMNet2-rxn — Isayev lab, organic-only (H/C/N/O/F/S/Cl/...), charge-aware.

    The HF-Hub model id `isayevlab/aimnet2-rxn` is what AIMNet2Calculator
    expects; the local registry path under ``models--isayevlab--aimnet2-rxn``
    is the HF cache directory, picked up automatically when HF_HOME is set.
    """
    try:
        from aimnet.calculators.aimnet2ase import AIMNet2ASE  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            f"AIMNet2 requested ({model!r}) but `aimnet` is not installed in "
            f"this environment — run inside the quantum_chem container."
        ) from exc

    # alias → upstream HF revision; default to plain 'aimnet2' if unspecified.
    revision = "aimnet2-rxn" if "rxn" in model.lower() else None
    log.info(f"Loading AIMNet2 {model!r}"
             + (f" (HF revision={revision})" if revision else ""))
    return AIMNet2ASE(base_calc="aimnet2",
                      charge=charge if charge is not None else 0,
                      mult=spin if spin is not None else 1)


# UMA lives in its own apptainer image (the main quantum_chem.sif can't
# host fairchem-core — numpy 1.26 vs >=2.0 conflict). The factory imports
# fairchem.core directly if it's importable in the current Python (i.e.
# the user is already running inside the sidecar). Otherwise this branch
# raises an ImportError that names the sidecar so the caller can apptainer-
# exec into it.
_UMA_SIDECAR_GLOB = ("/net/software/containers/users/woodbuse/quantum_chem/"
                     "uma-*.sif")


def _resolve_uma_sidecar() -> str | None:
    import glob
    found = sorted(glob.glob(_UMA_SIDECAR_GLOB), reverse=True)
    return found[0] if found else None


def _make_uma(model: str, model_path: str | None, device: str,
              charge: int | None, spin: int | None) -> "Calculator":
    """FairChem UMA — universal foundation models.

    Lives in a dedicated apptainer sidecar (``uma-YYYYMMDD.sif``) because
    fairchem-core's numpy/e3nn pins conflict with the main qcb container.
    When called from a Python that already has ``fairchem.core`` importable
    (i.e. the caller has apptainer-exec'd into the sidecar), this builds
    the calculator in-process. Otherwise it raises a clear ImportError
    naming the sidecar path so the caller can wrap their command.
    """
    sidecar = _resolve_uma_sidecar()
    try:
        from fairchem.core.calculate.pretrained_mlip import load_predict_unit  # noqa: PLC0415
        from fairchem.core.calculate.ase_calculator import FAIRChemCalculator   # noqa: PLC0415
    except ImportError as exc:
        if sidecar:
            hint = (f"  - fairchem-core lives in the UMA sidecar — re-run "
                    f"your command inside it:\n"
                    f"      apptainer exec --nv --bind /home --bind /net "
                    f"{sidecar} \\\n          python <your_qcb_call_here>")
        else:
            hint = ("  - no UMA sidecar found under "
                    f"{_UMA_SIDECAR_GLOB.replace('*', 'YYYYMMDD')!r}.\n"
                    "    Build one with `apptainer build --fakeroot "
                    "deps/uma_sidecar.def` (see the def file for details).")
        raise ImportError(
            f"UMA requested ({model!r}) but `fairchem.core` is not importable "
            f"in this Python.\n{hint}"
        ) from exc

    if model_path is None or not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"UMA {model!r}: local checkpoint not on disk at {model_path!r}. "
            f"Run /net/databases/huggingface/mlFF_models/download_uma_models.sh "
            f"with an HF_TOKEN (FAIR Chemistry License v1).")

    log.info(f"Loading UMA {model!r} from {model_path}  "
             f"(charge={charge}, spin={spin})")
    # NB: load_predict_unit takes a file path; get_predict_unit takes a
    # registered model NAME and tries to hf_hub_download even when the
    # checkpoint already exists locally (and 401s on gated repos).
    predictor = load_predict_unit(model_path, device=device)
    calc = FAIRChemCalculator(predictor, task_name="omol")
    # charge and spin live on the calc, read from atoms.info at force time
    if charge is not None:
        calc.charge = int(charge)
    if spin is not None:
        calc.spin = int(spin)
    return calc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def make_calc(
    model: str = "mace-omol",
    head: str | None = None,
    device: str = "cuda",
    default_dtype: str = "float64",
    charge: int | None = None,
    spin: int | None = None,
) -> "Calculator":
    """Instantiate an ASE-compatible ML calculator.

    Args:
        model: Alias or an absolute path. Aliases recognised by family —
            MACE: mace-mp / mace-off / mace-omol / mace-mh / mace-polar.
            ORB: orb-mol / orb-mol-conservative.
            AIMNet2: aimnet2-rxn.
            UMA: uma-s-1p1 / uma-s-1p2 / uma-m-1p1 / uma-sm.
        head: Multi-head MACE only (e.g. "omol" for mace-mh-1).
        device: 'cuda' or 'cpu'.
        default_dtype: MACE only — 'float32' or 'float64'.
        charge: System net charge. Used by charge-aware MACE (via
            ``atoms.info``), ORB (via SystemConfig), AIMNet2 (at construction),
            UMA (on the calc).
        spin: 2S+1 multiplicity. ORB / AIMNet2 / UMA. Ignored by MACE.
    """
    models = list_models()
    family = _family_of(model)

    # Resolve to a local file path (or None → family-specific auto-download).
    if os.path.isfile(model):
        model_path = model
    elif model in models and models[model] and os.path.isfile(models[model]):
        model_path = models[model]
    else:
        if model in models and models[model]:
            log.warning(f"Registry path missing on disk: {models[model]!r} — "
                        f"falling back to family auto-download")
        model_path = None

    if family == "uma":
        return _make_uma(model, model_path or models.get(model), device,
                         charge, spin)
    if family == "orb":
        return _make_orb(model, model_path, device, charge, spin)
    if family == "aimnet":
        return _make_aimnet(model, model_path, device, charge, spin)
    return _make_mace(model, model_path, head, device, default_dtype, charge)


def make_calc_fn(**kwargs) -> Callable[[], "Calculator"]:
    """Return a zero-arg callable that constructs a fresh calculator each call.

    Useful for NEB where each image needs its own calculator instance.
    """
    def _factory():
        return make_calc(**kwargs)
    return _factory
