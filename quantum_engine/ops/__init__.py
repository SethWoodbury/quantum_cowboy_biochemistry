"""QCB operations: Gaussian-style atomic-scale jobs.

Each op is a callable that takes an ASE Atoms + calculator + options and
returns a standardized result dict.

Available ops:
    sp       — single-point energy
    opt      — energy minimization
    md       — molecular dynamics
    freq     — vibrational frequency analysis
    scan     — 1D/2D coordinate scan
    saddle   — multi-backend saddle-point search (registry: make_saddle_optimizer)
    irc      — intrinsic reaction coordinate (registry: make_irc)
    neb      — nudged elastic band
    gsm      — FSM / double-ended GSM path search
    path_search — unified path-method factory (registry: make_path_method)
    mtd      — well-tempered metadynamics
    ts       — high-level TS pipeline (composes saddle/irc/neb/mtd)
"""
from quantum_engine.ops import (
    sp, opt, md, freq, scan, saddle, irc, neb, path_search, mtd,
    mtd_walkers, gsm, ts,
)

__all__ = ["sp", "opt", "md", "freq", "scan", "saddle", "irc", "neb",
           "path_search", "mtd", "mtd_walkers", "gsm", "ts"]
