"""QCB operations: Gaussian-style atomic-scale jobs.

Each op is a callable that takes an ASE Atoms + calculator + options and
returns a standardized result dict.

Available ops:
    sp       — single-point energy
    opt      — energy minimization
    md       — molecular dynamics
    freq     — vibrational frequency analysis
    scan     — 1D/2D coordinate scan
    scan_modes — bond-difference / two-sided / auto reaction-coordinate scans
    bond_monitor — non-constraining bond + metal-coordination report
    autoneb  — adaptive-image NEB (ASE-native) path method
    saddle   — multi-backend saddle-point search (registry: make_saddle_optimizer)
    irc      — intrinsic reaction coordinate (registry: make_irc)
    neb      — nudged elastic band
    gsm      — FSM / double-ended GSM path search
    path_search — unified path-method factory (registry: make_path_method)
    mtd      — well-tempered metadynamics
    ts       — high-level TS pipeline (composes saddle/irc/neb/mtd)
    ts_entry — reaction-agnostic TS orchestrator (ReactionSpec/RunContext → validated TS)
"""
from quantum_engine.ops import (
    sp, opt, md, freq, scan, scan_modes, bond_monitor, saddle, irc, neb,
    autoneb, path_search, mtd, mtd_walkers, gsm, ts, ts_entry,
)

__all__ = ["sp", "opt", "md", "freq", "scan", "scan_modes", "bond_monitor",
           "saddle", "irc", "neb", "autoneb", "path_search", "mtd",
           "mtd_walkers", "gsm", "ts", "ts_entry"]
