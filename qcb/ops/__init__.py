"""QCB operations: Gaussian-style atomic-scale jobs.

Each op is a callable that takes an ASE Atoms + calculator + options and
returns a standardized result dict.

Available ops:
    sp       — single-point energy
    opt      — energy minimization
    md       — molecular dynamics
    freq     — vibrational frequency analysis
    scan     — 1D/2D coordinate scan
    saddle   — Sella saddle-point search
    irc      — intrinsic reaction coordinate
    neb      — nudged elastic band
    mtd      — well-tempered metadynamics
    ts       — high-level TS pipeline (composes saddle/irc/neb/mtd)
"""
from qcb.ops import sp, opt, md, freq, scan, saddle, irc, neb, mtd, gsm, ts

__all__ = ["sp", "opt", "md", "freq", "scan", "saddle", "irc", "neb", "mtd", "gsm", "ts"]
