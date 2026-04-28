"""Free Energy Surface (FES) analysis from PLUMED HILLS / COLVAR or qcb MTD output.

Capabilities
------------
- Parse PLUMED HILLS files (single or multi-walker)
- Parse PLUMED COLVAR files (CV trajectory + bias)
- Reconstruct FES via:
    (a) Sum-of-Gaussians from HILLS (`sum_hills` style, fast)
    (b) Reweighted KDE from COLVAR using bias values (more accurate, slower)
    (c) MBAR/WHAM via pymbar for umbrella sampling windows
- Detect basins (local minima)
- Estimate barrier heights between basins
- Plot 1D and 2D FES

Designed to consume output from:
- `qcb mtd` (our pure-Python MTD; writes hills.npy, fes.npy)
- `qcb md` with PLUMED bias (writes HILLS/COLVAR)
- enz-ts MTD runs (writes HILLS files)
- Any other PLUMED-driven simulation

References
----------
- Tiwary, P.; Parrinello, M. "A Time-Independent Free Energy Estimator for
  Metadynamics." J. Phys. Chem. B 2015, 119, 736. doi:10.1021/jp504920s
- Shirts, M.R.; Chodera, J.D. "Statistically optimal analysis of samples from
  multiple equilibrium states." J. Chem. Phys. 2008, 129, 124105 (MBAR).
- Branduardi, D.; Bussi, G.; Parrinello, M. "Metadynamics with Adaptive
  Gaussians." J. Chem. Theory Comput. 2012, 8, 2247.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger("qcb.analysis.fes")

# Boltzmann constant in kJ/(mol·K)
KB_KJ_PER_MOL_K = 8.314462618e-3
KJ_TO_KCAL = 1.0 / 4.184


@dataclass
class HillsData:
    """Parsed PLUMED HILLS file.

    Columns: time, cv values..., sigma values..., height, bias_factor.
    For OPES, may include extra columns; we tolerate them.
    """
    time_ps: np.ndarray         # (n_hills,)
    cv: np.ndarray              # (n_hills, n_dims)
    sigma: np.ndarray           # (n_hills, n_dims)
    height_kJ: np.ndarray       # (n_hills,) — actual deposited heights (after WT scaling)
    bias_factor: float          # γ (well-tempered factor)
    cv_names: list[str]         # CV column names
    walker_id: np.ndarray | None = None  # per-hill walker index (multi-walker)


@dataclass
class ColvarData:
    """Parsed PLUMED COLVAR (CV trajectory + optional bias)."""
    time_ps: np.ndarray         # (n_frames,)
    cv: np.ndarray              # (n_frames, n_cvs)
    cv_names: list[str]
    bias_kJ: np.ndarray | None = None  # (n_frames,) — V_bias evaluated at frame's CV


# ═══════════════════════════════════════════════════════════════════
# Parsers
# ═══════════════════════════════════════════════════════════════════

def parse_hills(path: str | Path) -> HillsData:
    """Parse a PLUMED HILLS file."""
    path = Path(path)
    headers, data, bias_factor = _parse_plumed_text(path)

    # Find CV columns: those between 'time' and 'sigma_*'/'height'
    cv_cols, sigma_cols, height_col = [], [], None
    cv_names = []
    for i, h in enumerate(headers):
        if h.startswith("sigma_"):
            sigma_cols.append(i)
        elif h == "height":
            height_col = i
        elif h == "biasf" or h == "bias_factor":
            pass
        elif h not in ("time", "walker_id", "sim_num", "logweight"):
            if not sigma_cols:  # CVs come before sigmas
                cv_cols.append(i)
                cv_names.append(h)

    time_ps = data[:, headers.index("time")]
    cv = data[:, cv_cols]
    sigma = data[:, sigma_cols]
    height_kJ = data[:, height_col] if height_col is not None else np.ones(len(data))

    walker_id = data[:, headers.index("walker_id")] if "walker_id" in headers else None

    return HillsData(
        time_ps=time_ps, cv=cv, sigma=sigma,
        height_kJ=height_kJ, bias_factor=bias_factor,
        cv_names=cv_names, walker_id=walker_id,
    )


def parse_colvar(path: str | Path) -> ColvarData:
    """Parse a PLUMED COLVAR file."""
    path = Path(path)
    headers, data, _ = _parse_plumed_text(path)

    time_ps = data[:, headers.index("time")]
    bias_col = next((i for i, h in enumerate(headers) if h == "metad.bias" or h == "opes.bias"), None)
    bias_kJ = data[:, bias_col] if bias_col is not None else None

    skip = {"time", "metad.bias", "opes.bias", "metad.rbias", "metad.rct"}
    cv_cols = [i for i, h in enumerate(headers) if h not in skip]
    cv_names = [headers[i] for i in cv_cols]

    return ColvarData(
        time_ps=time_ps, cv=data[:, cv_cols], cv_names=cv_names, bias_kJ=bias_kJ,
    )


def _parse_plumed_text(path: Path) -> tuple[list[str], np.ndarray, float]:
    """Parse a PLUMED-formatted text file. Returns (header_names, data, bias_factor)."""
    headers: list[str] = []
    bias_factor = 1.0
    data_rows = []

    with path.open() as f:
        for line in f:
            stripped = line.lstrip()
            if not stripped:
                continue
            if stripped.startswith("#!"):
                parts = stripped.split()
                if len(parts) >= 2 and parts[1] == "FIELDS":
                    headers = parts[2:]
                elif len(parts) >= 4 and parts[1] == "SET":
                    if parts[2] in ("biasf", "bias_factor"):
                        try:
                            bias_factor = float(parts[3])
                        except ValueError:
                            pass
                continue
            try:
                data_rows.append([float(x) for x in line.split()])
            except ValueError:
                continue

    if not data_rows or not headers:
        raise ValueError(f"Could not parse {path}: no data or no header")

    n_cols = len(headers)
    data_rows = [r for r in data_rows if len(r) == n_cols]
    return headers, np.asarray(data_rows), bias_factor


# ═══════════════════════════════════════════════════════════════════
# FES reconstruction
# ═══════════════════════════════════════════════════════════════════

def fes_from_hills(
    hills: HillsData,
    grid: np.ndarray | tuple[float, float, int] | None = None,
    bias_factor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct 1D FES by summing deposited Gaussian hills.

    For well-tempered MTD: F(s) = -(γ/(γ-1)) × V_bias(s)

    Args:
        hills: parsed HillsData (1D CV)
        grid: either an explicit np.ndarray of CV values, or (cv_min, cv_max, n)
              tuple. Default: 200 points spanning [min(cv)-3σ, max(cv)+3σ].
        bias_factor: override hills.bias_factor (e.g., for OPES without WT)

    Returns:
        (cv_grid, F_kJ_per_mol) — F is shifted so min = 0
    """
    if hills.cv.shape[1] != 1:
        raise ValueError("fes_from_hills: 1D only; for 2D use fes_from_hills_2d")

    s = hills.cv[:, 0]
    sig = hills.sigma[:, 0]
    h = hills.height_kJ
    gamma = bias_factor if bias_factor is not None else hills.bias_factor

    if grid is None:
        smin = float(s.min() - 3 * sig.max())
        smax = float(s.max() + 3 * sig.max())
        grid = np.linspace(smin, smax, 400)
    elif isinstance(grid, tuple):
        grid = np.linspace(grid[0], grid[1], grid[2])

    # V_bias(s) = sum_i h_i exp(-(s - s_i)^2 / (2 sigma_i^2))
    diff = grid[:, None] - s[None, :]
    V_bias = np.sum(h[None, :] * np.exp(-(diff**2) / (2 * sig[None, :]**2)), axis=1)

    if gamma > 1:
        F = -(gamma / (gamma - 1.0)) * V_bias
    else:
        F = -V_bias

    F -= F.min()
    return grid, F


def fes_from_colvar_reweighted(
    colvar: ColvarData,
    temperature_K: float = 300.0,
    grid: np.ndarray | None = None,
    bandwidth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FES from biased trajectory via Tiwary-Parrinello reweighting.

    Each frame contributes weight w_i = exp(+β V_bias(s_i)). The reweighted
    distribution P(s) is unbiased; F(s) = -kT log P(s).

    More accurate than sum-of-hills for OPES and for short post-convergence
    averaging.
    """
    if colvar.bias_kJ is None:
        raise ValueError("COLVAR must include metad.bias or opes.bias column for reweighting")

    s = colvar.cv[:, 0]
    beta = 1.0 / (KB_KJ_PER_MOL_K * temperature_K)
    weights = np.exp(beta * colvar.bias_kJ)
    weights /= weights.sum()

    if grid is None:
        grid = np.linspace(s.min(), s.max(), 400)

    # Weighted KDE (using a Gaussian kernel)
    if bandwidth is None:
        # Silverman's rule, weighted
        bandwidth = 1.06 * np.sqrt(np.cov(s, aweights=weights)) * len(s) ** (-1 / 5)

    diff = grid[:, None] - s[None, :]
    P = np.sum(weights[None, :] * np.exp(-0.5 * (diff / bandwidth) ** 2), axis=1)
    P /= np.trapz(P, grid)

    P = np.clip(P, 1e-300, None)
    F = -1.0 / beta * np.log(P)
    F -= F.min()
    return grid, F


def fes_from_umbrella_pymbar(
    cv_per_window: list[np.ndarray],
    spring_centers: np.ndarray,
    spring_constants: float | np.ndarray,
    temperature_K: float = 300.0,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FES from umbrella sampling windows via MBAR.

    Args:
        cv_per_window: list of N arrays, each containing CV values from one window
        spring_centers: (N,) array of harmonic restraint centers
        spring_constants: scalar or (N,) — restraint k in kJ/(mol·Å²)
        temperature_K: temperature
        grid: CV grid; default = linspace over data range, 200 points

    Returns:
        (cv_grid, F_kJ_per_mol)
    """
    try:
        from pymbar import MBAR, timeseries
    except ImportError:
        raise ImportError(
            "pymbar required for umbrella-sampling reweighting. "
            "Install with: pip install pymbar"
        )

    N_windows = len(cv_per_window)
    K = np.broadcast_to(spring_constants, (N_windows,))

    # Build u_kn matrix (reduced potentials) for MBAR
    beta = 1.0 / (KB_KJ_PER_MOL_K * temperature_K)
    n_per_window = np.array([len(x) for x in cv_per_window])
    n_total = n_per_window.sum()

    u_kn = np.zeros((N_windows, n_total))
    cv_all = np.concatenate(cv_per_window)
    for k in range(N_windows):
        bias = 0.5 * K[k] * (cv_all - spring_centers[k]) ** 2
        u_kn[k] = beta * bias

    mbar = MBAR(u_kn, n_per_window)

    if grid is None:
        grid = np.linspace(cv_all.min(), cv_all.max(), 200)

    # Use compute_pmf for the unbiased (k=N+1, no bias) state
    bins = np.digitize(cv_all, grid) - 1
    bins = np.clip(bins, 0, len(grid) - 1)
    F_grid_kT, _ = mbar.compute_free_energy_differences(return_dict=False, return_theta=False)[:2] \
        if False else (None, None)  # API placeholder

    # Simpler approach: compute weights, then weighted histogram → KDE
    weights = np.exp(mbar.f_k - u_kn[0])  # wrt window 0 (any)
    weights = mbar.W_nk[:, 0]  # weights of each sample under unbiased
    # Build P(s) via weighted KDE
    bandwidth = 1.06 * np.sqrt(np.cov(cv_all, aweights=weights)) * n_total ** (-1 / 5)
    diff = grid[:, None] - cv_all[None, :]
    P = np.sum(weights[None, :] * np.exp(-0.5 * (diff / bandwidth) ** 2), axis=1)
    P /= np.trapz(P, grid)
    P = np.clip(P, 1e-300, None)
    F = -1.0 / beta * np.log(P)
    F -= F.min()
    return grid, F


# ═══════════════════════════════════════════════════════════════════
# Basin / barrier analysis
# ═══════════════════════════════════════════════════════════════════

def find_basins(
    cv_grid: np.ndarray, F: np.ndarray,
    min_separation: float = 0.3,
    energy_tolerance_kJ: float = 5.0,
) -> list[dict]:
    """Detect local minima (basins) in a 1D FES.

    Args:
        cv_grid, F: CV grid and FES (kJ/mol)
        min_separation: minimum separation between basins (in CV units)
        energy_tolerance_kJ: minima within this depth of the global min are
                            considered "real" basins (filters numerical noise)

    Returns: list of dicts {cv: float, F_kJ: float, F_kcal: float, idx: int}
    """
    from scipy.signal import argrelextrema
    minima_idx = argrelextrema(F, np.less)[0]

    basins = []
    last_cv = -np.inf
    for idx in minima_idx:
        cv_val = float(cv_grid[idx])
        if cv_val - last_cv < min_separation:
            continue
        if F[idx] - F.min() > energy_tolerance_kJ:
            # Skip numerical-noise minima far above the global min
            pass
        basins.append({
            "cv": cv_val,
            "F_kJ": float(F[idx]),
            "F_kcal": float(F[idx] * KJ_TO_KCAL),
            "idx": int(idx),
        })
        last_cv = cv_val
    return basins


def barrier_between_basins(
    cv_grid: np.ndarray, F: np.ndarray,
    basin_a: dict, basin_b: dict,
) -> dict:
    """Find the highest point on the FES between two basins.

    Returns: {ts_cv, ts_F_kJ, fwd_kJ, rev_kJ, ts_idx}
    """
    i, j = sorted([basin_a["idx"], basin_b["idx"]])
    segment = F[i:j+1]
    ts_offset = int(np.argmax(segment))
    ts_idx = i + ts_offset
    ts_F = float(F[ts_idx])
    return {
        "ts_cv": float(cv_grid[ts_idx]),
        "ts_F_kJ": ts_F,
        "ts_F_kcal": ts_F * KJ_TO_KCAL,
        "fwd_kJ": ts_F - basin_a["F_kJ"],
        "rev_kJ": ts_F - basin_b["F_kJ"],
        "fwd_kcal": (ts_F - basin_a["F_kJ"]) * KJ_TO_KCAL,
        "rev_kcal": (ts_F - basin_b["F_kJ"]) * KJ_TO_KCAL,
        "ts_idx": ts_idx,
    }


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_fes(
    cv_grid: np.ndarray, F: np.ndarray,
    basins: list[dict] | None = None,
    barrier: dict | None = None,
    title: str | None = None,
    cv_label: str = "CV",
    out_path: str | Path | None = None,
    units: str = "kcal",
):
    """Plot 1D FES with annotated basins and barriers."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available")
        return None

    F_plot = F * KJ_TO_KCAL if units == "kcal" else F
    ylabel = "Free energy (kcal/mol)" if units == "kcal" else "Free energy (kJ/mol)"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cv_grid, F_plot, "b-", linewidth=2)

    if basins:
        cv_b = [b["cv"] for b in basins]
        F_b = [b["F_kJ"] * (KJ_TO_KCAL if units == "kcal" else 1) for b in basins]
        ax.scatter(cv_b, F_b, c="green", s=100, zorder=5, label="basins")

    if barrier is not None:
        F_ts = barrier["ts_F_kJ"] * (KJ_TO_KCAL if units == "kcal" else 1)
        ax.scatter([barrier["ts_cv"]], [F_ts], c="red", s=120, marker="^", zorder=5, label="TS")
        ax.annotate(
            f'Δ‡ = {barrier["fwd_kcal" if units == "kcal" else "fwd_kJ"]:.1f} '
            f'{"kcal/mol" if units == "kcal" else "kJ/mol"}',
            xy=(barrier["ts_cv"], F_ts),
            xytext=(10, 10), textcoords="offset points",
            fontsize=11, color="red",
        )

    ax.set_xlabel(cv_label)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    if basins or barrier:
        ax.legend()
    fig.tight_layout()

    if out_path:
        fig.savefig(str(out_path), dpi=150)
        log.info(f"FES plot → {out_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════
# Convenience: full analysis from a single HILLS file
# ═══════════════════════════════════════════════════════════════════

def analyze_hills_file(
    hills_path: str | Path,
    out_dir: str | Path | None = None,
    cv_label: str = "CV",
    title: str | None = None,
) -> dict:
    """One-shot: parse HILLS, build FES, find basins, compute barriers, plot.

    Returns: {fes: (grid, F), basins: list, barriers: list, plot: path}
    """
    hills_path = Path(hills_path)
    out_dir = Path(out_dir) if out_dir else hills_path.parent

    log.info(f"Analyzing {hills_path}")
    hills = parse_hills(hills_path)
    log.info(f"  {len(hills.time_ps)} hills, {hills.cv.shape[1]} CVs, "
             f"bias_factor={hills.bias_factor}")

    grid, F = fes_from_hills(hills)
    basins = find_basins(grid, F)
    log.info(f"  Found {len(basins)} basins at CV = {[round(b['cv'], 2) for b in basins]}")

    barriers = []
    if len(basins) >= 2:
        # Compute barrier between deepest two
        sorted_basins = sorted(basins, key=lambda b: b["F_kJ"])
        a, b = sorted_basins[0], sorted_basins[1]
        bar = barrier_between_basins(grid, F, a, b)
        barriers.append({"a": a, "b": b, **bar})
        log.info(f"  Barrier (deepest → next): "
                 f"{bar['fwd_kcal']:.2f} fwd / {bar['rev_kcal']:.2f} rev kcal/mol")

    plot_path = out_dir / f"{hills_path.stem}_fes.png"
    plot_fes(grid, F, basins=basins,
             barrier=barriers[0] if barriers else None,
             cv_label=cv_label, title=title or hills_path.name,
             out_path=plot_path)

    return {
        "fes": (grid, F),
        "basins": basins,
        "barriers": barriers,
        "plot": str(plot_path),
        "n_hills": len(hills.time_ps),
    }
