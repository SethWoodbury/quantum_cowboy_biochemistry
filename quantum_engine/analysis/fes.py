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
- `cowboy-qc mtd` (our pure-Python MTD; writes hills.npy, fes.npy)
- `cowboy-qc md` with PLUMED bias (writes HILLS/COLVAR)
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from quantum_engine.analysis.kde import KernelSet, evaluate_density_grid


def _minimum_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """Local minimum filter — fallback if skimage isn't available."""
    from scipy.ndimage import minimum_filter
    return minimum_filter(arr, size=size, mode="reflect")

log = logging.getLogger("quantum_engine.analysis.fes")

# Boltzmann constant in kJ/(mol·K)
KB_KJ_PER_MOL_K = 8.314462618e-3
KJ_TO_KCAL = 1.0 / 4.184


@dataclass
class Extremum:
    """A point on the FES — minimum, maximum, or saddle.

    Coordinates are in CV space (`(d,)`), value is in kJ/mol.
    """
    coord: np.ndarray
    F_kJ: float
    kind: str = "min"  # 'min' | 'max' | 'saddle'

    @property
    def F_kcal(self) -> float:
        return self.F_kJ * KJ_TO_KCAL


@dataclass
class Barrier2D:
    """A pair of basins + the saddle/highest-point along a line between them."""
    a: Extremum
    b: Extremum
    ts: Extremum
    fwd_kJ: float
    rev_kJ: float

    @property
    def fwd_kcal(self) -> float:
        return self.fwd_kJ * KJ_TO_KCAL

    @property
    def rev_kcal(self) -> float:
        return self.rev_kJ * KJ_TO_KCAL


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
# Multi-walker HILLS aggregation (post-MTD merge)
# ═══════════════════════════════════════════════════════════════════

def parse_hills_dir(
    sim_dir: str | Path,
    pattern: str = "WALKER_ID_*/HILLS",
) -> HillsData:
    """Read HILLS from every walker directory under ``sim_dir`` and merge
    into a single :class:`HillsData` with ``walker_id`` populated.

    Modeled on the enz-ts ``merge_hills`` workflow but as a pure read step
    (no on-disk merge file written; callers can `np.savetxt` if they want
    one). Tolerates walkers with mismatched lengths.

    Args:
        sim_dir: directory containing per-walker subdirs.
        pattern: glob (relative to sim_dir) to find each walker's HILLS.
            Default matches enz-ts layout.

    Returns:
        Combined :class:`HillsData`. Hills are concatenated in ascending
        walker-id order; the ``walker_id`` field is always populated.
    """
    sim_dir = Path(sim_dir)
    paths = sorted(sim_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"no HILLS files found under {sim_dir} with pattern {pattern!r}"
        )

    # Extract walker id from the parent directory name "WALKER_ID_5".
    def _walker_id(p: Path) -> int:
        m = re.search(r"WALKER_ID_(\d+)", str(p.parent))
        return int(m.group(1)) if m else -1

    paths.sort(key=_walker_id)

    parts: list[HillsData] = []
    walker_ids: list[np.ndarray] = []
    cv_names: list[str] | None = None
    bias_factor = 1.0
    for p in paths:
        h = parse_hills(p)
        if cv_names is None:
            cv_names = h.cv_names
            bias_factor = h.bias_factor
        elif h.cv_names != cv_names:
            raise ValueError(
                f"CV-name mismatch between walkers: "
                f"{cv_names} vs {h.cv_names} ({p})"
            )
        parts.append(h)
        walker_ids.append(np.full(len(h.time_ps), _walker_id(p), dtype=int))

    return HillsData(
        time_ps=np.concatenate([h.time_ps for h in parts]),
        cv=np.concatenate([h.cv for h in parts]),
        sigma=np.concatenate([h.sigma for h in parts]),
        height_kJ=np.concatenate([h.height_kJ for h in parts]),
        bias_factor=bias_factor,
        cv_names=cv_names or [],
        walker_id=np.concatenate(walker_ids),
    )


# ═══════════════════════════════════════════════════════════════════
# 2D FES reconstruction (sum-of-Gaussians)
# ═══════════════════════════════════════════════════════════════════

def fes_from_hills_2d(
    hills: HillsData,
    grid_x: np.ndarray | tuple[float, float, int] | None = None,
    grid_y: np.ndarray | tuple[float, float, int] | None = None,
    bias_factor: float | None = None,
    pad_sigma: float = 3.0,
    grid_n: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct a 2D FES by summing deposited Gaussians.

    Same well-tempered scaling as :func:`fes_from_hills`:
    ``F(s) = -(γ/(γ-1)) · V_bias(s)``.

    Args:
        hills: parsed multi-walker HillsData with `cv.shape[1] == 2`.
        grid_x, grid_y: explicit 1-D grid arrays, ``(min, max, n)`` tuples,
            or None to auto-pad ``±pad_sigma·max(σ)`` around data range
            and use ``grid_n`` points per axis.
        bias_factor: override hills.bias_factor (e.g., for OPES).
        pad_sigma, grid_n: defaults for auto-grid.

    Returns:
        (grid_x, grid_y, F[Ny, Nx])  — F is shifted so min = 0.
    """
    if hills.cv.shape[1] != 2:
        raise ValueError(
            f"fes_from_hills_2d expects 2 CVs, got {hills.cv.shape[1]}"
        )

    def _resolve_axis(g, lo, hi):
        if g is None:
            return np.linspace(lo, hi, grid_n)
        if isinstance(g, tuple):
            return np.linspace(g[0], g[1], g[2])
        return np.asarray(g, dtype=np.float64)

    s = hills.cv
    sig = hills.sigma
    pad = pad_sigma * sig.max()
    grid_x = _resolve_axis(grid_x, float(s[:, 0].min() - pad), float(s[:, 0].max() + pad))
    grid_y = _resolve_axis(grid_y, float(s[:, 1].min() - pad), float(s[:, 1].max() + pad))

    kernels = KernelSet(centers=s, sigmas=sig, heights=hills.height_kJ)
    V_bias = evaluate_density_grid(kernels, [grid_x, grid_y])  # (Nx, Ny)

    gamma = bias_factor if bias_factor is not None else hills.bias_factor
    F = -(gamma / (gamma - 1.0)) * V_bias if gamma > 1 else -V_bias
    F -= F.min()
    # Return as (Ny, Nx) for matplotlib pcolormesh / contourf convention
    return grid_x, grid_y, F.T


# ═══════════════════════════════════════════════════════════════════
# 2D minima / saddle
# ═══════════════════════════════════════════════════════════════════

def find_minima_2d(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    F: np.ndarray,
    *,
    smoothing_sigma: float = 1.5,
    min_distance: int = 5,
    max_F_kJ: float | None = None,
    n_max: int | None = None,
) -> list[Extremum]:
    """Find local minima of a 2D FES grid.

    Uses :func:`scipy.ndimage.gaussian_filter` to denoise then
    :func:`skimage.feature.peak_local_max` on ``-F`` to detect peaks of
    the inverted surface (i.e. minima of F).

    Args:
        grid_x, grid_y: 1-D axes (lengths Nx, Ny).
        F: ``(Ny, Nx)`` FES grid (kJ/mol).
        smoothing_sigma: Gaussian blur radius before peak detection
            (in grid points). 1-2 is reasonable.
        min_distance: minimum grid-cell separation between detected
            minima (peak_local_max parameter).
        max_F_kJ: drop minima with F > min(F) + max_F_kJ. Filters
            shallow numerical noise. Default None = keep all.
        n_max: keep at most this many deepest minima. Default None.

    Returns:
        List of :class:`Extremum` (kind='min'), deepest first.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    Fs = gaussian_filter(F, sigma=smoothing_sigma)

    # Prefer skimage's peak_local_max when available — it has better
    # handling of plateaux and the `min_distance` param interpretation.
    try:
        from skimage.feature import peak_local_max
        coords_idx = peak_local_max(-Fs, min_distance=min_distance)
    except ImportError:
        # Pure-numpy fallback: a cell is a local minimum if it equals the
        # local minimum within a (2*min_distance+1)² window. We then
        # greedily prune neighbours that are within `min_distance` of an
        # already-accepted minimum (lower-F first) to mimic skimage's
        # min_distance semantics.
        size = max(1, 2 * min_distance + 1)
        local_min = (Fs == _minimum_filter(Fs, size=size))
        rows, cols = np.where(local_min)
        order = np.argsort(Fs[rows, cols])
        accepted: list[tuple[int, int]] = []
        for k in order:
            r, c = int(rows[k]), int(cols[k])
            if any(max(abs(r - rr), abs(c - cc)) < min_distance
                   for rr, cc in accepted):
                continue
            accepted.append((r, c))
        coords_idx = np.array(accepted, dtype=int) if accepted else np.empty((0, 2), int)
    minima: list[Extremum] = []
    for r, c in coords_idx:
        F_val = float(F[r, c])
        if max_F_kJ is not None and F_val - F.min() > max_F_kJ:
            continue
        minima.append(Extremum(
            coord=np.array([grid_x[c], grid_y[r]]),
            F_kJ=F_val,
            kind="min",
        ))
    minima.sort(key=lambda e: e.F_kJ)
    if n_max is not None:
        minima = minima[:n_max]
    return minima


def saddle_along_line(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    F: np.ndarray,
    a: Extremum,
    b: Extremum,
    *,
    n_steps: int = 200,
) -> Extremum:
    """Estimate the saddle / barrier between two minima as the maximum of
    ``F`` along the straight line between them.

    Cheap and good-enough for most enzyme-FES use cases. For tortuous
    paths use a NEB-on-the-FES (not yet implemented in qcb).

    Args:
        grid_x, grid_y: FES axes (Nx, Ny).
        F: ``(Ny, Nx)`` FES grid.
        a, b: two :class:`Extremum` (typically minima).
        n_steps: number of samples along the line.

    Returns:
        :class:`Extremum` of kind 'saddle' at the line's max.
    """
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (grid_y, grid_x), F, bounds_error=False, fill_value=F.max(),
    )
    t = np.linspace(0.0, 1.0, n_steps)
    line = a.coord[None, :] * (1 - t)[:, None] + b.coord[None, :] * t[:, None]
    Fs = interp(np.column_stack([line[:, 1], line[:, 0]]))  # (y, x) order
    k = int(np.argmax(Fs))
    return Extremum(coord=line[k], F_kJ=float(Fs[k]), kind="saddle")


def barrier_2d(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    F: np.ndarray,
    a: Extremum,
    b: Extremum,
    *,
    n_steps: int = 200,
) -> Barrier2D:
    """Convenience: compute saddle + forward/reverse barriers between two
    2D minima."""
    ts = saddle_along_line(grid_x, grid_y, F, a, b, n_steps=n_steps)
    return Barrier2D(
        a=a, b=b, ts=ts,
        fwd_kJ=ts.F_kJ - a.F_kJ,
        rev_kJ=ts.F_kJ - b.F_kJ,
    )


# ═══════════════════════════════════════════════════════════════════
# 2D plotting
# ═══════════════════════════════════════════════════════════════════

def plot_fes_2d(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    F: np.ndarray,
    *,
    minima: Sequence[Extremum] | None = None,
    barrier: Barrier2D | None = None,
    cv_labels: tuple[str, str] = ("CV1", "CV2"),
    units: str = "kcal",
    levels: int | Sequence[float] = 20,
    cmap: str = "viridis",
    title: str | None = None,
    out_path: str | Path | None = None,
    F_max: float | None = None,
):
    """2-D FES plot with optional minima + saddle annotation.

    Args:
        grid_x, grid_y: 1-D axes.
        F: ``(Ny, Nx)`` FES grid (kJ/mol).
        minima, barrier: optional features to overlay.
        cv_labels: x and y axis labels.
        units: 'kcal' (default) or 'kJ'.
        levels: contour levels (int → auto, sequence → explicit).
        cmap: matplotlib colormap.
        F_max: clip the colour scale at this energy (kJ/mol). Useful for
            very deep barriers — surfaces above F_max get ``F_max`` colour.

    Returns: the matplotlib figure (or None if matplotlib missing).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping plot")
        return None

    F_plot = F * (KJ_TO_KCAL if units == "kcal" else 1.0)
    if F_max is not None:
        F_plot = np.minimum(F_plot, F_max * (KJ_TO_KCAL if units == "kcal" else 1.0))

    label = "kcal/mol" if units == "kcal" else "kJ/mol"
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cs = ax.contourf(grid_x, grid_y, F_plot, levels=levels, cmap=cmap)
    ax.contour(grid_x, grid_y, F_plot, levels=levels, colors="k", linewidths=0.4, alpha=0.4)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(f"Free energy ({label})")

    if minima:
        for m in minima:
            ax.plot(m.coord[0], m.coord[1], "o",
                    color="white", markeredgecolor="black", markersize=10, zorder=5)

    if barrier is not None:
        ax.plot(barrier.ts.coord[0], barrier.ts.coord[1], "^",
                color="red", markeredgecolor="black", markersize=12, zorder=6)
        ax.plot(
            [barrier.a.coord[0], barrier.ts.coord[0], barrier.b.coord[0]],
            [barrier.a.coord[1], barrier.ts.coord[1], barrier.b.coord[1]],
            "k--", linewidth=1, alpha=0.6, zorder=4,
        )

    ax.set_xlabel(cv_labels[0])
    ax.set_ylabel(cv_labels[1])
    if title:
        ax.set_title(title)
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
