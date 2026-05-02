"""
Per-kernel-bandwidth Gaussian kernel-density estimation.

The deposited PLUMED hills (METAD or OPES KERNELS) are a sum of Gaussians
with *per-kernel* sigma — scipy's :class:`scipy.stats.gaussian_kde` only
takes a single bandwidth and so is not directly usable. This module
provides a small, vectorised KDE that:

* Accepts ``(N, d)`` centres, ``(N, d)`` sigmas, ``(N,)`` heights.
* Evaluates the density at arbitrary query points (vectorised).
* Evaluates the density on a regular grid efficiently.
* Optionally returns the gradient at query points so callers can refine
  extrema with :func:`scipy.optimize.minimize`.

Modernisation notes
-------------------
This is a port of enz-ts ``kde.py`` (which carried a homegrown KDE plus a
companion ``analyze_KDE`` class). The enz-ts version is ~300 lines for
what we do in ~120: by collapsing the ``KDE`` / ``analyze_KDE`` classes
into pure functions and letting the caller hold any state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class KernelSet:
    """A set of (centre, sigma, height) Gaussian kernels.

    The density at a point ``x`` is
    ``ρ(x) = Σᵢ hᵢ · ∏_d exp(-(xᵢ_d - x_d)² / (2 σᵢ_d²))``.

    Attributes
    ----------
    centers : ``(N, d)``
        Centre coordinates of each kernel.
    sigmas : ``(N, d)``
        Bandwidths per kernel per dimension. Anisotropic kernels supported.
    heights : ``(N,)``
        Per-kernel scalar weight (deposited PLUMED hill height in
        kJ/mol, after well-tempering scaling). Pass an array of ones for
        equal weighting.
    """
    centers: np.ndarray   # (N, d)
    sigmas: np.ndarray    # (N, d)
    heights: np.ndarray   # (N,)

    def __post_init__(self) -> None:
        self.centers = np.atleast_2d(np.asarray(self.centers, dtype=np.float64))
        self.sigmas = np.atleast_2d(np.asarray(self.sigmas, dtype=np.float64))
        self.heights = np.asarray(self.heights, dtype=np.float64).ravel()
        if self.centers.shape != self.sigmas.shape:
            raise ValueError(
                f"centers {self.centers.shape} vs sigmas {self.sigmas.shape}: "
                "must have identical shape"
            )
        if self.heights.shape != (self.centers.shape[0],):
            raise ValueError(
                f"heights {self.heights.shape} doesn't match "
                f"{self.centers.shape[0]} kernels"
            )

    @property
    def n_kernels(self) -> int:
        return self.centers.shape[0]

    @property
    def n_dims(self) -> int:
        return self.centers.shape[1]


def evaluate_density(
    kernels: KernelSet,
    X: np.ndarray,
    *,
    chunk: int = 4096,
) -> np.ndarray:
    """Evaluate the kernel-sum density at one or more points.

    Args:
        kernels: the KernelSet.
        X: query points, shape ``(M, d)`` or ``(d,)`` for a single point.
        chunk: number of query points to process per pass; tune for memory
            on very large grids. ``M_chunk × N`` doubles are allocated.

    Returns:
        ``(M,)`` density values.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[1] != kernels.n_dims:
        raise ValueError(
            f"query points have {X.shape[1]} dims, kernels have {kernels.n_dims}"
        )
    M = X.shape[0]
    out = np.empty(M, dtype=np.float64)

    centers = kernels.centers          # (N, d)
    inv_two_sig2 = 1.0 / (2.0 * kernels.sigmas ** 2)  # (N, d)
    heights = kernels.heights          # (N,)

    for start in range(0, M, chunk):
        stop = min(start + chunk, M)
        # (chunk, 1, d) - (1, N, d) -> (chunk, N, d)
        diff = X[start:stop, None, :] - centers[None, :, :]
        # multiply per-dim by 1/(2 σ²)
        squared = (diff ** 2) * inv_two_sig2[None, :, :]
        # sum over dims → (chunk, N), exponentiate, weight
        out[start:stop] = np.exp(-squared.sum(axis=2)) @ heights
    return out


def evaluate_density_grad(
    kernels: KernelSet,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Density + analytical gradient at one or more query points.

    For ``ρ(x) = Σᵢ hᵢ Πd exp(-(xd - μid)² / 2σid²)`` the gradient is
    ``∂ρ/∂x_d = -Σᵢ hᵢ Kᵢ(x) (x_d - μid)/σid²``.

    Returns:
        ``(M,)`` density and ``(M, d)`` gradient.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[1] != kernels.n_dims:
        raise ValueError("dim mismatch")
    centers = kernels.centers
    sig2 = kernels.sigmas ** 2
    diff = X[:, None, :] - centers[None, :, :]               # (M, N, d)
    K = np.exp(-((diff ** 2) / (2.0 * sig2[None, :, :])).sum(axis=2))  # (M, N)
    Kh = K * kernels.heights[None, :]                        # (M, N)
    rho = Kh.sum(axis=1)
    # gradient: -Σ Kh * (diff / sig2)
    grad = -np.einsum("MN,MNd->Md", Kh, diff / sig2[None, :, :])
    return rho, grad


def evaluate_density_grid(
    kernels: KernelSet,
    grid_axes: Iterable[np.ndarray],
    *,
    chunk: int = 16384,
) -> np.ndarray:
    """Evaluate density on a Cartesian grid.

    Args:
        kernels: KernelSet (any number of dimensions).
        grid_axes: ``d`` 1-D arrays defining the grid axes (in order).
        chunk: max query points per evaluation pass — controls memory.

    Returns:
        ``(len(g0), len(g1), ...)`` density grid.
    """
    grid_axes = list(grid_axes)
    if len(grid_axes) != kernels.n_dims:
        raise ValueError(
            f"grid_axes has {len(grid_axes)} axes, "
            f"kernels are {kernels.n_dims}-dimensional"
        )

    mesh = np.meshgrid(*grid_axes, indexing="ij")
    flat = np.stack([m.ravel() for m in mesh], axis=1)
    rho_flat = evaluate_density(kernels, flat, chunk=chunk)
    return rho_flat.reshape(mesh[0].shape)
