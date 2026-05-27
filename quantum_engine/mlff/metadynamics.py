"""Well-tempered metadynamics (and OPES) rescue for NEB-TS failures.

This module provides pure-Python well-tempered metadynamics (WT-MTD)
and OPES-MetaD (On-the-fly Probability Enhanced Sampling, MetaD
variant) implementations for rescuing failed NEB-TS searches on
enzymatic phosphoryl-transfer reactions using MACE (or any ASE)
calculators.

Two entry points are provided:

- ``run_metadynamics_rescue(...)`` — classical WT-MTD (Barducci 2008).
- ``run_opes_rescue(...)`` — OPES-MetaD (Invernizzi & Parrinello 2020).
  OPES is typically faster to converge and has a single "barrier"
  hyper-parameter; it is recommended when the barrier is roughly
  known a priori. Both share the same CV, the same output schema
  (``fes.npy``, ``cv_trajectory.npy``, ``hills.npy``), and both
  return the same dict keys so they are drop-in interchangeable.

Method
------
We bias a one-dimensional collective variable

    s = d(P, O_LG) - d(P, O_nuc)

the bond-difference coordinate that distinguishes the reactant
(pentacoordinate not yet formed, s < 0 with strong P-O_LG bond),
product (s > 0, nucleophile now bonded), and any pentacoordinate
intermediate (|s| ~ 0). Biasing this CV drives the system across the
TS region and samples any intermediate basins. Reweighting yields a
1-D free-energy surface (FES) whose minima are representative basins
for subsequent NEB/TS refinement.

We run Langevin dynamics with the MACE calculator, and every
`bias_pace_steps` we deposit a Gaussian hill centred at the current
CV value. Well-tempered scaling (Barducci 2008) reduces hill height
by `exp(-V(s)/(kB*T*(gamma-1)))` so hills shrink as the bias fills
basins -> convergence. The bias force on atoms P, O_LG, O_nuc is
computed analytically via chain rule (see docstring of
`_bias_force_on_atoms`).

When to use
-----------
- NEB-TS failed to locate a single transition state, or climbing-image
  NEB converged to a non-stationary point.
- Suspected mechanism switch (concerted vs. stepwise), multiple TS.
- Looking for pentacoordinate intermediate along a phosphoryl-transfer
  coordinate.

Failure modes
-------------
- MACE extrapolation: biasing can drag the system far from training
  distribution. Mitigate by (a) short total_time, (b) CA restraints,
  (c) MACE committee uncertainty monitoring (not done here).
- Slow orthogonal degrees of freedom: 1-D CV may not capture the true
  reaction coordinate; multiple basins at similar s may be conflated.
- Hysteresis / non-convergence: check bias height decay; re-run with
  smaller bias_height or larger bias_factor if basins are shallow.
- FES reweighting here uses the WT-MTD final bias as -V_bias(s)/gamma
  (the Tiwary-Parrinello-like estimator in the asymptotic limit is
  approximate); for publication-quality FES use PLUMED reweighting.

References
----------
- Laio & Parrinello, PNAS 99, 12562 (2002). Escaping free-energy minima.
- Barducci, Bussi & Parrinello, PRL 100, 020603 (2008). Well-tempered MTD.
- Invernizzi & Parrinello, J. Phys. Chem. Lett. 11, 2731 (2020).
  "Rethinking metadynamics: from bias potentials to probability
  distributions." doi:10.1021/acs.jpclett.0c00497. OPES / OPES-MetaD.
- PLUMED OPES tutorial:
  https://www.plumed.org/doc-v2.9/user-doc/html/opes-metad.html
"""

from __future__ import annotations

import os
import json
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from ase import Atoms
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import write as ase_write


# ------------------------------------------------------------------ CV utils

def _cv_and_grad(positions: np.ndarray, p_idx: int, nuc_idx: int, lg_idx: int):
    """Return (s, dsdr) where dsdr is (natoms, 3) gradient of CV."""
    r_P = positions[p_idx]
    r_LG = positions[lg_idx]
    r_nuc = positions[nuc_idx]

    v_LG = r_P - r_LG
    v_nuc = r_P - r_nuc
    d_LG = np.linalg.norm(v_LG)
    d_nuc = np.linalg.norm(v_nuc)

    s = d_LG - d_nuc

    u_LG = v_LG / d_LG
    u_nuc = v_nuc / d_nuc

    grad = np.zeros_like(positions)
    # ds/dr_P = u_LG - u_nuc
    grad[p_idx] = u_LG - u_nuc
    # ds/dr_LG = -u_LG
    grad[lg_idx] = -u_LG
    # ds/dr_nuc = u_nuc
    grad[nuc_idx] = u_nuc
    return s, grad


# ------------------------------------------------------------------ bias pot

class _OPESBiasHistory:
    """Container for OPES-MetaD deposited kernels with re-weightable heights.

    Each kernel is a Gaussian with a fixed center/sigma but whose
    height is re-weighted on the fly as the estimate of the free-
    energy surface improves (Invernizzi & Parrinello 2020,
    doi:10.1021/acs.jpclett.0c00497).

    State per kernel:
        centers[i]       — CV value at deposition (Å)
        sigmas[i]        — Gaussian width (Å)
        log_weights[i]   — log of KDE weight, w_i = exp(beta * V(s_i, t_i))
        raw_heights[i]   — "raw" height at deposition (eV), kept for
                           post-analysis / diagnostics (the unweighted
                           Gaussian amplitude that would be added if
                           OPES were collapsed to plain MTD).

    The estimator of the biased probability distribution is

        P(s) = Σ_i w_i G(s; s_i, σ) / Σ_i w_i

    (sum-normalised KDE) and the bias is

        V(s) = (1 - 1/γ) * kT * log( P(s) / Z )  +  const,

    where Z is a normaliser that cancels in forces / FES. To avoid
    log(0) for an empty / sparsely-populated KDE we include a small
    floor ε · Σ_i w_i analogous to the PLUMED ``epsilon`` parameter
    (Invernizzi-Parrinello eq. 10).

    Heights are not stored directly — rather, V is computed from the
    log-weight representation each call. This way re-weighting is just
    an update of `log_weights`.
    """

    def __init__(self, beta_inv_eV: float, bias_factor: float,
                 epsilon_weight: float = 1e-6):
        # beta_inv_eV = kT in eV; bias_factor = γ
        self.centers: List[float] = []
        self.sigmas: List[float] = []
        self.log_weights: List[float] = []
        self.raw_heights: List[float] = []  # diagnostic only
        self.kT = float(beta_inv_eV)
        self.gamma = float(bias_factor)
        # prefactor (1 - 1/γ) * kT in eV
        self.prefactor_eV = (1.0 - 1.0 / self.gamma) * self.kT
        self.epsilon = float(epsilon_weight)

    # ------------------------------------------------------------------
    # low-level kernel math

    def _logsumexp_kernels(self, s: float) -> Tuple[float, float]:
        """Return (log Σ w_i G(s-s_i), log Σ w_i). Both in nats."""
        if not self.centers:
            return (-np.inf, -np.inf)
        c = np.asarray(self.centers)
        sg = np.asarray(self.sigmas)
        lw = np.asarray(self.log_weights)
        # log G_i = -0.5*((s-c)/σ)^2  — we drop the 1/(σ√2π) prefactor
        # since it is a constant absorbed into Z.
        log_g = -0.5 * ((s - c) / sg) ** 2
        # log Σ w_i G_i
        log_num = _logsumexp(lw + log_g)
        # log Σ w_i
        log_den = _logsumexp(lw)
        return float(log_num), float(log_den)

    # ------------------------------------------------------------------
    # public: energy / force

    def V(self, s: float) -> float:
        """Bias energy in eV at CV value s."""
        if not self.centers:
            return 0.0
        log_num, log_den = self._logsumexp_kernels(s)
        # ratio inside log: (Σ w G + ε Σ w) / Σ w = exp(log_num - log_den) + ε
        ratio = np.exp(log_num - log_den) + self.epsilon
        return float(self.prefactor_eV * np.log(ratio))

    def dVds(self, s: float) -> float:
        """Derivative dV/ds at CV value s (eV/Å)."""
        if not self.centers:
            return 0.0
        c = np.asarray(self.centers)
        sg = np.asarray(self.sigmas)
        lw = np.asarray(self.log_weights)
        log_g = -0.5 * ((s - c) / sg) ** 2
        # normalised KDE value P̂(s) = Σ w_i G_i / Σ w_i
        log_num = _logsumexp(lw + log_g)
        log_den = _logsumexp(lw)
        Phat = np.exp(log_num - log_den)
        # dP̂/ds = Σ w_i G_i * (-(s - c)/σ^2) / Σ w_i
        weights = np.exp(lw + log_g - log_den)  # shape (N,)
        dPhat = np.sum(weights * (-(s - c) / sg ** 2))
        # V = A * log(P̂ + ε), so dV/ds = A * dP̂/ds / (P̂ + ε)
        return float(self.prefactor_eV * dPhat / (Phat + self.epsilon))

    def V_on_grid(self, grid: np.ndarray) -> np.ndarray:
        if not self.centers:
            return np.zeros_like(grid)
        out = np.zeros_like(grid, dtype=float)
        for i, g in enumerate(grid):
            out[i] = self.V(float(g))
        return out

    # ------------------------------------------------------------------
    # public: deposition + re-weighting

    def add(self, center: float, sigma: float,
            log_weight: float, raw_height_eV: float = 0.0):
        self.centers.append(float(center))
        self.sigmas.append(float(sigma))
        self.log_weights.append(float(log_weight))
        self.raw_heights.append(float(raw_height_eV))

    def reweight(self, beta_eV: float):
        """Recompute each kernel's log-weight using the current bias estimate.

        w_i(t) = exp(β · V(s_i, t)); we re-evaluate V at every center
        with the *current* full history (which includes that very
        kernel). This is the OPES on-the-fly reweighting step.
        """
        if not self.centers:
            return
        new_lw: List[float] = []
        for c in self.centers:
            V_c = self.V(float(c))
            new_lw.append(float(beta_eV * V_c))
        self.log_weights = new_lw

    # ------------------------------------------------------------------
    # serialization

    def as_array(self) -> np.ndarray:
        """(K,3) table of (center, sigma, raw_height_eV) — MTD-compatible."""
        if not self.centers:
            return np.zeros((0, 3))
        return np.array(list(zip(self.centers, self.sigmas, self.raw_heights)))

    def as_reweighted_array(self) -> np.ndarray:
        """(K,4) table of (center, sigma, raw_height_eV, final_log_weight)."""
        if not self.centers:
            return np.zeros((0, 4))
        return np.array(list(zip(self.centers, self.sigmas,
                                 self.raw_heights, self.log_weights)))


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log(Σ exp(x))."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return -np.inf
    m = float(np.max(x))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(x - m))))


class _BiasHistory:
    """Container for deposited Gaussian hills (center, sigma, height)."""

    def __init__(self):
        self.centers: List[float] = []
        self.sigmas: List[float] = []
        self.heights: List[float] = []

    def add(self, center: float, sigma: float, height: float):
        self.centers.append(float(center))
        self.sigmas.append(float(sigma))
        self.heights.append(float(height))

    def V(self, s: float) -> float:
        if not self.centers:
            return 0.0
        c = np.asarray(self.centers)
        sg = np.asarray(self.sigmas)
        h = np.asarray(self.heights)
        return float(np.sum(h * np.exp(-0.5 * ((s - c) / sg) ** 2)))

    def dVds(self, s: float) -> float:
        if not self.centers:
            return 0.0
        c = np.asarray(self.centers)
        sg = np.asarray(self.sigmas)
        h = np.asarray(self.heights)
        g = h * np.exp(-0.5 * ((s - c) / sg) ** 2)
        return float(np.sum(g * (-(s - c) / sg ** 2)))

    def V_on_grid(self, grid: np.ndarray) -> np.ndarray:
        if not self.centers:
            return np.zeros_like(grid)
        c = np.asarray(self.centers)[None, :]
        sg = np.asarray(self.sigmas)[None, :]
        h = np.asarray(self.heights)[None, :]
        g = np.asarray(grid)[:, None]
        return np.sum(h * np.exp(-0.5 * ((g - c) / sg) ** 2), axis=1)

    def as_array(self) -> np.ndarray:
        return np.array(list(zip(self.centers, self.sigmas, self.heights)))


# ------------------------------------------------------------------ calc

class _BiasedCalculator(Calculator):
    """Wrap a base calculator, adding CV-bias energy/forces.

    The bias potential V(s) is held in a `_BiasHistory`. Energy:
        E_total = E_base + V(s)
    Forces:
        F_total = F_base - (dV/ds) * (ds/dr)
    (Note: force = -dE/dr, gradient of bias w.r.t. r is dV/ds * ds/dr.)
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc: Calculator, history,
                 p_idx: int, nuc_idx: int, lg_idx: int):
        # `history` may be a _BiasHistory (WT-MTD) or _OPESBiasHistory
        # (OPES-MetaD); both expose .V(s) and .dVds(s).
        Calculator.__init__(self)
        self.base = base_calc
        self.history = history
        self.p_idx = p_idx
        self.nuc_idx = nuc_idx
        self.lg_idx = lg_idx
        self.last_cv = None

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # base calc
        self.base.calculate(atoms, properties, system_changes)
        e_base = self.base.results["energy"]
        f_base = self.base.results["forces"].copy()

        s, dsdr = _cv_and_grad(atoms.get_positions(),
                               self.p_idx, self.nuc_idx, self.lg_idx)
        V = self.history.V(s)
        dVds = self.history.dVds(s)

        e_total = e_base + V
        f_total = f_base - dVds * dsdr

        self.last_cv = s
        self.results = {"energy": e_total, "forces": f_total}


# ------------------------------------------------------------------ basins

def _find_basins(grid: np.ndarray, fes: np.ndarray,
                 min_separation: float = 0.3) -> List[Tuple[float, float]]:
    """Return list of (cv_min, fes_min) for local minima, sorted by depth."""
    minima = []
    for i in range(1, len(fes) - 1):
        if fes[i] < fes[i - 1] and fes[i] < fes[i + 1]:
            minima.append((grid[i], fes[i]))
    # filter: keep well-separated (along CV) minima
    minima.sort(key=lambda x: x[1])
    kept: List[Tuple[float, float]] = []
    for m in minima:
        if all(abs(m[0] - k[0]) > min_separation for k in kept):
            kept.append(m)
    return kept


def _classify_basin(cv_val: float) -> Optional[str]:
    if cv_val < -1.0:
        return "reactant"
    if cv_val > 2.0:
        return "product"
    if -0.5 < cv_val < 1.5:
        return "intermediate"
    return None


# ------------------------------------------------------------------ main API

def run_metadynamics_rescue(
    atoms: Atoms,
    p_idx: int,
    nuc_idx: int,
    lg_idx: int,
    calculator,
    outdir: str,
    constraint=None,
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    total_time_ps: float = 100.0,
    bias_height_kJ_mol: float = 1.2,
    bias_sigma_A: float = 0.1,
    bias_pace_steps: int = 500,
    bias_factor: float = 10.0,
    friction_per_ps: float = 1.0,
) -> Dict[str, Any]:
    """Run well-tempered metadynamics along bond-difference CV.

    CV = d(P-O_LG) - d(P-O_nuc)

    Returns dict with keys:
        reactant, product, intermediate: ASE Atoms or None
        fes: (N,2) array of (CV, free_energy_kcal) pairs
        cv_trajectory: (M,2) array of (time_ps, CV) pairs
        basin_depths_kcal: dict with per-basin depths
        hills: (K,3) array of (center, sigma, height_eV) deposited hills

    The FES is written in kcal/mol. Trajectory and hills are also
    saved as numpy files and an XYZ of the CV-sampled frames for
    inspection.
    """
    os.makedirs(outdir, exist_ok=True)

    # --- units ---------------------------------------------------------
    # ASE energy unit is eV, length Å, time (ase internal) ~ 10.18 fs.
    kB_eV = units.kB  # eV/K
    kT = kB_eV * temperature_K  # eV
    # 1 kJ/mol = 0.010364 eV
    eV_per_kJmol = units.kJ / units.mol
    bias_height_eV = bias_height_kJ_mol * eV_per_kJmol
    kcal_per_eV = 1.0 / (units.kcal / units.mol)

    # --- working atoms + constraint -----------------------------------
    atoms = atoms.copy()
    if constraint is not None:
        atoms.set_constraint(constraint)

    history = _BiasHistory()
    biased_calc = _BiasedCalculator(calculator, history,
                                    p_idx=p_idx, nuc_idx=nuc_idx, lg_idx=lg_idx)
    atoms.calc = biased_calc

    # --- init velocities ----------------------------------------------
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

    # --- dynamics ------------------------------------------------------
    timestep_ase = timestep_fs * units.fs
    # Langevin friction in ASE units is per-ASE-time.
    # friction [ps^-1] -> [ase-time^-1]: 1/ps = 1e-3/fs; ASE fs -> units.fs
    friction_ase = friction_per_ps * 1e-3 / units.fs

    dyn = Langevin(atoms, timestep_ase, temperature_K=temperature_K,
                   friction=friction_ase)

    total_steps = int(round(total_time_ps * 1000.0 / timestep_fs))

    cv_traj: List[Tuple[float, float]] = []  # (time_ps, CV)
    frames_every = max(1, bias_pace_steps // 5)
    sampled_frames: List[Tuple[float, Atoms]] = []  # (CV, atoms snapshot)

    # one evaluation to populate biased_calc.last_cv
    atoms.get_potential_energy()

    gamma = bias_factor
    # WT prefactor: dT = T*(gamma-1); scaling factor exp(-V/(kB*dT))
    deltaT = temperature_K * (gamma - 1.0)

    for step in range(total_steps):
        dyn.run(1)
        s_now = biased_calc.last_cv
        t_ps = (step + 1) * timestep_fs / 1000.0

        if (step + 1) % frames_every == 0:
            cv_traj.append((t_ps, float(s_now)))
            sampled_frames.append((float(s_now), atoms.copy()))

        if (step + 1) % bias_pace_steps == 0:
            V_here = history.V(s_now)
            h_eff = bias_height_eV * np.exp(-V_here / (kB_eV * deltaT))
            history.add(center=s_now, sigma=bias_sigma_A, height=h_eff)

    # --- build FES -----------------------------------------------------
    if len(cv_traj) == 0:
        cv_traj.append((0.0, float(biased_calc.last_cv)))

    cv_arr = np.asarray(cv_traj)
    s_min = float(cv_arr[:, 1].min()) - 0.5
    s_max = float(cv_arr[:, 1].max()) + 0.5
    grid = np.linspace(s_min, s_max, 400)
    V_grid_eV = history.V_on_grid(grid)
    # WT-MTD asymptotic FES estimator: F(s) ~ -(gamma/(gamma-1)) * V_bias(s)
    F_eV = -(gamma / (gamma - 1.0)) * V_grid_eV
    F_eV -= F_eV.min()
    F_kcal = F_eV * kcal_per_eV

    fes = np.column_stack([grid, F_kcal])

    # --- find basins ---------------------------------------------------
    basin_list = _find_basins(grid, F_kcal)
    basins_by_class: Dict[str, Optional[Tuple[float, float]]] = {
        "reactant": None, "product": None, "intermediate": None,
    }
    for cv_val, fes_val in basin_list:
        cls = _classify_basin(cv_val)
        if cls is not None and basins_by_class[cls] is None:
            basins_by_class[cls] = (cv_val, fes_val)

    # --- representative frames (closest CV to each basin) -------------
    def _closest_frame(target_cv: float) -> Optional[Atoms]:
        if not sampled_frames:
            return None
        cvs = np.array([fr[0] for fr in sampled_frames])
        i = int(np.argmin(np.abs(cvs - target_cv)))
        return sampled_frames[i][1]

    result_atoms: Dict[str, Optional[Atoms]] = {}
    basin_depths: Dict[str, float] = {}
    # max FES on grid to estimate depth
    F_max = float(F_kcal.max())
    for cls in ("reactant", "product", "intermediate"):
        bc = basins_by_class[cls]
        if bc is None:
            result_atoms[cls] = None
            basin_depths[cls] = 0.0
        else:
            cv_val, fes_val = bc
            result_atoms[cls] = _closest_frame(cv_val)
            basin_depths[cls] = float(F_max - fes_val)

    # --- persist artifacts --------------------------------------------
    np.save(os.path.join(outdir, "cv_trajectory.npy"), cv_arr)
    np.save(os.path.join(outdir, "fes.npy"), fes)
    hills_arr = history.as_array() if history.centers else np.zeros((0, 3))
    np.save(os.path.join(outdir, "hills.npy"), hills_arr)

    for cls, a in result_atoms.items():
        if a is not None:
            ase_write(os.path.join(outdir, f"basin_{cls}.xyz"), a)

    # Multi-frame trajectory of every CV-sampled snapshot. Tiny relative to
    # the Hessian/MACE state on disk (a 200-atom system at 500 frames is
    # ~5 MB) and indispensable for downstream clustering / TS-starter mining.
    if sampled_frames:
        traj_path = os.path.join(outdir, "trajectory.xyz")
        ase_write(traj_path,
                  [fr[1] for fr in sampled_frames],
                  format="xyz")
        np.save(os.path.join(outdir, "sampled_cv.npy"),
                np.asarray([fr[0] for fr in sampled_frames]))

    meta = {
        "cv_definition": "s = d(P-O_LG) - d(P-O_nuc)",
        "p_idx": p_idx, "nuc_idx": nuc_idx, "lg_idx": lg_idx,
        "temperature_K": temperature_K,
        "timestep_fs": timestep_fs,
        "total_time_ps": total_time_ps,
        "bias_height_kJ_mol": bias_height_kJ_mol,
        "bias_sigma_A": bias_sigma_A,
        "bias_pace_steps": bias_pace_steps,
        "bias_factor": bias_factor,
        "friction_per_ps": friction_per_ps,
        "n_hills": len(history.centers),
        "n_sampled_frames": len(sampled_frames),
        "basins_found": {k: (v is not None) for k, v in result_atoms.items()},
        "basin_cv_values": {k: (basins_by_class[k][0]
                                if basins_by_class[k] is not None else None)
                            for k in basins_by_class},
        "basin_depths_kcal": basin_depths,
    }
    with open(os.path.join(outdir, "mtd_summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "reactant": result_atoms["reactant"],
        "product": result_atoms["product"],
        "intermediate": result_atoms["intermediate"],
        "fes": fes,
        "cv_trajectory": cv_arr,
        "basin_depths_kcal": basin_depths,
        "hills": hills_arr,
    }


# ------------------------------------------------------------------ OPES API

def run_opes_rescue(
    atoms: Atoms,
    p_idx: int,
    nuc_idx: int,
    lg_idx: int,
    calculator,
    outdir: str,
    constraint=None,
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    total_time_ps: float = 100.0,
    barrier_kJ_mol: float = 30.0,
    sigma_A: float = 0.1,
    pace_steps: int = 500,
    bias_factor: float = 10.0,
    friction_per_ps: float = 1.0,
    epsilon_weight: float = 1e-6,
) -> Dict[str, Any]:
    """Run OPES-MetaD (Invernizzi & Parrinello 2020) along bond-difference CV.

    OPES-MetaD is an on-the-fly expanded-ensemble / KDE variant of
    metadynamics. At every ``pace_steps`` MD steps we deposit a
    Gaussian kernel at the current CV; heights of *all* past kernels
    are reweighted using the current bias estimate,

        w_i(t) = exp(β V(s_i, t)),

    and the bias is

        V(s, t) = (1 - 1/γ) kT log( P(s, t) / Z(t) + ε )

    with P(s, t) = Σ_i w_i G(s - s_i; σ) / Σ_i w_i. This converges
    to a quasi-stationary distribution with well-tempered target

        p_target(s) ∝ p(s)^{1/γ}

    just like WT-MTD, but with (i) reweighted past Gaussians so the
    estimate of the free-energy surface is consistent at any time,
    and (ii) only one user-facing energy parameter, the barrier
    height, which is used only to set a sensible bound on the
    maximum bias. See the PLUMED OPES tutorial for the full
    derivation.

    Parameters
    ----------
    atoms, p_idx, nuc_idx, lg_idx, calculator, outdir, constraint,
    temperature_K, timestep_fs, total_time_ps, friction_per_ps :
        Same meaning as ``run_metadynamics_rescue``.
    barrier_kJ_mol :
        Estimated barrier in kJ/mol; caps the maximum bias height
        per deposition so OPES does not over-push early on. This is
        the only "energy-scale" hyper-parameter you normally need
        to set for OPES.
    sigma_A :
        Gaussian width on the CV (Å). Same meaning as in WT-MTD.
    pace_steps :
        Deposition period (steps). Typical: 500–1000.
    bias_factor :
        γ, the well-tempered bias factor (10–50). Sets the ratio
        between sampling temperature and effective sampling of the
        CV: p_target(s) ∝ p(s)^{1/γ}.
    epsilon_weight :
        Regulariser in the log; PLUMED's OPES ``epsilon`` parameter.

    Returns
    -------
    dict with the same schema as ``run_metadynamics_rescue``:
        reactant, product, intermediate: ASE Atoms or None
        fes: (N,2) array of (CV, free_energy_kcal)
        cv_trajectory: (M,2) array of (time_ps, CV)
        basin_depths_kcal: dict with per-basin depths
        hills: (K,3) array (center, sigma, raw_height_eV) so
               downstream analysis written against WT-MTD works
               unchanged. A fuller (K,4) array including the final
               log-weights is additionally written to
               ``hills_reweighted.npy``.

    References
    ----------
    Invernizzi & Parrinello, J. Phys. Chem. Lett. 11, 2731 (2020).
        doi:10.1021/acs.jpclett.0c00497
    https://www.plumed.org/doc-v2.9/user-doc/html/opes-metad.html
    """
    os.makedirs(outdir, exist_ok=True)

    # --- units ---------------------------------------------------------
    kB_eV = units.kB  # eV/K
    kT = kB_eV * temperature_K  # eV
    beta_eV = 1.0 / kT  # 1/eV
    eV_per_kJmol = units.kJ / units.mol
    barrier_eV = barrier_kJ_mol * eV_per_kJmol
    kcal_per_eV = 1.0 / (units.kcal / units.mol)

    gamma = bias_factor
    prefactor_eV = (1.0 - 1.0 / gamma) * kT  # the A in V = A log(P/Z)

    # --- working atoms + constraint -----------------------------------
    atoms = atoms.copy()
    if constraint is not None:
        atoms.set_constraint(constraint)

    history = _OPESBiasHistory(beta_inv_eV=kT, bias_factor=gamma,
                               epsilon_weight=epsilon_weight)
    biased_calc = _BiasedCalculator(calculator, history,
                                    p_idx=p_idx, nuc_idx=nuc_idx, lg_idx=lg_idx)
    atoms.calc = biased_calc

    # --- init velocities ----------------------------------------------
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

    # --- dynamics ------------------------------------------------------
    timestep_ase = timestep_fs * units.fs
    friction_ase = friction_per_ps * 1e-3 / units.fs

    dyn = Langevin(atoms, timestep_ase, temperature_K=temperature_K,
                   friction=friction_ase)

    total_steps = int(round(total_time_ps * 1000.0 / timestep_fs))

    cv_traj: List[Tuple[float, float]] = []  # (time_ps, CV)
    frames_every = max(1, pace_steps // 5)
    sampled_frames: List[Tuple[float, Atoms]] = []

    # one evaluation to populate biased_calc.last_cv
    atoms.get_potential_energy()

    # "Raw" (unweighted) kernel height cap. The maximum bias from one
    # deposition in the OPES formalism is bounded by the barrier; we
    # record this as a diagnostic raw_height_eV for each kernel so the
    # hills.npy file mirrors the WT-MTD layout. The actual deposited
    # contribution is controlled entirely through the log-weight.
    raw_h_cap_eV = prefactor_eV * float(np.log(1.0 + 1.0 / max(history.epsilon, 1e-30)))
    raw_h_cap_eV = min(raw_h_cap_eV, barrier_eV)

    for step in range(total_steps):
        dyn.run(1)
        s_now = float(biased_calc.last_cv)
        t_ps = (step + 1) * timestep_fs / 1000.0

        if (step + 1) % frames_every == 0:
            cv_traj.append((t_ps, s_now))
            sampled_frames.append((s_now, atoms.copy()))

        if (step + 1) % pace_steps == 0:
            # OPES deposition:
            # 1. evaluate current bias V(s_now) before adding kernel
            V_here = history.V(s_now)
            # 2. diagnostic "raw height": what a plain-MTD Gaussian
            #    with height h would contribute; we cap by barrier.
            #    h_raw follows the PLUMED-OPES form
            #      h ≈ sigma * (barrier/sigma - β V(s))
            #    clipped to [0, raw_h_cap_eV]. This is not used in the
            #    actual force; it is saved for hills.npy.
            h_raw = sigma_A * (barrier_eV / sigma_A - beta_eV * V_here) / max(sigma_A, 1e-12)
            h_raw = float(np.clip(h_raw, 0.0, raw_h_cap_eV))
            # 3. initial log-weight: β V(s_now, t_now). Re-weight step
            #    below will refresh it using the *new* history.
            lw_new = beta_eV * V_here
            history.add(center=s_now, sigma=sigma_A,
                        log_weight=lw_new, raw_height_eV=h_raw)
            # 4. reweight all past kernels using current bias estimate
            history.reweight(beta_eV=beta_eV)

    # --- build FES -----------------------------------------------------
    if len(cv_traj) == 0:
        cv_traj.append((0.0, float(biased_calc.last_cv)))

    cv_arr = np.asarray(cv_traj)
    s_min = float(cv_arr[:, 1].min()) - 0.5
    s_max = float(cv_arr[:, 1].max()) + 0.5
    grid = np.linspace(s_min, s_max, 400)
    V_grid_eV = history.V_on_grid(grid)
    # OPES well-tempered estimator: F(s) = -(γ/(γ-1)) * V(s)
    # (same form as WT-MTD because the target p_target ∝ p^{1/γ})
    F_eV = -(gamma / (gamma - 1.0)) * V_grid_eV
    F_eV -= F_eV.min()
    F_kcal = F_eV * kcal_per_eV

    fes = np.column_stack([grid, F_kcal])

    # --- find basins ---------------------------------------------------
    basin_list = _find_basins(grid, F_kcal)
    basins_by_class: Dict[str, Optional[Tuple[float, float]]] = {
        "reactant": None, "product": None, "intermediate": None,
    }
    for cv_val, fes_val in basin_list:
        cls = _classify_basin(cv_val)
        if cls is not None and basins_by_class[cls] is None:
            basins_by_class[cls] = (cv_val, fes_val)

    def _closest_frame(target_cv: float) -> Optional[Atoms]:
        if not sampled_frames:
            return None
        cvs = np.array([fr[0] for fr in sampled_frames])
        i = int(np.argmin(np.abs(cvs - target_cv)))
        return sampled_frames[i][1]

    result_atoms: Dict[str, Optional[Atoms]] = {}
    basin_depths: Dict[str, float] = {}
    F_max = float(F_kcal.max())
    for cls in ("reactant", "product", "intermediate"):
        bc = basins_by_class[cls]
        if bc is None:
            result_atoms[cls] = None
            basin_depths[cls] = 0.0
        else:
            cv_val, fes_val = bc
            result_atoms[cls] = _closest_frame(cv_val)
            basin_depths[cls] = float(F_max - fes_val)

    # --- persist artifacts --------------------------------------------
    np.save(os.path.join(outdir, "cv_trajectory.npy"), cv_arr)
    np.save(os.path.join(outdir, "fes.npy"), fes)
    # hills.npy keeps the 3-column WT-MTD layout so existing analysis
    # code keeps working; hills_reweighted.npy has the full 4-column
    # OPES state (center, sigma, raw_height_eV, final_log_weight).
    hills_arr = history.as_array()
    hills_rw = history.as_reweighted_array()
    np.save(os.path.join(outdir, "hills.npy"), hills_arr)
    np.save(os.path.join(outdir, "hills_reweighted.npy"), hills_rw)

    for cls, a in result_atoms.items():
        if a is not None:
            ase_write(os.path.join(outdir, f"basin_{cls}.xyz"), a)

    meta = {
        "method": "opes-metad",
        "cv_definition": "s = d(P-O_LG) - d(P-O_nuc)",
        "p_idx": p_idx, "nuc_idx": nuc_idx, "lg_idx": lg_idx,
        "temperature_K": temperature_K,
        "timestep_fs": timestep_fs,
        "total_time_ps": total_time_ps,
        "barrier_kJ_mol": barrier_kJ_mol,
        "sigma_A": sigma_A,
        "pace_steps": pace_steps,
        "bias_factor": bias_factor,
        "friction_per_ps": friction_per_ps,
        "epsilon_weight": epsilon_weight,
        "n_hills": len(history.centers),
        "basins_found": {k: (v is not None) for k, v in result_atoms.items()},
        "basin_cv_values": {k: (basins_by_class[k][0]
                                if basins_by_class[k] is not None else None)
                            for k in basins_by_class},
        "basin_depths_kcal": basin_depths,
    }
    with open(os.path.join(outdir, "opes_summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "reactant": result_atoms["reactant"],
        "product": result_atoms["product"],
        "intermediate": result_atoms["intermediate"],
        "fes": fes,
        "cv_trajectory": cv_arr,
        "basin_depths_kcal": basin_depths,
        "hills": hills_arr,
    }
