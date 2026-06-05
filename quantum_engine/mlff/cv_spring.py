"""
Bond-difference collective variable (CV) spring for unbiased endpoint generation.

Motivation
----------
Classic "nuc-only" and "both" spring modes drive individual bond distances.
These have two failure modes:
1. nuc-only: may generate unstable products that bounce back during MD
2. both: imposes a specific mechanistic ordering (concerted-synchronous)

The bond-difference CV `s = d(P-O_LG) − d(P-O_nuc)` is the natural More
O'Ferrall–Jencks (MOJ) reaction coordinate for nucleophilic substitution
at phosphorus (or any center). It encodes the *identity* of reactant vs
product without dictating *timing*:

- Reactant: nuc far (d_nuc large), LG bonded (d_LG small) → s = large negative (~ −2 Å)
- Product:  nuc bonded (d_nuc small), LG far (d_LG large) → s = large positive (~ +3 Å)
- TS:       both bonds partial → s ≈ 0
- Pentacoordinate intermediate: both bonded → s ≈ (1.7 − 1.7) = 0, but CV alone doesn't discriminate

A single spring on this CV pulls the system toward reactant (s_target < 0) or
product (s_target > 0), but does NOT force the two bonds to change together.
The PES decides whether the mechanism is concerted or stepwise.

Why this respects "no biasing"
------------------------------
The user's concern with `--spring-mode both` is that applying two independent
springs biases the mechanism toward concerted-synchronous. The CV spring applies
*one* spring on *one* 1D coordinate — the same 1D coordinate that defines the
difference between reactant and product. This is the correct reaction coordinate
for SN2-at-center; it is not "biasing both directions."

Math
----
CV: s = |r_P − r_LG| − |r_P − r_nuc|

Gradient (used by adjust_forces):
  ∂s/∂r_P  = (r_P − r_LG)/|r_P − r_LG| − (r_P − r_nuc)/|r_P − r_nuc|
  ∂s/∂r_LG = −(r_P − r_LG)/|r_P − r_LG|
  ∂s/∂r_nuc = (r_P − r_nuc)/|r_P − r_nuc|

Spring potential: U = 0.5 k (s − s_target)² with force cap fmax
Spring force:     F_atom = −(∂U/∂s) × (∂s/∂r_atom)

Recommended targets
-------------------
- Reactant: s_target = −2.0 Å (strong nuc dissociation, LG intact)
- Product:  s_target = +2.5 Å (strong LG dissociation, nuc bonded)

After the spring drives the system to the target CV value, release the spring
and relax. If the product basin is real, the geometry stays near s_target.
If not, s drifts back (diagnostic — likely a pentacoordinate intermediate
or concerted-mechanism product).

When to use
-----------
- Phosphoryl transfer / SN2-at-P (default for PTE active sites)
- Any nucleophilic substitution (SN2-at-C, SN1 with discrete intermediate)
- Proton transfer: s = d(donor-H) − d(acceptor-H)
- Ligand exchange in metal complexes
- Generalizable to any A + B-C → A-B + C reaction with identifiable A, B, C atoms

When NOT to use
---------------
- Multi-bond rearrangements that cannot be captured by a single coordinate
  (e.g., pericyclic reactions, concerted double-bond shifts)
- Electron-transfer reactions (no geometric CV describes the TS well)
- When input is already a TS guess (use IRC-from-TS instead)

References
----------
- More O'Ferrall, R.A. "Relationships between E2 and E1cb mechanisms of β-elimination."
  J. Chem. Soc. B 1970, 274. (MOJ diagram)
- Jencks, W.P. "A primer for the Bema Hapothle. An empirical approach to the
  characterization of changing transition-state structures." Chem. Rev. 1985, 85, 511.
- Bernasconi, C.F. "The principle of nonperfect synchronization." Adv. Phys. Org.
  Chem. 1992, 27, 119.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("quantum_engine.cv_spring")


class BondDifferenceCVSpring:
    """Harmonic spring on the CV s = d(P-LG) - d(P-nuc).

    Compatible with ASE's constraint interface — just pass it to atoms.set_constraint().

    Parameters
    ----------
    p_idx : int
        Index of the central atom (phosphorus)
    nuc_idx : int
        Index of the attacking nucleophile atom
    lg_idx : int
        Index of the leaving group atom
    k : float
        Spring constant in eV/Å² (default 3.0, consistent with existing
        individual-bond springs in this codebase)
    s_target : float
        Target CV value in Å.
        - Negative (−2.0 to −1.0) → drive to reactant
        - Positive (+2.0 to +3.0) → drive to product
        - 0.0 → drive toward TS (not useful; TS is found by NEB/Sella)
    fmax : float, optional
        Force magnitude cap in eV/Å (keeps spring from overwhelming PES forces).
        Default: 3.0
    mode : str
        "both": spring acts whenever s ≠ s_target
        "attractive": only when s < s_target (push to higher s)
        "repulsive":  only when s > s_target (push to lower s)
    """

    def __init__(
        self,
        p_idx: int,
        nuc_idx: int,
        lg_idx: int,
        k: float = 3.0,
        s_target: float = 2.5,
        fmax: Optional[float] = 3.0,
        mode: str = "both",
    ):
        assert mode in ("both", "attractive", "repulsive")
        self.p = p_idx
        self.nuc = nuc_idx
        self.lg = lg_idx
        self.k = k
        self.s_target = s_target
        self.fmax = fmax
        self.mode = mode

    # ASE constraint protocol

    def get_removed_dof(self, atoms):
        return 0

    def adjust_positions(self, atoms, new):
        pass

    def adjust_momenta(self, atoms, momenta):
        pass

    def todict(self):
        return {
            "name": "BondDifferenceCVSpring",
            "kwargs": {
                "p_idx": self.p,
                "nuc_idx": self.nuc,
                "lg_idx": self.lg,
                "k": self.k,
                "s_target": self.s_target,
                "fmax": self.fmax,
                "mode": self.mode,
            },
        }

    def compute_cv(self, atoms):
        """Return current CV value s = |r_P-r_LG| - |r_P-r_nuc|."""
        r_P = atoms.positions[self.p]
        r_nuc = atoms.positions[self.nuc]
        r_lg = atoms.positions[self.lg]
        d_lg = float(np.linalg.norm(r_P - r_lg))
        d_nuc = float(np.linalg.norm(r_P - r_nuc))
        return d_lg - d_nuc

    def compute_cv_gradient(self, atoms):
        """Return {atom_idx: gradient_vector}.

        ∂s/∂r_P  = u_LG - u_nuc
        ∂s/∂r_LG = -u_LG
        ∂s/∂r_nuc = u_nuc
        where u_X = (r_P - r_X) / |r_P - r_X|
        """
        r_P = atoms.positions[self.p]
        r_nuc = atoms.positions[self.nuc]
        r_lg = atoms.positions[self.lg]

        v_lg = r_P - r_lg
        v_nuc = r_P - r_nuc
        d_lg = float(np.linalg.norm(v_lg))
        d_nuc = float(np.linalg.norm(v_nuc))

        u_lg = v_lg / d_lg if d_lg > 1e-8 else np.zeros(3)
        u_nuc = v_nuc / d_nuc if d_nuc > 1e-8 else np.zeros(3)

        return {
            self.p: u_lg - u_nuc,
            self.lg: -u_lg,
            self.nuc: u_nuc,
        }

    def adjust_forces(self, atoms, forces):
        s = self.compute_cv(atoms)
        ds = self.s_target - s  # positive → need to increase s → need force pushing s up

        # Mode gating
        if self.mode == "attractive" and ds <= 0:
            return
        if self.mode == "repulsive" and ds >= 0:
            return

        # Force on CV: F_s = k × ds
        F_on_cv = self.k * ds
        if self.fmax is not None:
            F_on_cv = float(np.clip(F_on_cv, -self.fmax, self.fmax))

        # Project onto atoms via chain rule: F_atom = F_on_cv × ds/dr_atom
        grad = self.compute_cv_gradient(atoms)
        for idx, g in grad.items():
            forces[idx] += F_on_cv * g

    def adjust_potential_energy(self, atoms):
        """Return the bias energy added by this spring (for MD energy conservation)."""
        s = self.compute_cv(atoms)
        ds = self.s_target - s

        if self.mode == "attractive" and ds <= 0:
            return 0.0
        if self.mode == "repulsive" and ds >= 0:
            return 0.0

        # With force cap: above the cap, potential is linear (piecewise);
        # below the cap, quadratic
        raw_force = self.k * abs(ds)
        if self.fmax is None or raw_force <= self.fmax:
            return 0.5 * self.k * ds * ds

        # Piecewise: quadratic up to cap, then linear
        d_cap = self.fmax / self.k
        excess = abs(ds) - d_cap
        return 0.5 * self.k * d_cap * d_cap + self.fmax * excess

    def __repr__(self):
        return (f"BondDifferenceCVSpring(P={self.p}, nuc={self.nuc}, LG={self.lg}, "
                f"k={self.k}, s_target={self.s_target:.2f}, mode={self.mode})")


# Per-reaction-class CV-target HEURISTICS for the bond-difference coordinate
# s = |r_P - r_LG| - |r_P - r_nuc| (Å). These are rough endpoint-GENERATION
# guesses for a reaction class, NOT universal truths and NOT acceptance criteria
# — always verify with a quick scan. Add a class by adding an entry here; the
# pipeline never assumes one silently.
CV_TARGET_HEURISTICS: dict[str, tuple[float, float]] = {
    # OPAA / PTE di-Zn phosphotriesterase test case (SN2-at-phosphorus)
    "sn2-at-phosphorus": (-2.0, 2.5),
    # symmetric backside SN2 at carbon (e.g. Cl- + CH3Cl)
    "sn2-at-carbon": (-1.8, 1.8),
}


def suggest_cv_targets(
    reactant_s: float | None = None,
    product_s: float | None = None,
    *,
    reaction_type: str | None = None,
) -> tuple[float, float]:
    """Resolve bond-difference CV targets for endpoint generation.

    Precedence: explicit ``reactant_s``/``product_s`` win; otherwise a
    ``reaction_type`` heuristic (see :data:`CV_TARGET_HEURISTICS`) fills the gaps.
    With NEITHER an explicit value NOR a ``reaction_type``, this RAISES — there is
    no silent reaction-specific default (the old ``-2.0/+2.5`` were SN2-at-P
    heuristics and are now only returned for ``reaction_type="sn2-at-phosphorus"``).

    Args:
        reactant_s / product_s: explicit CV targets (Å). Either or both.
        reaction_type: a key of :data:`CV_TARGET_HEURISTICS` to fill any unset
            target with that class's heuristic.

    Returns:
        ``(s_reactant, s_product)``.

    Raises:
        ValueError: if a target is unset and no (valid) ``reaction_type`` supplies it.
    """
    r, p = reactant_s, product_s
    if reaction_type is not None:
        key = reaction_type.lower()
        if key not in CV_TARGET_HEURISTICS:
            raise ValueError(
                f"unknown reaction_type {reaction_type!r}; known: "
                f"{sorted(CV_TARGET_HEURISTICS)} — or pass explicit "
                f"reactant_s + product_s.")
        hr, hp = CV_TARGET_HEURISTICS[key]
        r = hr if r is None else r
        p = hp if p is None else p
    if r is None or p is None:
        raise ValueError(
            "suggest_cv_targets: pass explicit reactant_s AND product_s, or a "
            f"reaction_type (known: {sorted(CV_TARGET_HEURISTICS)}). No "
            "reaction-specific default is assumed — run a quick scan if unsure.")
    return (float(r), float(p))
