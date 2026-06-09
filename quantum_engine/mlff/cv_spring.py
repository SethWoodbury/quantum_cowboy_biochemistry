"""
Bond-distance collective variables (CVs) for endpoint generation and biasing.

General CV (reaction-agnostic)
------------------------------
The collective variable is a WEIGHTED SUM of pairwise bond distances::

    s = Σ_{breaking bonds} d(i,j)  −  Σ_{forming bonds} d(k,l)

i.e. every term is a pairwise distance carrying a coefficient: ``+1`` for a bond
that BREAKS, ``−1`` for a bond that FORMS. With this sign convention ``s``
increases monotonically along the reaction, independent of the mechanism::

    reactant  → breaking bonds short, forming bonds long  → s ≪ 0
    TS        → bonds partial                             → s ≈ 0  (not guaranteed)
    product   → breaking bonds long,  forming bonds short → s ≫ 0

This single primitive subsumes many mechanism classes WITHOUT hardcoding any of
them — nothing here assumes phosphorus, a nucleophile, a metal, or a charge:

  - **Substitution at a center** (1 break + 1 form sharing a central atom):
    ``s = d(center, breaking) − d(center, forming)``. The common case; e.g. SN2
    (at C or P), ligand exchange. Build it with ``center/breaking/forming``.
  - **Multi-bond / concerted** (pericyclic, double substitution, E2): several
    breaking and/or forming terms summed. Build with :meth:`from_bonds`.
  - **Association / "click" chemistry** (bonds FORM, none break): forming terms
    only → ``s = −Σ d(forming)`` (still monotonic: long → short ⇒ −large → −small).
  - **Dissociation / fragmentation** (bonds BREAK, none form): breaking terms only.
  - **No shared center**: the two bonds need not share an atom (``s = d(a,b) − d(c,d)``).

A single harmonic spring on this 1-D coordinate drives the system toward a
reactant (``s_target < 0``) or product (``s_target > 0``) basin WITHOUT dictating
the mechanism's *timing* — the PES decides concerted vs stepwise. (Applying
independent springs to each individual bond, by contrast, biases toward a
concerted-synchronous path; one spring on this one difference coordinate does not.)

Driving the CV from a :class:`~quantum_engine.reaction_spec.ReactionSpec`
-----------------------------------------------------------------------
``ResolvedReaction`` already lists ``forming``/``breaking`` bonds (and an optional
explicit bond-difference ``cv``). :meth:`BondDifferenceCVSpring.from_resolved_reaction`
turns those directly into CV terms — so a user describes the reaction once and the
CV follows, for any mechanism, with no per-reaction code.

Math
----
For terms ``[(i, j, w), …]``::

    s          = Σ w · |r_i − r_j|
    ∂s/∂r_i   = +w · (r_i − r_j)/|r_i − r_j|
    ∂s/∂r_j   = −w · (r_i − r_j)/|r_i − r_j|

Spring potential: ``U = 0.5 k (s − s_target)²`` (quadratic below the optional
force cap ``fmax``, linear above it). Spring force: ``F_atom = −(∂U/∂s)·∂s/∂r_atom``.

When NOT to use a single bond-distance CV
-----------------------------------------
- Rearrangements no linear combination of bond distances can capture (some
  electrocyclic ring closures need an angle/dihedral CV instead).
- Electron-transfer (no geometric CV describes the TS well).
- When the input is already a TS guess (use IRC-from-TS instead).

References
----------
- More O'Ferrall, R.A. J. Chem. Soc. B 1970, 274. (MOJ diagram)
- Jencks, W.P. Chem. Rev. 1985, 85, 511.
- Bernasconi, C.F. Adv. Phys. Org. Chem. 1992, 27, 119.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import numpy as np

log = logging.getLogger("quantum_engine.cv_spring")

# A CV term is (atom_i, atom_j, weight). s = Σ weight · |r_i − r_j|.
BondTerm = tuple[int, int, float]


# ---------------------------------------------------------------------------
# General CV primitive — the single source of truth for the CV math.
# Shared by the endpoint spring (below) and the metadynamics bias calculator.
# ---------------------------------------------------------------------------
def bond_distance_cv(positions, terms: Sequence[BondTerm]):
    """Weighted bond-distance CV ``s = Σ w·|r_i − r_j|`` and its gradient.

    Args:
        positions: (N,3) array-like of Cartesian coordinates.
        terms: iterable of ``(i, j, weight)`` — ``+1`` for breaking bonds,
            ``−1`` for forming bonds (so ``s`` grows reactant→product).

    Returns:
        ``(s, grad)`` where ``s`` is a float and ``grad`` is an (N,3) ndarray
        with ``grad[a] = ∂s/∂r_a``. A degenerate (near-zero-length) term
        contributes 0 to the gradient (the direction is undefined there).
    """
    pos = np.asarray(positions, dtype=float)
    grad = np.zeros_like(pos)
    s = 0.0
    for i, j, w in terms:
        v = pos[i] - pos[j]
        d = float(np.linalg.norm(v))
        s += w * d
        if d > 1e-8:
            u = (w / d) * v
            grad[i] += u
            grad[j] -= u
    return s, grad


def bond_cv_terms_from_roles(center_idx: int, breaking_idx: int,
                             forming_idx: int) -> list[BondTerm]:
    """Shared-center substitution CV: ``s = d(center,breaking) − d(center,forming)``.

    The common single-center case (SN2-at-C/P, ligand exchange). ``breaking`` is
    the bond that lengthens (coeff ``+1``), ``forming`` the bond that shortens
    (coeff ``−1``).
    """
    return [(int(center_idx), int(breaking_idx), 1.0),
            (int(center_idx), int(forming_idx), -1.0)]


def bond_cv_terms_from_bonds(breaking_bonds: Iterable[tuple[int, int]] = (),
                             forming_bonds: Iterable[tuple[int, int]] = (),
                             ) -> list[BondTerm]:
    """General CV terms from lists of breaking/forming bonds (atom-index pairs).

    Handles multi-bond, no-shared-center, forming-only ("click"), and
    breaking-only (dissociation). Each breaking bond gets coeff ``+1``, each
    forming bond ``−1``. Raises if BOTH lists are empty.
    """
    terms: list[BondTerm] = [(int(i), int(j), 1.0) for i, j in breaking_bonds]
    terms += [(int(i), int(j), -1.0) for i, j in forming_bonds]
    if not terms:
        raise ValueError(
            "bond_cv_terms_from_bonds: need at least one breaking or forming bond")
    return terms


class BondDifferenceCVSpring:
    """Harmonic spring on the weighted bond-distance CV ``s = Σ w·d(i,j)``.

    Compatible with ASE's constraint interface — pass it to
    ``atoms.set_constraint()``.

    Construct it three ways:

    - **Shared-center (common case)** — pass ``center_idx`` + ``breaking_idx`` +
      ``forming_idx``: ``s = d(center,breaking) − d(center,forming)``.
    - **General bond lists** — :meth:`from_bonds(breaking=[(i,j),…],
      forming=[(k,l),…])`: multi-bond / no-center / forming-only / breaking-only.
    - **From a resolved ReactionSpec** — :meth:`from_resolved_reaction(resolved)`.

    Or pass ``terms=[(i,j,w),…]`` directly for full control.

    Parameters
    ----------
    center_idx, breaking_idx, forming_idx : int, optional
        Shared-center convenience (all three required together).
    terms : sequence of (int, int, float), optional
        Explicit CV terms; mutually exclusive with the ``*_idx`` trio.
    k : float
        Spring constant in eV/Å² (default 3.0).
    s_target : float
        Target CV value in Å. Negative → drive to reactant; positive → product.
    fmax : float, optional
        Force-magnitude cap in eV/Å (keeps the spring from overwhelming the PES).
        Default 3.0; ``None`` for an uncapped quadratic spring.
    mode : str
        ``"both"`` (spring acts whenever ``s ≠ s_target``), ``"attractive"``
        (only when ``s < s_target``), ``"repulsive"`` (only when ``s > s_target``).
    """

    def __init__(
        self,
        center_idx: Optional[int] = None,
        breaking_idx: Optional[int] = None,
        forming_idx: Optional[int] = None,
        *,
        terms: Optional[Sequence[BondTerm]] = None,
        k: float = 3.0,
        s_target: float = 2.5,
        fmax: Optional[float] = 3.0,
        mode: str = "both",
    ):
        if mode not in ("both", "attractive", "repulsive"):
            raise ValueError(f"mode must be both/attractive/repulsive, got {mode!r}")
        if terms is None:
            if center_idx is None or breaking_idx is None or forming_idx is None:
                raise ValueError(
                    "provide center_idx+breaking_idx+forming_idx (shared-center), "
                    "or terms=[(i,j,w),…], or use from_bonds()/from_resolved_reaction()")
            terms = bond_cv_terms_from_roles(center_idx, breaking_idx, forming_idx)
        self.terms: list[BondTerm] = [(int(i), int(j), float(w)) for i, j, w in terms]
        if not self.terms:
            raise ValueError("BondDifferenceCVSpring needs at least one bond term")
        self.k = k
        self.s_target = s_target
        self.fmax = fmax
        self.mode = mode

    # ---- alternative constructors ----
    @classmethod
    def from_bonds(cls, breaking: Iterable[tuple[int, int]] = (),
                   forming: Iterable[tuple[int, int]] = (), **kw) -> "BondDifferenceCVSpring":
        """Build from lists of breaking/forming bonds (atom-index pairs).

        Handles multi-bond, no-shared-center, forming-only ("click"), and
        breaking-only reactions.
        """
        return cls(terms=bond_cv_terms_from_bonds(breaking, forming), **kw)

    @classmethod
    def from_resolved_reaction(cls, resolved, **kw) -> "BondDifferenceCVSpring":
        """Build from a :class:`~quantum_engine.reaction_spec.ResolvedReaction`.

        Prefers an explicit bond-difference ``cv`` (``cv_bond_difference =
        (center, breaking, forming)``) if the spec declares one; otherwise uses
        the resolved ``breaking``/``forming`` bond lists (any mechanism).
        """
        cvbd = getattr(resolved, "cv_bond_difference", None)
        if cvbd is not None:
            center, breaking, forming = cvbd
            return cls(terms=bond_cv_terms_from_roles(center, breaking, forming), **kw)
        return cls.from_bonds(breaking=getattr(resolved, "breaking", ()),
                              forming=getattr(resolved, "forming", ()), **kw)

    # ---- ASE constraint protocol ----
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
                "terms": [list(t) for t in self.terms],
                "k": self.k,
                "s_target": self.s_target,
                "fmax": self.fmax,
                "mode": self.mode,
            },
        }

    # ---- CV evaluation ----
    def compute_cv(self, atoms) -> float:
        """Return the current CV value ``s = Σ w·d(i,j)``."""
        s, _ = bond_distance_cv(atoms.positions, self.terms)
        return s

    def compute_cv_gradient(self, atoms) -> dict[int, np.ndarray]:
        """Return ``{atom_idx: ∂s/∂r_atom}`` (only atoms appearing in a term)."""
        _, grad = bond_distance_cv(atoms.positions, self.terms)
        idxs = {i for i, _, _ in self.terms} | {j for _, j, _ in self.terms}
        return {i: grad[i] for i in idxs}

    # ---- spring force / energy ----
    def adjust_forces(self, atoms, forces):
        s, grad = bond_distance_cv(atoms.positions, self.terms)
        ds = self.s_target - s  # >0 → push s up toward target
        if self.mode == "attractive" and ds <= 0:
            return
        if self.mode == "repulsive" and ds >= 0:
            return
        f_on_cv = self.k * ds
        if self.fmax is not None:
            f_on_cv = float(np.clip(f_on_cv, -self.fmax, self.fmax))
        forces += f_on_cv * grad

    def adjust_potential_energy(self, atoms):
        """Bias energy added by the spring (for MD energy bookkeeping)."""
        s = self.compute_cv(atoms)
        ds = self.s_target - s
        if self.mode == "attractive" and ds <= 0:
            return 0.0
        if self.mode == "repulsive" and ds >= 0:
            return 0.0
        raw_force = self.k * abs(ds)
        if self.fmax is None or raw_force <= self.fmax:
            return 0.5 * self.k * ds * ds
        # piecewise: quadratic up to the cap, linear beyond it
        d_cap = self.fmax / self.k
        excess = abs(ds) - d_cap
        return 0.5 * self.k * d_cap * d_cap + self.fmax * excess

    def __repr__(self):
        return (f"BondDifferenceCVSpring(terms={self.terms}, k={self.k}, "
                f"s_target={self.s_target:.2f}, mode={self.mode})")


# Per-reaction-class CV-target HEURISTICS for the bond-difference coordinate (Å).
# These are rough endpoint-GENERATION guesses for a reaction class, NOT universal
# truths and NOT acceptance criteria — always verify with a quick scan. Add a
# class by adding an entry here; the pipeline never assumes one silently.
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


__all__ = [
    "BondTerm", "bond_distance_cv", "bond_cv_terms_from_roles",
    "bond_cv_terms_from_bonds", "BondDifferenceCVSpring",
    "CV_TARGET_HEURISTICS", "suggest_cv_targets",
]
