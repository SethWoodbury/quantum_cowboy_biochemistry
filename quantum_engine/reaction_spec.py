"""ReactionSpec + RunContext — the reaction-agnostic core of the TS pipeline.

NOTHING about a specific reaction (SN2-at-P, particular residues/charges/atom
indices) is hardcoded in the pipeline; the *user* describes the reaction via a
``ReactionSpec`` and the run via a ``RunContext``, and every step consumes them.
OPAA's di-Zn phosphotriesterase is only a test case.

``ReactionSpec`` (what the reaction IS):
  - ``forming_bonds`` / ``breaking_bonds``: lists of :class:`BondSpec` (atom token
    pair + optional target length) — the bonds that change.
  - ``driving_coords``: optional generic internal-coordinate drivers for SE-GSM
    (kind ∈ ADD/BREAK/STRETCH/ANGLE/DIHEDRAL/COORDINATION + atom tokens + target).
  - ``cv``: optional collective variable (e.g. bond-difference s=d(a,b)-d(a,c)).
  - ``reactive_atoms``: atoms that move along the reaction coordinate (for the
    partial Hessian / imag-mode-overlap acceptance gate).
  - ``active_region``: optional selector for the expanded-Hessian region.
  - ``atom_map``: optional reactant→product index map for double-ended methods
    (NEB/GSM assume matching atom order unless this is given).

``RunContext`` (how this RUN executes): charge, spin, calculator/engine choice,
constraints, scratch/restart dirs, units, and reusable Hessian/mode vectors
(so saddle→IRC→validate can share them).

Atom tokens (resolved to **0-based ASE indices** by :meth:`resolve`):
  - ``"5"``                 → 1-based PDB serial 5  (needs an atom list / template)
  - ``"0:5"``               → 0-based index 5       (template-free)
  - ``int``                 → 0-based index         (template-free)
  - ``"A:169:NZ"``          → chain A, resid 169, atom name NZ (needs template)
The richer forms reuse ``ops.refine_ts._resolve_reactive_indices`` lazily.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Atom-token resolution (0-based ASE index out)
# ---------------------------------------------------------------------------


def resolve_atom_token(token, atoms=None, template=None) -> int:
    """Resolve a single atom token to a 0-based ASE index.

    Template-free forms (``int``, ``"0:idx"``) never need ``atoms``/``template``.
    Other forms (1-based serial, ``RES:ID:NAME``) defer to the canonical resolver
    in ``ops.refine_ts`` (lazy import) so the grammar stays in one place.
    """
    if isinstance(token, int):
        return int(token)
    s = str(token).strip()
    if s.startswith("0:"):
        return int(s[2:])
    if s.isdigit() and atoms is None and template is None:
        # bare integer string, no structure available → treat as 0-based
        return int(s)
    # Defer richer/serial/name forms to the shared resolver
    # (canonical signature: _resolve_reactive_indices(atoms, bt_template, tokens)).
    from quantum_engine.ops.refine_ts import _resolve_reactive_indices  # noqa: PLC0415
    idxs = _resolve_reactive_indices(atoms, template, [s])
    if not idxs:
        raise ValueError(f"could not resolve atom token {token!r}")
    return int(idxs[0])


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass
class BondSpec:
    a: Any
    b: Any
    role: str = "monitor"            # "forming" | "breaking" | "monitor"
    r0: Optional[float] = None       # optional target length (Å)

    def resolve(self, atoms=None, template=None) -> tuple[int, int]:
        return (resolve_atom_token(self.a, atoms, template),
                resolve_atom_token(self.b, atoms, template))


@dataclass
class ReactionSpec:
    forming_bonds: list[BondSpec] = field(default_factory=list)
    breaking_bonds: list[BondSpec] = field(default_factory=list)
    driving_coords: list[dict] = field(default_factory=list)
    cv: Optional[dict] = None
    reactive_atoms: list[Any] = field(default_factory=list)
    active_region: Optional[str] = None
    atom_map: Optional[dict] = None

    # ---- construction ----
    @classmethod
    def from_dict(cls, d: dict) -> "ReactionSpec":
        def _bonds(key, default_role):
            out = []
            for b in (d.get(key) or []):
                if isinstance(b, (list, tuple)):
                    out.append(BondSpec(b[0], b[1], role=default_role,
                                        r0=(b[2] if len(b) > 2 else None)))
                elif isinstance(b, dict):
                    out.append(BondSpec(b["a"], b["b"], role=b.get("role", default_role),
                                        r0=b.get("r0")))
                else:
                    raise ValueError(f"bad bond entry under {key!r}: {b!r}")
            return out
        spec = cls(
            forming_bonds=_bonds("forming_bonds", "forming"),
            breaking_bonds=_bonds("breaking_bonds", "breaking"),
            driving_coords=list(d.get("driving_coords") or []),
            cv=d.get("cv"),
            reactive_atoms=list(d.get("reactive_atoms") or []),
            active_region=d.get("active_region"),
            atom_map={int(k): int(v) for k, v in (d.get("atom_map") or {}).items()} or None,
        )
        spec.validate()
        return spec

    @classmethod
    def from_yaml(cls, path) -> "ReactionSpec":
        import yaml  # noqa: PLC0415
        data = yaml.safe_load(Path(path).read_text()) or {}
        if "reaction" in data:
            data = data["reaction"]
        return cls.from_dict(data)

    # ---- validation ----
    def validate(self) -> None:
        if self.cv is not None:
            kind = self.cv.get("kind")
            if kind == "bond_difference":
                if len(self.cv.get("atoms", [])) != 3:
                    raise ValueError(
                        "cv kind='bond_difference' needs atoms=[a,b,c] "
                        "(s = d(a,b) - d(a,c))")
            elif kind is None:
                raise ValueError("cv must declare a 'kind'")
        for dc in self.driving_coords:
            if "kind" not in dc or "atoms" not in dc:
                raise ValueError(f"driving_coord needs 'kind' and 'atoms': {dc!r}")

    @property
    def changing_bonds(self) -> list[BondSpec]:
        return list(self.forming_bonds) + list(self.breaking_bonds)

    @property
    def n_changing(self) -> int:
        return len(self.changing_bonds)

    # ---- resolution to 0-based indices ----
    def resolve(self, atoms=None, template=None) -> "ResolvedReaction":
        forming = [b.resolve(atoms, template) for b in self.forming_bonds]
        breaking = [b.resolve(atoms, template) for b in self.breaking_bonds]
        reactive = [resolve_atom_token(t, atoms, template) for t in self.reactive_atoms]
        cv_idx = None
        if self.cv and self.cv.get("kind") == "bond_difference":
            cv_idx = tuple(resolve_atom_token(t, atoms, template)
                           for t in self.cv["atoms"])
        return ResolvedReaction(forming=forming, breaking=breaking,
                                reactive=reactive, cv_bond_difference=cv_idx,
                                spec=self)


@dataclass
class ResolvedReaction:
    forming: list[tuple[int, int]]
    breaking: list[tuple[int, int]]
    reactive: list[int]
    cv_bond_difference: Optional[tuple[int, int, int]]
    spec: ReactionSpec


@dataclass
class RunContext:
    """Per-run state threaded through every step."""
    charge: int = 0
    spin: int = 1                    # multiplicity (2S+1)
    model: str = "mace-omol"         # energy function alias
    head: Optional[str] = None
    engine: Optional[str] = None     # None = ASE-calc path; else 'orca'/'turbomole'/...
    device: str = "cuda"
    scratch_dir: Optional[str] = None
    restart_dir: Optional[str] = None
    constraints: list = field(default_factory=list)
    # reusable artifacts (saddle -> IRC -> validate share these when present)
    hessian: Any = None
    imag_mode: Any = None
    energy_unit: str = "eV"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "RunContext":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kw = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        ctx = cls(**kw)
        ctx.extra.update(extra)
        return ctx


__all__ = ["BondSpec", "ReactionSpec", "ResolvedReaction", "RunContext",
           "resolve_atom_token"]
