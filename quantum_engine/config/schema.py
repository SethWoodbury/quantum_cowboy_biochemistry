"""
Pydantic YAML config schema for qcb.

Replaces the verbose, string-heavy enz-ts YAML format with named entities
that reference each other by name. Supports file inheritance (defaults +
overrides) via a top-level ``defaults`` key or the ``defaults`` argument to
:func:`load_config`.

Pydantic compatibility
----------------------
Written against the **pydantic v1** API (``validator`` / ``root_validator`` /
``class Config`` / ``parse_obj``), because the production container
``quantum_chem-*.sif`` pins ``pydantic<2`` (global dependency hard-pinning to
keep fairchem/torch/numpy from drifting under MACE/SCINE). These v1-style
APIs also run under pydantic v2 (via its v1 compatibility layer, with
deprecation warnings), so the schema is portable across both.

Schema overview
---------------
A qcb config has these top-level fields:

* ``qcb_version``  — schema version (currently 1)
* ``name`` / ``description`` — optional metadata
* ``structure`` — input structure (PDB/XYZ/CIF), charge, and ligand names
* ``selectors`` — named atom selectors using the grammar in
  :mod:`quantum_engine.select` (``residue YYL``, ``atoms P1``, ``chain B``,
  ``resid 100``, ``range 0 99``, ``element C``, ``all``, ``none``)
* ``geometry`` — named geometric quantities (bonds, angles, dihedrals, or
  bond-difference CVs); their ``atoms`` field references selector names
* ``constraints`` — list of ASE constraints applied during MD/opt
* ``calculator`` — MLFF calculator config (model alias, head, device, dtype)
* ``operation`` — a tagged union; the ``kind`` field picks the subclass
  (opt, md, mtd, umbrella, scan, freq, irc, neb, ts)

Cross-references are validated:

* every name listed in a ``GeometrySpec.atoms`` must be a defined selector
* a constraint's ``geom`` and an operation's ``cv`` / ``geom`` must name a
  defined geometry
* a constraint's ``selector`` must name a defined selector
* operations referencing selectors by name (``cv_atoms``, ``atoms`` for
  ``bond_difference_cv``) must point at defined selectors

Example
-------
.. code-block:: yaml

    qcb_version: 1
    name: yyl_phosphoryl_transfer_R2

    structure:
      path: /path/to/active_site.pdb
      charge: -2
      ligand_names: [YYL]

    selectors:
      P1: "residue YYL atoms P1"          # raw-string shorthand
      O_nuc: "residue ASP atoms OD2 resid 165"

    geometry:
      d_PO_bond:
        kind: distance
        atoms: [P1, O_nuc]

    constraints:
      - kind: harmonic_restraint
        geom: d_PO_bond
        k: 5.0
        r0: 1.65

    calculator:
      model: mace-mh-1
      head: omol
      device: cuda

    operation:
      kind: ts
      strategy: cv-spring
      n_images: 15

Programmatic usage
------------------
.. code-block:: python

    from quantum_engine.config.schema import load_config, normalize_selectors

    cfg = load_config("override.yaml", defaults=["defaults.yaml"])
    selectors = normalize_selectors(cfg)          # {name: spec_string}
    if cfg.operation and cfg.operation.kind == "ts":
        print(cfg.operation.strategy)             # "cv-spring"
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, root_validator, validator


# ---------------------------------------------------------------------------
# Selector grammar — must stay in sync with quantum_engine/select.py
# ---------------------------------------------------------------------------

_VALID_SELECTOR_TOKENS: frozenset = frozenset(
    {"residue", "resid", "chain", "atoms", "element", "range", "all", "none"}
)


def _validate_selector_string(spec: str) -> str:
    """Cheap structural check for selector strings (no biotite handle).

    We only verify that the leading token is recognised by the grammar in
    :mod:`quantum_engine.select`. Full semantic validation happens at runtime
    against the real structure.
    """
    s = spec.strip()
    if not s:
        raise ValueError("selector spec is empty")
    head = s.split()[0].lower()
    if head not in _VALID_SELECTOR_TOKENS:
        raise ValueError(
            f"selector spec '{spec}' starts with unknown token '{head}'. "
            f"Valid leading tokens: {sorted(_VALID_SELECTOR_TOKENS)}"
        )
    return s


# ---------------------------------------------------------------------------
# Leaf models
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Base for all schema models.

    Extras are IGNORED (silently dropped) rather than forbidden. This tolerates
    unknown keys from legacy-config translation, allows forward-compat (newer
    configs with extra fields load on older code), and keeps
    ``validate_assignment`` so explicit setattr still type-checks.
    """

    class Config:
        extra = "ignore"
        validate_assignment = True


class StructureSpec(_StrictModel):
    """Input structure: file path plus chemical metadata."""

    path: str = Field(..., description="Path to PDB / XYZ / CIF file")
    charge: int = Field(0, description="Net charge of the system (electrons)")
    ligand_names: list[str] = Field(
        default_factory=list,
        description="Residue names of any non-standard ligands (e.g. ['YYL']).",
    )


class SelectorSpec(_StrictModel):
    """Named atom selector; ``spec`` follows :mod:`quantum_engine.select` grammar."""

    spec: str = Field(..., description="Selector spec string, e.g. 'residue YYL atoms P1'")

    @validator("spec")
    def _check_spec(cls, v):
        return _validate_selector_string(v)


class GeometrySpec(_StrictModel):
    """A named geometric quantity built from selector references.

    The ``atoms`` list contains *selector names* (not raw atom names): each
    must resolve to a defined :class:`SelectorSpec` in the parent
    :class:`QcbConfig`. Cardinality depends on ``kind``:

    * ``distance``        — 2 selectors
    * ``angle``           — 3 selectors (vertex in the middle)
    * ``dihedral``        — 4 selectors
    * ``distance_diff``   — 3 selectors: ``d(a, b) - d(a, c)``
    """

    kind: Literal["distance", "angle", "dihedral", "distance_diff"]
    atoms: list[str] = Field(..., description="Selector names referenced by this geometry")
    log: bool = Field(True, description="Log this quantity to COLVAR during dynamics")

    @root_validator(skip_on_failure=True)
    def _check_arity(cls, values):
        kind = values.get("kind")
        atoms = values.get("atoms") or []
        expected = {"distance": 2, "angle": 3, "dihedral": 4, "distance_diff": 3}.get(kind)
        if expected is not None and len(atoms) != expected:
            raise ValueError(
                f"GeometrySpec(kind='{kind}') needs {expected} selector names, "
                f"got {len(atoms)}: {atoms}"
            )
        return values


class ConstraintSpec(_StrictModel):
    """An ASE constraint to apply during MD/optimisation.

    Different ``kind`` values use different fields:

    * ``fix_atoms``           — uses ``selector``
    * ``harmonic_restraint``  — uses ``geom``, ``k``, ``r0``, optional ``mode``
    * ``harmonic_walls``      — uses ``geom``, ``k``, ``r0``, ``mode``
    * ``harmonic_init``       — uses ``selector`` (anchor atoms at initial pos)
    * ``bond_difference_cv``  — uses ``atoms`` (3 selector names: P, nuc, LG)
    """

    kind: Literal["fix_atoms", "harmonic_restraint", "harmonic_walls",
                  "harmonic_init", "bond_difference_cv"]
    name: Optional[str] = Field(None, description="Optional human-readable label")
    selector: Optional[str] = Field(
        None, description="Selector name (for fix_atoms / harmonic_init)",
    )
    geom: Optional[str] = Field(None, description="Geometry name (for harmonic_*)")
    snapshot: bool = Field(
        False,
        description="harmonic_init: anchor each atom to its initial position "
                    "(equivalent to ASE Hookean restraint per atom)",
    )
    atoms: Optional[list[str]] = Field(
        None, description="Selector names (e.g. [P, nuc, LG] for bond_difference_cv)",
    )
    k: float = Field(0.0, description="Spring constant (eV/A^2 or eV/rad^2)")
    r0: float = Field(0.0, description="Equilibrium value or wall centre")
    mode: Literal["attractive", "repulsive", "both"] = "both"
    fmax: Optional[float] = Field(None, description="Optional force cap on this constraint")

    @root_validator(skip_on_failure=True)
    def _check_kind_fields(cls, values):
        kind = values.get("kind")
        if kind == "fix_atoms":
            if not values.get("selector"):
                raise ValueError("ConstraintSpec(kind='fix_atoms') requires 'selector'")
        elif kind == "harmonic_init":
            if not values.get("selector"):
                raise ValueError(
                    "ConstraintSpec(kind='harmonic_init') requires 'selector' "
                    "(which atoms to anchor at initial positions)"
                )
        elif kind in {"harmonic_restraint", "harmonic_walls"}:
            if not values.get("geom"):
                raise ValueError(
                    f"ConstraintSpec(kind='{kind}') requires 'geom' "
                    "(name of a GeometrySpec)"
                )
        elif kind == "bond_difference_cv":
            atoms = values.get("atoms")
            if not atoms or len(atoms) != 3:
                raise ValueError(
                    "ConstraintSpec(kind='bond_difference_cv') requires "
                    "atoms=[P, nuc, LG] (3 selector names)"
                )
        return values


class CalculatorSpec(_StrictModel):
    """MLFF calculator settings (consumed by :mod:`quantum_engine.calc` factories)."""

    model: str = Field("mace-omol", description="Model alias from quantum_engine.calc factory")
    head: Optional[str] = Field(None, description="Head name for multi-head models")
    device: Literal["cuda", "cpu"] = "cuda"
    dtype: Literal["float32", "float64"] = "float64"


# ---------------------------------------------------------------------------
# Operation tagged union (discriminated by the Literal ``kind`` field)
# ---------------------------------------------------------------------------


class _OperationBase(_StrictModel):
    """Base for operation specs. Subclasses set ``kind`` as a Literal."""

    kind: str


class OptOperation(_OperationBase):
    kind: Literal["opt"] = "opt"
    optimizer: Literal["lbfgs", "bfgs", "fire"] = "lbfgs"
    fmax: float = 0.05
    max_steps: int = 500


class MDOperation(_OperationBase):
    kind: Literal["md"] = "md"
    ensemble: Literal["langevin_nvt", "verlet_nve"] = "langevin_nvt"
    timestep_fs: float = 1.0
    total_time_ps: float = 10.0
    temperature_K: float = 300.0
    friction_per_ps: float = 1.0


class MTDOperation(_OperationBase):
    kind: Literal["mtd"] = "mtd"
    variant: Literal["wt", "opes"] = "wt"
    cv: str = Field(..., description="Name of a GeometrySpec used as the CV")
    backend: Literal["pure_python", "plumed"] = "pure_python"
    bias_factor: float = 10.0
    sigma_A: float = 0.1
    pace_steps: int = 500
    total_time_ps: float = 50.0
    temperature_K: float = 300.0
    friction_per_ps: float = 1.0
    walkers: int = 1


class UmbrellaOperation(_OperationBase):
    kind: Literal["umbrella"] = "umbrella"
    cv: str = Field(..., description="Name of a GeometrySpec used as the CV")
    centers: list[float] = Field(..., description="Window centres along the CV")
    k: float = Field(..., description="Restraint k (kJ/mol/A^2)")
    total_time_ps_per_window: float = 5.0
    temperature_K: float = 300.0


class ScanOperation(_OperationBase):
    kind: Literal["scan"] = "scan"
    geom: str = Field(..., description="Name of a GeometrySpec to scan")
    start: float
    end: float
    n_steps: int = 15
    relax_other: bool = True


class FreqOperation(_OperationBase):
    kind: Literal["freq"] = "freq"
    indices_selector: Optional[str] = Field(
        None, description="Selector name for partial Hessian (None = full system)"
    )
    delta: float = 0.02
    method: Literal["central", "forward"] = "central"
    temperature_K: float = 298.15


class IRCOperation(_OperationBase):
    kind: Literal["irc"] = "irc"
    refine_ts: bool = True
    saddle_fmax: float = 0.02
    irc_step: float = 0.1
    irc_fmax: float = 0.03


class NEBOperation(_OperationBase):
    kind: Literal["neb"] = "neb"
    reactant: str = Field(..., description="Path to reactant PDB")
    product: str = Field(..., description="Path to product PDB")
    n_images: int = 15
    k_spring: float = 1.0
    interpolation: Literal["geodesic", "idpp", "linear"] = "geodesic"


class TSOperation(_OperationBase):
    kind: Literal["ts"] = "ts"
    strategy: Literal["legacy", "irc", "cv-spring", "mtd"] = "cv-spring"
    n_images: int = 15
    interpolation: str = "geodesic"
    cv_atoms: Optional[list[str]] = Field(
        None, description="Selector names [P, nuc, LG] for cv-spring/mtd strategies",
    )
    cv_s_reactant: float = -2.0
    cv_s_product: float = 2.5


# A plain Union; each member's ``Literal`` kind self-discriminates under
# pydantic v1's left-to-right Union matching (no explicit discriminator needed).
_OperationUnion = Union[
    OptOperation, MDOperation, MTDOperation, UmbrellaOperation, ScanOperation,
    FreqOperation, IRCOperation, NEBOperation, TSOperation,
]


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class QcbConfig(_StrictModel):
    """Top-level qcb config. Validates internal cross-references."""

    qcb_version: int = 1
    name: Optional[str] = None
    description: Optional[str] = None

    # `structure` and `operation` are required for FULL configs but allowed to
    # be None on defaults-only files (intended for inheritance via `defaults:`).
    structure: Optional[StructureSpec] = None
    selectors: dict[str, SelectorSpec] = Field(default_factory=dict)
    geometry: dict[str, GeometrySpec] = Field(default_factory=dict)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    calculator: CalculatorSpec = Field(default_factory=CalculatorSpec)
    operation: Optional[_OperationUnion] = None

    @validator("operation", pre=True)
    def _require_operation_kind(cls, v):
        """Reject an operation dict with no ``kind``.

        Without an explicit discriminator, pydantic v1's plain-Union matching
        would otherwise silently accept a kind-less mapping as the first member
        (OptOperation). The v2 discriminated union rejected this, so we restore
        that behavior explicitly.
        """
        if isinstance(v, dict) and "kind" not in v:
            raise ValueError(
                "operation must declare a 'kind' (one of opt/md/mtd/umbrella/"
                "scan/freq/irc/neb/ts)"
            )
        return v

    @validator("selectors", pre=True)
    def _normalize_selectors(cls, v):
        """Allow shorthand: ``selectors: {P1: "residue YYL atoms P1"}``."""
        if not isinstance(v, dict):
            return v
        out: dict = {}
        for name, val in v.items():
            out[name] = {"spec": val} if isinstance(val, str) else val
        return out

    @root_validator(skip_on_failure=True)
    def _validate_references(cls, values):
        selectors = values.get("selectors") or {}
        geometry = values.get("geometry") or {}
        constraints = values.get("constraints") or []
        operation = values.get("operation")
        sel_names = sorted(selectors.keys())
        geom_names = sorted(geometry.keys())

        def _check_selector(ref, where):
            if ref not in selectors:
                raise ValueError(
                    f"{where} references undefined selector '{ref}'. "
                    f"Defined selectors: {sel_names or '(none)'}"
                )

        def _check_geometry(ref, where):
            if ref not in geometry:
                raise ValueError(
                    f"{where} references undefined geometry '{ref}'. "
                    f"Defined geometries: {geom_names or '(none)'}"
                )

        # Geometry → selectors
        for gname, geom in geometry.items():
            for atom_ref in geom.atoms:
                _check_selector(atom_ref, f"Geometry '{gname}' (atom)")

        # Constraints → selectors / geometry
        for i, c in enumerate(constraints):
            tag = f"Constraint #{i} (kind='{c.kind}')"
            if c.kind == "fix_atoms":
                _check_selector(c.selector, tag)
            elif c.kind in {"harmonic_restraint", "harmonic_walls"}:
                _check_geometry(c.geom, tag)
            elif c.kind == "bond_difference_cv":
                for a in (c.atoms or []):
                    _check_selector(a, tag)

        # Operation → selectors / geometry (skipped if defaults-only config)
        if operation is not None:
            op = operation
            op_tag = f"Operation(kind='{op.kind}')"
            if isinstance(op, MTDOperation):
                _check_geometry(op.cv, op_tag)
            elif isinstance(op, UmbrellaOperation):
                _check_geometry(op.cv, op_tag)
            elif isinstance(op, ScanOperation):
                _check_geometry(op.geom, op_tag)
            elif isinstance(op, FreqOperation):
                if op.indices_selector is not None:
                    _check_selector(op.indices_selector, op_tag)
            elif isinstance(op, TSOperation):
                for a in (op.cv_atoms or []):
                    _check_selector(a, op_tag)

        return values


# ---------------------------------------------------------------------------
# YAML loading + inheritance
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (later wins).

    Mappings are merged key-by-key; everything else (including lists) is
    replaced wholesale. Returns a new dict; inputs are not mutated.
    """
    out: dict = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Top-level of {p} must be a YAML mapping, got {type(data).__name__}"
        )
    return data


def load_config(path, defaults=None) -> QcbConfig:
    """Load a YAML config, optionally inheriting from one or more defaults files.

    Resolution order (later wins):

    1. each path in the ``defaults`` argument, in order
    2. each path in the loaded file's top-level ``defaults:`` key, in order
    3. the loaded file itself

    The ``defaults`` keys are stripped before validation. Relative paths in the
    in-file ``defaults:`` list resolve against the loaded file's directory.
    """
    main_path = Path(path)
    main_data = _read_yaml(main_path)

    in_file_defaults = main_data.pop("defaults", None) or []
    if isinstance(in_file_defaults, str):
        in_file_defaults = [in_file_defaults]

    merged: dict = {}
    for d in defaults or []:
        merged = _deep_merge(merged, _read_yaml(d))
    for d in in_file_defaults:
        d_path = Path(d)
        if not d_path.is_absolute():
            d_path = main_path.parent / d_path
        merged = _deep_merge(merged, _read_yaml(d_path))
    merged = _deep_merge(merged, main_data)

    try:
        return QcbConfig.parse_obj(merged)
    except Exception as e:
        raise ValueError(
            f"Failed to validate qcb config from '{main_path}': {e}"
        ) from e


def normalize_selectors(config: QcbConfig) -> dict:
    """Return ``{name: spec_string}`` for every selector in ``config``.

    The form most callers want: a flat mapping of selector names to the raw
    strings that :func:`quantum_engine.select.parse_constraints` understands.
    """
    return {name: sel.spec for name, sel in config.selectors.items()}


__all__ = [
    "StructureSpec",
    "SelectorSpec",
    "GeometrySpec",
    "ConstraintSpec",
    "CalculatorSpec",
    "OptOperation",
    "MDOperation",
    "MTDOperation",
    "UmbrellaOperation",
    "ScanOperation",
    "FreqOperation",
    "IRCOperation",
    "NEBOperation",
    "TSOperation",
    "QcbConfig",
    "load_config",
    "normalize_selectors",
]
