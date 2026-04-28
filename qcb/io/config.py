"""
Pydantic v2-based YAML config schema for qcb.

Replaces the verbose, string-heavy enz-ts YAML format with named entities
that reference each other by name. Supports file inheritance (defaults +
overrides) via a top-level ``defaults`` key or the ``defaults`` argument to
:func:`load_config`.

Schema overview
---------------
A qcb config has these top-level fields:

* ``qcb_version``  — schema version (currently 1)
* ``name`` / ``description`` — optional metadata
* ``structure`` — input structure (PDB/XYZ/CIF), charge, and ligand names
* ``selectors`` — named atom selectors using the grammar in
  :mod:`qcb.io.constraints` (``residue YYL``, ``atoms P1``, ``chain B``,
  ``resid 100``, ``range 0 99``, ``element C``, ``all``, ``none``)
* ``geometry`` — named geometric quantities (bonds, angles, dihedrals, or
  bond-difference CVs); their ``atoms`` field references selector names
* ``constraints`` — list of ASE constraints applied during MD/opt
* ``calculator`` — MLFF calculator config (model alias, head, device, dtype)
* ``operation`` — discriminated union; the ``kind`` field picks the subclass
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
    description: NEB-TS for phosphoryl transfer in the YYL active site

    structure:
      path: /home/woodbuse/runs/r2/active_site.pdb
      charge: -2
      ligand_names: [YYL]

    selectors:
      P1: "residue YYL atoms P1"          # raw-string shorthand
      O3:                                  # full SelectorSpec form
        spec: "residue YYL atoms O3"
      O_nuc: "residue ASP atoms OD2 resid 165"
      backbone_heavy: "atoms CA C N O"

    geometry:
      d_PO_bond:
        kind: distance
        atoms: [P1, O3]
      d_PO_form:
        kind: distance
        atoms: [P1, O_nuc]
      cv_diff:
        kind: distance_diff
        atoms: [P1, O3, O_nuc]            # d(P1-O3) - d(P1-O_nuc)
        log: true

    constraints:
      - kind: fix_atoms
        selector: backbone_heavy
      - kind: harmonic_restraint
        geom: d_PO_bond
        k: 5.0
        r0: 1.65

    calculator:
      model: mace-omol
      device: cuda
      dtype: float64

    operation:
      kind: ts
      strategy: cv-spring
      n_images: 15
      interpolation: geodesic
      cv_atoms: [P1, O_nuc, O3]            # [P, nuc, LG]
      cv_s_reactant: -2.0
      cv_s_product:  2.5

Inheritance
-----------
A config may declare ``defaults: [path1.yaml, path2.yaml]`` at the top, or
the caller may pass ``defaults=[...]`` to :func:`load_config`. Defaults are
merged in order (later wins), with the main file applied last. Mappings are
merged key-by-key; lists are replaced wholesale.

Programmatic usage
------------------
.. code-block:: python

    from qcb.io.config import load_config, normalize_selectors

    cfg = load_config("override.yaml", defaults=["defaults.yaml"])
    selectors: dict[str, str] = normalize_selectors(cfg)
    # selectors["P1"] == "residue YYL atoms P1"

    if cfg.operation.kind == "ts":
        print(cfg.operation.strategy)        # "cv-spring"
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Selector grammar — must stay in sync with qcb/io/constraints.py
# ---------------------------------------------------------------------------

# Top-level tokens recognised by qcb.io.constraints.parse_constraints.
_VALID_SELECTOR_TOKENS: frozenset[str] = frozenset(
    {"residue", "resid", "chain", "atoms", "element", "range", "all", "none"}
)


def _validate_selector_string(spec: str) -> str:
    """Cheap structural check for selector strings (no biotite handle).

    We only verify that the leading token is recognised by the grammar in
    :mod:`qcb.io.constraints`. Full semantic validation happens at runtime
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

    Extras are IGNORED (silently dropped) rather than forbidden. This:
    - Tolerates unknown keys from legacy-config translation
    - Allows forward-compat: newer configs with extra fields load on older code
    - Keeps validate_assignment=True so explicit setattr still type-checks
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)


class StructureSpec(_StrictModel):
    """Input structure: file path plus chemical metadata."""

    path: str = Field(..., description="Path to PDB / XYZ / CIF file")
    charge: int = Field(0, description="Net charge of the system (electrons)")
    ligand_names: list[str] = Field(
        default_factory=list,
        description="Residue names of any non-standard ligands (e.g. ['YYL']).",
    )


class SelectorSpec(_StrictModel):
    """Named atom selector; ``spec`` follows :mod:`qcb.io.constraints` grammar."""

    spec: str = Field(..., description="Selector spec string, e.g. 'residue YYL atoms P1'")

    @field_validator("spec")
    @classmethod
    def _check_spec(cls, v: str) -> str:
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

    @model_validator(mode="after")
    def _check_arity(self) -> "GeometrySpec":
        expected = {"distance": 2, "angle": 3, "dihedral": 4, "distance_diff": 3}[self.kind]
        if len(self.atoms) != expected:
            raise ValueError(
                f"GeometrySpec(kind='{self.kind}') needs {expected} selector names, "
                f"got {len(self.atoms)}: {self.atoms}"
            )
        return self


class ConstraintSpec(_StrictModel):
    """An ASE constraint to apply during MD/optimisation.

    Different ``kind`` values use different fields:

    * ``fix_atoms``           — uses ``selector``
    * ``harmonic_restraint``  — uses ``geom``, ``k``, ``r0``, optional ``mode``
    * ``harmonic_walls``      — uses ``geom``, ``k``, ``r0``, ``mode``
    * ``bond_difference_cv``  — uses ``atoms`` (3 selector names: P, nuc, LG),
                                ``k``, ``r0``
    """

    kind: Literal["fix_atoms", "harmonic_restraint", "harmonic_walls",
                  "harmonic_init", "bond_difference_cv"]
    name: str | None = Field(None, description="Optional human-readable label")
    selector: str | None = Field(
        None,
        description="Selector name (for fix_atoms / harmonic_init)",
    )
    geom: str | None = Field(None, description="Geometry name (for harmonic_*)")
    snapshot: bool = Field(
        False,
        description="harmonic_init: anchor each atom to its initial position "
                    "(equivalent to ASE Hookean restraint per atom)",
    )
    atoms: list[str] | None = Field(
        None,
        description="Selector names (e.g. [P, nuc, LG] for bond_difference_cv)",
    )
    k: float = Field(0.0, description="Spring constant (eV/Å² or eV/rad²)")
    r0: float = Field(0.0, description="Equilibrium value or wall centre")
    mode: Literal["attractive", "repulsive", "both"] = "both"
    fmax: float | None = Field(None, description="Optional force cap on this constraint")

    @model_validator(mode="after")
    def _check_kind_fields(self) -> "ConstraintSpec":
        if self.kind == "fix_atoms":
            if not self.selector:
                raise ValueError("ConstraintSpec(kind='fix_atoms') requires 'selector'")
        elif self.kind == "harmonic_init":
            # Per-atom anchor at initial positions; needs which atoms
            if not self.selector:
                raise ValueError(
                    "ConstraintSpec(kind='harmonic_init') requires 'selector' "
                    "(which atoms to anchor at initial positions)"
                )
        elif self.kind in {"harmonic_restraint", "harmonic_walls"}:
            if not self.geom:
                raise ValueError(
                    f"ConstraintSpec(kind='{self.kind}') requires 'geom' "
                    "(name of a GeometrySpec)"
                )
        elif self.kind == "bond_difference_cv":
            if not self.atoms or len(self.atoms) != 3:
                raise ValueError(
                    "ConstraintSpec(kind='bond_difference_cv') requires "
                    "atoms=[P, nuc, LG] (3 selector names)"
                )
        return self


class CalculatorSpec(_StrictModel):
    """MLFF calculator settings (consumed by :mod:`qcb.calc` factories)."""

    model: str = Field("mace-omol", description="Model alias from qcb.calc factory")
    head: str | None = Field(None, description="Head name for multi-head models")
    device: Literal["cuda", "cpu"] = "cuda"
    dtype: Literal["float32", "float64"] = "float64"


# ---------------------------------------------------------------------------
# Operation discriminated union
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
    k: float = Field(..., description="Restraint k (kJ/mol/Å²)")
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
    indices_selector: str | None = Field(
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
    cv_atoms: list[str] | None = Field(
        None,
        description="Selector names [P, nuc, LG] for cv-spring/mtd strategies",
    )
    cv_s_reactant: float = -2.0
    cv_s_product: float = 2.5


_OperationUnion = Annotated[
    Union[
        OptOperation,
        MDOperation,
        MTDOperation,
        UmbrellaOperation,
        ScanOperation,
        FreqOperation,
        IRCOperation,
        NEBOperation,
        TSOperation,
    ],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class QcbConfig(_StrictModel):
    """Top-level qcb config. Validates internal cross-references."""

    qcb_version: int = 1
    name: str | None = None
    description: str | None = None

    # `structure` and `operation` are required for FULL configs but allowed to
    # be None on defaults-only files (intended for inheritance via `defaults:`).
    # If a config is loaded as the top-level user file (no defaults) and lacks
    # these, downstream code should error.
    structure: StructureSpec | None = None
    selectors: dict[str, SelectorSpec] = Field(default_factory=dict)
    geometry: dict[str, GeometrySpec] = Field(default_factory=dict)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    calculator: CalculatorSpec = Field(default_factory=CalculatorSpec)
    operation: _OperationUnion | None = None

    @field_validator("selectors", mode="before")
    @classmethod
    def _normalize_selectors(cls, v: Any) -> Any:
        """Allow shorthand: ``selectors: {P1: "residue YYL atoms P1"}``."""
        if not isinstance(v, dict):
            return v
        out: dict[str, Any] = {}
        for name, val in v.items():
            if isinstance(val, str):
                out[name] = {"spec": val}
            else:
                out[name] = val
        return out

    @model_validator(mode="after")
    def _validate_references(self) -> "QcbConfig":
        sel_names = sorted(self.selectors.keys())
        geom_names = sorted(self.geometry.keys())

        def _check_selector(ref: str, where: str) -> None:
            if ref not in self.selectors:
                raise ValueError(
                    f"{where} references undefined selector '{ref}'. "
                    f"Defined selectors: {sel_names or '(none)'}"
                )

        def _check_geometry(ref: str, where: str) -> None:
            if ref not in self.geometry:
                raise ValueError(
                    f"{where} references undefined geometry '{ref}'. "
                    f"Defined geometries: {geom_names or '(none)'}"
                )

        # Geometry → selectors
        for gname, geom in self.geometry.items():
            for atom_ref in geom.atoms:
                _check_selector(atom_ref, f"Geometry '{gname}' (atom)")

        # Constraints → selectors / geometry
        for i, c in enumerate(self.constraints):
            tag = f"Constraint #{i} (kind='{c.kind}')"
            if c.kind == "fix_atoms":
                _check_selector(c.selector, tag)  # type: ignore[arg-type]
            elif c.kind in {"harmonic_restraint", "harmonic_walls"}:
                _check_geometry(c.geom, tag)  # type: ignore[arg-type]
            elif c.kind == "bond_difference_cv":
                for a in c.atoms or []:
                    _check_selector(a, tag)

        # Operation → selectors / geometry (skipped if defaults-only config)
        if self.operation is not None:
            op = self.operation
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
                for a in op.cv_atoms or []:
                    _check_selector(a, op_tag)

        return self


# ---------------------------------------------------------------------------
# YAML loading + inheritance
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (later wins).

    Mappings are merged key-by-key; everything else (including lists) is
    replaced wholesale. Returns a new dict; inputs are not mutated.
    """
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: str | Path) -> dict[str, Any]:
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


def load_config(
    path: str | Path,
    defaults: list[str | Path] | None = None,
) -> QcbConfig:
    """Load a YAML config, optionally inheriting from one or more defaults files.

    Resolution order (later wins):

    1. each path in the ``defaults`` argument, in order
    2. each path in the loaded file's top-level ``defaults:`` key, in order
    3. the loaded file itself

    The ``defaults`` keys are stripped before validation. Relative paths in
    the in-file ``defaults:`` list are resolved against the loaded file's
    directory.
    """
    main_path = Path(path)
    main_data = _read_yaml(main_path)

    # Pull in-file defaults (relative paths resolve against main_path's dir).
    in_file_defaults = main_data.pop("defaults", None) or []
    if isinstance(in_file_defaults, str):
        in_file_defaults = [in_file_defaults]

    merged: dict[str, Any] = {}
    for d in defaults or []:
        merged = _deep_merge(merged, _read_yaml(d))
    for d in in_file_defaults:
        d_path = Path(d)
        if not d_path.is_absolute():
            d_path = main_path.parent / d_path
        merged = _deep_merge(merged, _read_yaml(d_path))
    merged = _deep_merge(merged, main_data)

    try:
        return QcbConfig.model_validate(merged)
    except Exception as e:
        raise ValueError(
            f"Failed to validate qcb config from '{main_path}': {e}"
        ) from e


def normalize_selectors(config: QcbConfig) -> dict[str, str]:
    """Return ``{name: spec_string}`` for every selector in ``config``.

    This is the form most callers want: a flat mapping of selector names to
    the raw strings that :func:`qcb.io.constraints.parse_constraints`
    understands.
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
