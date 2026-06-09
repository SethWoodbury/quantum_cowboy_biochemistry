"""Charge / spin ledger — explicit per-residue charge bookkeeping for cluster jobs.

Why this exists
---------------
Every cluster TS-search job has an implicit assumption about which residues are
deprotonated, which metals are +n, and what the total cluster charge & spin
multiplicity are. Those assumptions live (right now) in scattered argparse
flags or worse — in operator memory. When the assumption is wrong, the MACE
calculator silently picks an electronic state that may be 5-30 kcal/mol off,
and no one notices until the freq calc shows a phantom imag mode.

This module implements a **fail-loud** ledger. Every cluster job MUST be able
to articulate, per residue / cofactor / ion, what charge it carries. The sum
must agree with the user's claimed cluster total charge (and a spin count
must be supplied). The ledger is then attached to ``atoms.info`` and
propagated to every output PDB as ``REMARK QCB CHARGE_LEDGER`` lines so the
provenance lives with the structure, not the slurm log.

Reaction-agnostic
-----------------
- No PTE-specific names. The ledger is a free-form mapping
  ``{label: charge_int}`` — labels can be ``Zn1`` or ``COFACTOR_A`` or
  ``protein_backbone``, whatever the user assigns.
- Charges are integers (we don't audit fractional point charges in the
  ledger; if you need partial charges, that's a force-field detail).
- Spin is the spin multiplicity ``2S+1`` as an int (1 = singlet, 2 = doublet,
  3 = triplet, ...). We require it explicitly because a default of 1 is wrong
  for half of all metalloenzymes.

CLI surface
-----------
The recommended use is via ``--charge-ledger ledger.yaml`` plumbed through
the existing tools. See :func:`load_ledger`, :func:`validate_ledger`, and
:func:`inject_into_atoms`.

Example YAML::

    # ledger.yaml
    total: -1
    spin: 1            # singlet
    components:
      Zn1: 2
      Zn2: 2
      OHX: -1
      KCX: -1          # carbamylated lysine
      GLU254: -1
      substrate: -2
      His254: 0
      waters: 0
      protein_backbone: 0

    # Optional: per-component notes for human readers (ignored by the
    # validator; written through to PDB REMARK QCB lines)
    notes:
      OHX: "Bridging hydroxide; pKa ~7.5"
      substrate: "Paraoxon dianion (post-deprotonation)"
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

log = logging.getLogger("quantum_engine.ops.charge_ledger")


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class ChargeLedger:
    """Explicit per-component charge accounting for a cluster job.

    Attributes:
        total: Net cluster charge (int).
        spin: Spin multiplicity 2S+1 (int >= 1).
        components: ``{label: charge_int}`` mapping. Labels are user-defined.
        notes: Optional per-component free-form notes (str -> str).
        source: Where the ledger was loaded from (path or ``'inline'``).
    """
    total: int
    spin: int
    components: dict[str, int] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    source: str = "inline"

    def __post_init__(self) -> None:
        # Coerce types defensively (YAML scalars sometimes come back as str)
        self.total = int(self.total)
        self.spin = int(self.spin)
        if self.spin < 1:
            raise ValueError(
                f"ChargeLedger.spin must be >= 1 (multiplicity 2S+1), got {self.spin}"
            )
        coerced: dict[str, int] = {}
        for k, v in self.components.items():
            coerced[str(k)] = int(v)
        self.components = coerced
        self.notes = {str(k): str(v) for k, v in (self.notes or {}).items()}

    @property
    def component_sum(self) -> int:
        """Sum of all component charges (should equal :attr:`total`)."""
        return sum(self.components.values())

    @property
    def is_balanced(self) -> bool:
        """``True`` iff component_sum == total."""
        return self.component_sum == self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "spin": self.spin,   # NB: holds the MULTIPLICITY M=2S+1 (calculator-convention key name)
            "components": dict(self.components),
            "notes": dict(self.notes),
            "source": self.source,
        }

    def to_remark_lines(self, max_per_line: int = 4) -> list[str]:
        """Render the ledger as a list of ``REMARK QCB CHARGE_LEDGER ...`` lines.

        PDB REMARK lines are capped at 80 characters; we wrap component
        listings at ``max_per_line`` items and add a header line first.
        """
        lines: list[str] = [
            f"REMARK QCB CHARGE_LEDGER total={self.total:+d} spin={self.spin}",
            f"REMARK QCB CHARGE_LEDGER source={self.source}",
            f"REMARK QCB CHARGE_LEDGER component_sum={self.component_sum:+d} "
            f"balanced={self.is_balanced}",
        ]
        items = list(self.components.items())
        for i in range(0, len(items), max_per_line):
            chunk = items[i : i + max_per_line]
            kvs = " ".join(f"{k}={v:+d}" for k, v in chunk)
            lines.append(f"REMARK QCB CHARGE_LEDGER components {kvs}")
        for label, note in self.notes.items():
            lines.append(f"REMARK QCB CHARGE_LEDGER note {label}: {note}")
        return lines


# ----------------------------------------------------------------------------
# Load / save
# ----------------------------------------------------------------------------
def load_ledger(path: str | Path) -> ChargeLedger:
    """Parse a YAML or JSON charge-ledger file into a :class:`ChargeLedger`.

    Required keys: ``total``, ``spin``. Optional: ``components`` (dict),
    ``notes`` (dict).

    Raises:
        FileNotFoundError: missing path.
        ValueError: missing required keys, malformed structure.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"charge-ledger file not found: {p}")
    text = p.read_text()
    data: dict[str, Any]
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required for YAML ledgers. Install pyyaml or use "
                "JSON (.json)."
            ) from exc
        data = yaml.safe_load(text) or {}
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        # Try YAML first, fall back to JSON
        try:
            import yaml
            data = yaml.safe_load(text) or {}
        except Exception:
            data = json.loads(text)

    if not isinstance(data, Mapping):
        raise ValueError(f"ledger {p}: top-level must be a mapping, got {type(data).__name__}")

    if "total" not in data:
        raise ValueError(f"ledger {p}: missing required key 'total'")
    # `multiplicity` (M=2S+1) is the canonical key; accept the legacy `spin` key too.
    if "multiplicity" not in data and "spin" not in data:
        raise ValueError(f"ledger {p}: missing required key 'multiplicity'")

    return ChargeLedger(
        total=int(data["total"]),
        spin=int(data.get("multiplicity", data.get("spin"))),
        components=dict(data.get("components") or {}),
        notes=dict(data.get("notes") or {}),
        source=str(p),
    )


def write_ledger(ledger: ChargeLedger, path: str | Path) -> Path:
    """Serialize a :class:`ChargeLedger` to YAML (or JSON if extension is ``.json``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = ledger.to_dict()
    payload.pop("source", None)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(payload, indent=2))
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML required to write YAML ledgers") from exc
        p.write_text(yaml.safe_dump(payload, sort_keys=False))
    return p


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
@dataclass
class LedgerValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            joined = "; ".join(self.errors)
            raise ValueError(f"ChargeLedger validation failed: {joined}")


def validate_ledger(
    ledger: ChargeLedger,
    *,
    pdb_charge_hint: int | None = None,
    cli_charge: int | None = None,
    require_balanced: bool = True,
) -> LedgerValidationResult:
    """Cross-check a ledger against external charge sources and self-consistency.

    Args:
        ledger: the parsed ledger.
        pdb_charge_hint: charge inferred from a PDB ``REMARK TOTAL_CHARGE``
            line (or ``netCHG_*`` filename pattern). If ``None``, skip.
        cli_charge: ``--charge`` value passed on the CLI (the operator's
            claim). If ``None``, skip.
        require_balanced: if ``True``, mismatched component_sum vs total is
            an ERROR (default — fail loud). If ``False``, demoted to a
            warning.

    Returns:
        A :class:`LedgerValidationResult`. ``ok`` is True only when every
        applicable check passed.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not ledger.is_balanced:
        msg = (
            f"component sum {ledger.component_sum:+d} does not match "
            f"declared total {ledger.total:+d} "
            f"(diff={ledger.component_sum - ledger.total:+d})"
        )
        if require_balanced:
            errors.append(msg)
        else:
            warnings.append(msg)

    if pdb_charge_hint is not None and pdb_charge_hint != ledger.total:
        warnings.append(
            f"PDB REMARK charge hint = {pdb_charge_hint:+d} disagrees with "
            f"ledger total = {ledger.total:+d}; ledger wins (operator-supplied)"
        )

    if cli_charge is not None and cli_charge != ledger.total:
        errors.append(
            f"--charge {cli_charge:+d} disagrees with ledger.total {ledger.total:+d}; "
            "fix one or the other before running"
        )

    if ledger.spin < 1:
        errors.append(f"spin multiplicity must be >= 1, got {ledger.spin}")

    # Sanity range (catch obvious goofs like spin=10 on an organic molecule)
    if ledger.spin > 11:
        warnings.append(
            f"spin multiplicity {ledger.spin} is unusually high — verify this is intentional"
        )

    return LedgerValidationResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
    )


# ----------------------------------------------------------------------------
# Inject + propagate
# ----------------------------------------------------------------------------
def inject_into_atoms(atoms, ledger: ChargeLedger) -> None:
    """Set ``atoms.info['charge']`` and ``atoms.info['spin']`` from the ledger.

    Also stores the full ledger dict under ``atoms.info['charge_ledger']`` so
    downstream code (e.g. PDB writers) can recover the complete provenance.
    """
    atoms.info["charge"] = ledger.total
    atoms.info["spin"] = ledger.spin
    atoms.info["charge_ledger"] = ledger.to_dict()


def append_remarks_to_pdb(pdb_path: str | Path, ledger: ChargeLedger) -> None:
    """Append (or refresh) ``REMARK QCB CHARGE_LEDGER`` lines on an existing PDB.

    If the file already contains CHARGE_LEDGER REMARKs, they are stripped and
    replaced (idempotent). Other REMARK lines are preserved.

    The lines are placed at the start of the file's REMARK block (right after
    any existing REMARK lines, before HETATM/ATOM records).
    """
    p = Path(pdb_path)
    if not p.is_file():
        raise FileNotFoundError(p)
    text = p.read_text()

    # Strip existing CHARGE_LEDGER REMARKs
    pattern = re.compile(r"^REMARK\s+QCB\s+CHARGE_LEDGER.*$\n?", re.MULTILINE)
    cleaned = pattern.sub("", text)

    new_lines = "\n".join(ledger.to_remark_lines()) + "\n"

    # Insert before the first ATOM/HETATM line so REMARKs cluster at the top
    match = re.search(r"^(?:ATOM|HETATM|MODEL|END)\b", cleaned, re.MULTILINE)
    if match:
        insert_at = match.start()
        out = cleaned[:insert_at] + new_lines + cleaned[insert_at:]
    else:
        # No structure records; append at end
        out = cleaned.rstrip("\n") + "\n" + new_lines

    p.write_text(out)


# ----------------------------------------------------------------------------
# CLI helper used by other tools that accept --charge-ledger
# ----------------------------------------------------------------------------
def resolve_charge_and_spin(
    *,
    ledger_path: str | Path | None,
    cli_charge: int | None,
    cli_spin: int | None,
    pdb_charge_hint: int | None,
    require_balanced: bool = True,
) -> tuple[int, int, ChargeLedger | None]:
    """Combine the various charge / spin sources into a single (charge, spin) pair.

    Resolution order:
        1. If a ledger is provided, validate it against ``cli_charge`` and
           ``pdb_charge_hint`` and use ``ledger.total`` / ``ledger.spin``.
        2. Else fall back to ``cli_charge`` / ``cli_spin`` / ``pdb_charge_hint``.

    Returns ``(charge, spin, ledger_or_None)``.

    Raises ValueError on a fail-loud validation issue.
    """
    if ledger_path is not None:
        ledger = load_ledger(ledger_path)
        result = validate_ledger(
            ledger,
            pdb_charge_hint=pdb_charge_hint,
            cli_charge=cli_charge,
            require_balanced=require_balanced,
        )
        for w in result.warnings:
            log.warning("charge-ledger: %s", w)
        result.raise_if_failed()
        spin = cli_spin if cli_spin is not None else ledger.spin
        if spin != ledger.spin:
            log.warning(
                "charge-ledger: --spin %s overrides ledger.spin %s",
                spin, ledger.spin,
            )
        return ledger.total, spin, ledger

    # No ledger: fall back to CLI / PDB hint
    charge = cli_charge if cli_charge is not None else (pdb_charge_hint or 0)
    spin = cli_spin if cli_spin is not None else 1
    if cli_charge is None and pdb_charge_hint is None:
        log.warning(
            "no --charge-ledger, no --charge, no PDB REMARK charge hint — "
            "falling back to charge=0 spin=1. Pass --charge-ledger to be explicit."
        )
    return int(charge), int(spin), None


# ----------------------------------------------------------------------------
# Auto-build a skeleton ledger from a biotite AtomArray (best-effort)
# ----------------------------------------------------------------------------
def auto_skeleton_ledger(
    bt_struct,
    *,
    total_charge: int,
    spin: int = 1,
    metal_oxidation: Mapping[str, int] | None = None,
) -> ChargeLedger:
    """Build a starter ledger from a parsed PDB.

    The skeleton lists every distinct residue / metal observed in the
    structure with charge=0 by default; the operator is then expected to
    fill in known nonzero entries (deprotonated GLU/ASP/etc) before the run.
    Useful for bootstrapping a ledger.yaml file.

    Args:
        bt_struct: biotite AtomArray (e.g. from quantum_engine.io.load_structure).
        total_charge: declared cluster charge.
        spin: declared spin multiplicity (default 1 = singlet).
        metal_oxidation: ``{metal_resname: ox_state}`` (e.g. ``{"ZN": 2}``).

    Returns:
        A :class:`ChargeLedger` with components zero-initialised; not balanced
        if the user supplied a nonzero total. Caller is expected to edit before
        validating.
    """
    metal_ox = {k.upper(): int(v) for k, v in (metal_oxidation or {}).items()}
    components: dict[str, int] = {}

    # Iterate residues in order; one entry per (chain, res_id, res_name).
    seen: set[tuple[str, int, str]] = set()
    for i in range(len(bt_struct)):
        chain = str(bt_struct.chain_id[i])
        res_id = int(bt_struct.res_id[i])
        res_name = str(bt_struct.res_name[i])
        key = (chain, res_id, res_name)
        if key in seen:
            continue
        seen.add(key)
        label = f"{res_name}{res_id}{chain}"
        if res_name.upper() in metal_ox:
            components[label] = metal_ox[res_name.upper()]
        else:
            components[label] = 0
    return ChargeLedger(
        total=total_charge,
        spin=spin,
        components=components,
        notes={
            "_auto": "skeleton generated by auto_skeleton_ledger; edit before use"
        },
        source="auto_skeleton",
    )
