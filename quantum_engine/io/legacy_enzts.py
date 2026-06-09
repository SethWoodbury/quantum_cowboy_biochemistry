"""Translator for legacy enz-ts YAML configs to the modern cowboy-qc config format.

Reads YAMLs in the enz-ts schema (sim_params / biotite_annot / ase_const / plumed)
and emits dictionaries matching the qcb/io/config.py schema (qcb_version /
structure / selectors / geometry / constraints / calculator / operation).

This module is deliberately defensive: legacy configs are heterogeneous and
some expressions can't be cleanly mapped to the cowboy-qc selector grammar. Anything
that we can't translate is preserved verbatim in a "# TODO: translate" comment
attached to the relevant key (and surfaced in the summary table).

Usage:

    from quantum_engine.io.legacy_enzts import translate_legacy_yaml, translate_directory
    cfg = translate_legacy_yaml("path/to/wBB_wKCX.yaml")
    paths = translate_directory("in_dir", "out_dir")

Or run as a script with no args to translate the 8 seth_pte configs in place:

    python -m quantum_engine.io.legacy_enzts
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Biotite expression -> cowboy-qc selector grammar
# ---------------------------------------------------------------------------

# Patterns we recognise. The cowboy-qc selector grammar (per the spec) uses tokens
# like "residue YYL", "atoms P1", with an implicit AND between them, and
# "--free" for set-difference. Anything else is preserved with a TODO comment.

_RES_RE = re.compile(r"\bres_([A-Za-z0-9]+)\b")
_ATOM_RE = re.compile(r"\batom_([A-Za-z0-9']+)\b")


def _translate_atom_disjunction(expr: str) -> list[str] | None:
    """If expr is "atom_X | atom_Y | atom_Z", return ['X','Y','Z']; else None."""
    parts = [p.strip() for p in expr.split("|")]
    atoms = []
    for p in parts:
        m = _ATOM_RE.fullmatch(p)
        if not m:
            return None
        atoms.append(m.group(1))
    return atoms


def translate_biotite_expr(expr: str) -> tuple[str, str | None]:
    """Translate a biotite annotation expression to a cowboy-qc selector string.

    Returns (translated_expr, todo_comment_or_None). If the expression couldn't
    be cleanly translated, the original is returned and a TODO comment is set.
    """
    e = expr.strip()

    # Simple disjunction of atoms: "atom_CA | atom_N | atom_C" -> "atoms CA N C"
    atom_list = _translate_atom_disjunction(e)
    if atom_list is not None:
        return f"atoms {' '.join(atom_list)}", None

    # Pure residue token: "res_YYL" -> "residue YYL"
    m = _RES_RE.fullmatch(e)
    if m:
        return f"residue {m.group(1)}", None

    # Conjunction: "res_YYL & atom_P1" -> "residue YYL atoms P1"
    if "&" in e and "|" not in e and "-" not in e:
        parts = [p.strip() for p in e.split("&")]
        residue = None
        atoms: list[str] = []
        ok = True
        for p in parts:
            mr = _RES_RE.fullmatch(p)
            ma = _ATOM_RE.fullmatch(p)
            if mr:
                residue = mr.group(1)
            elif ma:
                atoms.append(ma.group(1))
            else:
                ok = False
                break
        if ok and (residue or atoms):
            tokens = []
            if residue:
                tokens.append(f"residue {residue}")
            if atoms:
                tokens.append(f"atoms {' '.join(atoms)}")
            return " ".join(tokens), None

    # Set-difference: "backbone_atoms - ligand_atoms" -> "backbone_atoms --free ligand_atoms"
    # We pass through the operand names so the user / loader can resolve them
    # against other selectors.
    if "-" in e and "&" not in e and "|" not in e:
        lhs, _, rhs = e.partition("-")
        lhs, rhs = lhs.strip(), rhs.strip()
        # Recurse on each side in case they're translatable atomic exprs.
        lhs_t, lhs_todo = translate_biotite_expr(lhs)
        rhs_t, rhs_todo = translate_biotite_expr(rhs)
        todo = None
        if lhs_todo or rhs_todo:
            todo = f"original: {expr!r}"
        return f"{lhs_t} --free {rhs_t}", todo

    # Couldn't translate -> preserve original with a TODO.
    return e, f"could not translate biotite expr: {expr!r}"


# ---------------------------------------------------------------------------
# Two-atom expression for ase constraints, e.g. "P1 | O1" -> [P1, O1]
# ---------------------------------------------------------------------------

def _parse_geom_ref(expr: str) -> str | None:
    """A bare reference to another biotite key (no operators). Returns the key."""
    e = expr.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", e):
        return e
    return None


def _parse_two_atom_or(expr: str) -> list[str] | None:
    """Pattern "A | B" where A,B are bare names -> [A, B]."""
    if "|" not in expr:
        return None
    parts = [p.strip() for p in expr.split("|")]
    if len(parts) != 2:
        return None
    if all(_parse_geom_ref(p) for p in parts):
        return parts
    return None


# ---------------------------------------------------------------------------
# Section translators
# ---------------------------------------------------------------------------

def _translate_structure(sim_params: dict, name_hint: str) -> tuple[dict, list[str]]:
    """Translate sim_params -> structure block. Returns (block, warnings)."""
    warnings: list[str] = []
    structure: dict[str, Any] = {}

    if "input_structure" in sim_params:
        structure["path"] = sim_params["input_structure"]
    if "net_charge" in sim_params:
        structure["charge"] = sim_params["net_charge"]

    ligand_names = sim_params.get("ligand_names")
    if ligand_names:
        cleaned = []
        for raw in ligand_names:
            m = _RES_RE.fullmatch(str(raw).strip())
            if m:
                cleaned.append(m.group(1))
            else:
                cleaned.append(str(raw))
                warnings.append(f"ligand_name {raw!r} did not match res_XXX")
        structure["ligand_names"] = cleaned

    return structure, warnings


def _translate_biotite(annot: dict) -> tuple[dict, dict, list[str]]:
    """Split biotite_annot into (selectors, geometry, warnings).

    Position entries become selectors; distance/angle/dihedral entries become
    geometry entries. Anything else is dropped with a warning.
    """
    selectors: dict[str, Any] = {}
    geometry: dict[str, Any] = {}
    warnings: list[str] = []

    for name, spec in annot.items():
        if not isinstance(spec, dict):
            warnings.append(f"biotite_annot[{name}] is not a dict, skipped")
            continue
        kind = spec.get("type")
        expr = spec.get("expr", "")
        log = spec.get("log", False)

        if kind == "position":
            tr, todo = translate_biotite_expr(str(expr))
            selectors[name] = tr
            if todo:
                warnings.append(f"selector[{name}]: {todo}")
        elif kind in ("distance", "angle", "dihedral"):
            atoms = _parse_two_atom_or(str(expr))
            entry: dict[str, Any] = {"kind": kind}
            if atoms is not None:
                entry["atoms"] = atoms
            else:
                ref = _parse_geom_ref(str(expr))
                if ref is not None:
                    # Single reference - rare, but pass through.
                    entry["atoms"] = [ref]
                else:
                    warnings.append(
                        f"geometry[{name}]: could not parse expr {expr!r}"
                    )
                    entry["expr"] = str(expr)
            if log:
                entry["log"] = True
            geometry[name] = entry
        else:
            warnings.append(
                f"biotite_annot[{name}] has unknown type {kind!r}, skipped"
            )

    return selectors, geometry, warnings


def _translate_ase_const(
    ase_const: dict, geometry: dict, selectors: dict
) -> tuple[list[dict], list[str]]:
    """Translate ase_const -> list of constraint entries."""
    out: list[dict] = []
    warnings: list[str] = []

    for name, spec in ase_const.items():
        if not isinstance(spec, dict):
            warnings.append(f"ase_const[{name}] is not a dict, skipped")
            continue
        ctype = spec.get("type")
        expr = str(spec.get("expr", ""))
        entry: dict[str, Any] = {"name": name, "kind": "harmonic_restraint"}

        if ctype == "spring_2atoms":
            # expr is usually a reference to a geometry key like "P1_O1".
            ref = _parse_geom_ref(expr)
            if ref is not None and ref in geometry:
                entry["geom"] = ref
            else:
                # Try to parse "A | B" inline.
                atoms = _parse_two_atom_or(expr)
                if atoms is not None:
                    entry["atoms"] = atoms
                else:
                    entry["expr"] = expr
                    warnings.append(
                        f"constraint[{name}]: spring_2atoms expr {expr!r} "
                        f"did not resolve to a geometry key"
                    )
        elif ctype == "spring_init":
            # Per-atom anchor at initial positions → new schema's harmonic_init kind
            entry["kind"] = "harmonic_init"
            entry["snapshot"] = True
            tr, todo = translate_biotite_expr(expr)
            entry["selector"] = tr
            if todo:
                warnings.append(f"constraint[{name}]: {todo}")
        else:
            warnings.append(
                f"ase_const[{name}] has unknown type {ctype!r}, "
                f"emitted as-is with kind=harmonic_restraint"
            )
            entry["raw_type"] = ctype
            if expr:
                entry["expr"] = expr

        for k in ("mode", "k", "r0", "fmax"):
            if k in spec:
                entry[k] = spec[k]

        out.append(entry)

    return out, warnings


def _translate_plumed(plumed: dict, geometry: dict) -> tuple[dict | None, list[str]]:
    """Translate plumed.cvs -> operation block (single mtd CV)."""
    warnings: list[str] = []
    cvs = plumed.get("cvs") or {}
    if not cvs:
        return None, warnings

    # Pick the first CV; warn if there are more.
    cv_names = list(cvs.keys())
    if len(cv_names) > 1:
        warnings.append(
            f"plumed.cvs has {len(cv_names)} CVs ({cv_names}); "
            f"only first ({cv_names[0]!r}) was placed in operation.cv. "
            f"Others preserved under operation.extra_cvs"
        )

    primary = cv_names[0]
    primary_spec = cvs[primary]

    operation: dict[str, Any] = {
        "kind": "mtd",
        "variant": "wt",
        "backend": "plumed",
    }

    # Try to map the CV reference to a geometry key.
    if primary in geometry:
        operation["cv"] = primary
    else:
        # Heuristic: if the CV is a COORDINATION over a single GROUPA/GROUPB
        # pair whose names match a geometry key, link it.
        ga = primary_spec.get("GROUPA")
        gb = primary_spec.get("GROUPB")
        if (
            isinstance(ga, list) and len(ga) == 1
            and isinstance(gb, list) and len(gb) == 1
        ):
            cand = f"{ga[0]}_{gb[0]}"
            if cand in geometry:
                operation["cv"] = cand
            else:
                operation["cv"] = primary
                warnings.append(
                    f"plumed CV {primary!r} could not be linked to a geometry "
                    f"entry; emitted as raw cv name"
                )
        else:
            operation["cv"] = primary

    # Preserve raw plumed CV spec so nothing is lost.
    operation["plumed_cv_spec"] = {primary: primary_spec}
    if len(cv_names) > 1:
        operation["extra_cvs"] = {n: cvs[n] for n in cv_names[1:]}

    return operation, warnings


# ---------------------------------------------------------------------------
# Top-level translate
# ---------------------------------------------------------------------------

def translate_legacy_yaml(path: str | Path) -> dict:
    """Read a legacy enz-ts YAML config, return a dict matching the new cowboy-qc schema.

    Warnings are recorded under the "_translation_warnings" key (list of strings)
    and as YAML comments where possible. Unrecognised fields are preserved
    under "_legacy_extras" so no information is silently lost.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"legacy config not found: {path}")

    with path.open() as f:
        legacy = yaml.safe_load(f) or {}
    if not isinstance(legacy, dict):
        raise ValueError(f"{path}: top-level YAML is not a mapping")

    warnings: list[str] = []
    out: dict[str, Any] = {"qcb_version": 1, "name": path.stem}

    sim_params = legacy.get("sim_params") or {}
    if sim_params:
        structure, w = _translate_structure(sim_params, path.stem)
        warnings.extend(w)
        if structure:
            out["structure"] = structure

    biotite = legacy.get("biotite_annot") or {}
    if biotite:
        sels, geom, w = _translate_biotite(biotite)
        warnings.extend(w)
        if sels:
            out["selectors"] = sels
        if geom:
            out["geometry"] = geom

    ase_const = legacy.get("ase_const") or {}
    if ase_const:
        cons, w = _translate_ase_const(
            ase_const, out.get("geometry", {}), out.get("selectors", {})
        )
        warnings.extend(w)
        if cons:
            out["constraints"] = cons

    # Default calculator: mace-polar (matches Baker-Lab QCB convention).
    out["calculator"] = {"model": "mace-polar"}

    plumed = legacy.get("plumed") or {}
    if plumed:
        op, w = _translate_plumed(plumed, out.get("geometry", {}))
        warnings.extend(w)
        if op is not None:
            out["operation"] = op

    # Preserve any sim_params fields we didn't pull into structure.
    sim_extras = {
        k: v
        for k, v in sim_params.items()
        if k not in {"input_structure", "net_charge", "ligand_names"}
    }
    legacy_extras: dict[str, Any] = {}
    if sim_extras:
        legacy_extras["sim_params"] = sim_extras
    # Preserve unrecognised top-level keys.
    for k, v in legacy.items():
        if k not in {"sim_params", "biotite_annot", "ase_const", "plumed"}:
            legacy_extras[k] = v
    if legacy_extras:
        out["_legacy_extras"] = legacy_extras

    if warnings:
        out["_translation_warnings"] = warnings

    return out


# ---------------------------------------------------------------------------
# YAML emission with header comments
# ---------------------------------------------------------------------------

def _yaml_dump(data: dict, header_lines: list[str] | None = None) -> str:
    """Dump a dict as YAML with an optional comment header."""
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, indent=2)
    if header_lines:
        header = "\n".join(f"# {ln}" for ln in header_lines) + "\n"
        return header + body
    return body


def translate_directory(in_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Translate every *.yaml in in_dir; write *.quantum_engine.yaml under out_dir.

    Returns the list of output paths (in input-sorted order).
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    if not in_dir.is_dir():
        raise NotADirectoryError(f"input dir not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    rows: list[tuple[str, str, int]] = []  # (input, output, n_warnings)

    for src in sorted(in_dir.glob("*.yaml")):
        try:
            cfg = translate_legacy_yaml(src)
        except Exception as exc:
            print(f"  ! {src.name}: FAILED ({exc})", file=sys.stderr)
            continue
        warnings = cfg.get("_translation_warnings", [])
        header = [
            f"cowboy-qc config translated from legacy enz-ts: {src}",
            "Generated by quantum_engine.io.legacy_enzts.translate_directory",
        ]
        if warnings:
            header.append(f"{len(warnings)} TODO(s) — see _translation_warnings")
        out_name = src.stem + ".quantum_engine.yaml"
        out_path = out_dir / out_name
        out_path.write_text(_yaml_dump(cfg, header))
        written.append(out_path)
        rows.append((str(src), str(out_path), len(warnings)))

    # Print summary table.
    if rows:
        in_w = max(len(r[0]) for r in rows)
        out_w = max(len(r[1]) for r in rows)
        print()
        print(f"  {'INPUT'.ljust(in_w)}  ->  {'OUTPUT'.ljust(out_w)}  TODO")
        print(f"  {'-' * in_w}  --  {'-' * out_w}  ----")
        for src_s, out_s, n in rows:
            flag = f"{n:>4}" if n == 0 else f"{n:>4}  <-- review"
            print(f"  {src_s.ljust(in_w)}  ->  {out_s.ljust(out_w)}  {flag}")
        print()

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_IN = "/net/scratch/woodbuse/metad/config/seth_pte"
_DEFAULT_OUT = (
    "/home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/qcb/configs/seth_pte"
)


def _main(argv: list[str]) -> int:
    if len(argv) == 1:
        in_dir, out_dir = _DEFAULT_IN, _DEFAULT_OUT
    elif len(argv) == 3:
        in_dir, out_dir = argv[1], argv[2]
    else:
        print(
            "usage: python -m quantum_engine.io.legacy_enzts [IN_DIR OUT_DIR]\n"
            "  with no args, translates the seth_pte configs.",
            file=sys.stderr,
        )
        return 2

    print(f"translating: {in_dir}")
    print(f"        ->: {out_dir}")
    written = translate_directory(in_dir, out_dir)
    print(f"wrote {len(written)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
