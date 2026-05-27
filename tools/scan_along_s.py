#!/usr/bin/env python
"""scan_along_s.py — generalized relaxed-scan TS-finder along a 1-D collective
variable defined by an arbitrary set of bond distances.

Default behavior (no extra flags) reproduces the legacy SN2-at-P scan:
``s = d(P-Olg) − d(P-Onuc)`` swept on the line ``d(P-Olg) + d(P-Onuc) = sum_target``,
with both bonds pinned via FixBondLengths and FixInternals over CA-CA pairs.
That mode (``--drag-mode pin-both``, the default) is bit-for-bit equivalent to
the previous version.

New (opt-in) modes generalize this to any reaction:

  --drag-mode pin-onuc          # pin only d(P-Onuc), let d(P-Olg) relax
  --drag-mode pin-olg           # pin only d(P-Olg), let d(P-Onuc) relax
  --drag-mode pin-cv-asymmetric # pin only s = d(P-Olg) − d(P-Onuc)
                                # via FixInternals(bondcombos)
  --drag-mode pin-cv-custom     # pin a user-supplied CV defined by
                                # --bond and --cv-formula

Generic bond specification (any number of bonds, used for multi-bond
reactions like Diels-Alder):

  --bond ATOMA,ATOMB              # repeatable; spec is "name.resname"
                                  # (e.g. "P1.SUB,O3.OHX") or 1-based
                                  # PDB serial number
  --bond-role ATOMA,ATOMB:role    # role ∈ {forming,breaking}; optional

Custom CV expression (Python arithmetic on identifiers ``d_<atomA>_<atomB>``,
sanitized — dots become underscores):

  --cv-formula '(d_O7_SUB - d_O3_OHX) / 2'      # default for SN2-at-P
  --cv-formula 'mean(d_C1_C6, d_C2_C5)'         # synchronous Diels-Alder

For multi-bond reactions with K + L > 2 (e.g. Cope rearrangement), the 1-D
CV cannot capture the saddle accurately. Use NEB-CI (``qcb neb``) instead;
see ``docs/multi_bond_design.md`` for the full method-selection roadmap.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from reactive_autodetect import (  # noqa: E402
    DEFAULT_LG_CUTOFF_A,
    _autodetect_reactive_triplet,
)


# ----------------------------------------------------------------------------
# Atom resolution: parse "name.resname" or "<serial>" tokens
# ----------------------------------------------------------------------------
def _atom_token_to_index(token: str, src_atoms) -> int:
    """Resolve an atom-spec token to a 0-based atom index in ``src_atoms``.

    Accepted forms:
      - ``"NAME.RESNAME"`` — match by atom-name + residue-name (first hit)
      - ``"<serial>"`` — 1-based PDB serial (matches a.serial if present,
        otherwise falls back to 1-based positional index)
    """
    token = token.strip()
    if not token:
        raise ValueError("empty atom token")
    if "." in token:
        name, resname = token.split(".", 1)
        for i, a in enumerate(src_atoms):
            if a.name == name and a.resname == resname:
                return i
        raise ValueError(
            f"no atom with name={name!r} resname={resname!r} found in input")
    # numeric serial
    try:
        serial = int(token)
    except ValueError as exc:
        raise ValueError(
            f"atom token {token!r} is neither 'name.resname' nor an integer "
            "serial") from exc
    for i, a in enumerate(src_atoms):
        if getattr(a, "serial", None) == serial:
            return i
    # fallback: 1-based positional index
    if 1 <= serial <= len(src_atoms):
        return serial - 1
    raise ValueError(f"no atom with serial {serial} found in input")


def _atom_label(a) -> str:
    """Stable safe label for use in CV-formula identifiers."""
    return f"{a.name}.{a.resname}"


def _bond_key(label_a: str, label_b: str) -> str:
    """``label`` -> ``d_<label_a>_<label_b>`` with dots -> underscores.

    Order is preserved so the user's --bond order matches their --cv-formula.
    """
    sanitize = lambda s: s.replace(".", "_").replace("-", "_")
    return f"d_{sanitize(label_a)}_{sanitize(label_b)}"


# ----------------------------------------------------------------------------
# Safe CV-formula evaluator. Parses a small arithmetic expression with named
# identifiers (the bond keys) and a whitelist of math helpers. Rejects
# attribute access, calls to anything outside the whitelist, comprehensions,
# imports, and lambda. This is NOT a full sandbox (Python is not safely
# sandboxable in general), but it eliminates the obvious arbitrary-code
# vectors that come with raw eval().
# ----------------------------------------------------------------------------
class _CVFormulaError(ValueError):
    pass


_ALLOWED_FUNCS = {
    "mean": lambda *xs: sum(xs) / len(xs) if xs else 0.0,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_cv_formula(expr: str, names: dict[str, float]) -> float:
    """Evaluate a CV formula expression against a name->float dict."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise _CVFormulaError(f"cv-formula syntax error: {exc}") from exc

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise _CVFormulaError(
                f"cv-formula: only numeric constants allowed, got "
                f"{type(node.value).__name__}")
        if isinstance(node, ast.Name):
            if node.id in names:
                return float(names[node.id])
            if node.id in _ALLOWED_FUNCS:
                # Bare function name (without call) — disallow
                raise _CVFormulaError(
                    f"cv-formula: identifier {node.id!r} can only be used "
                    "as a function call")
            raise _CVFormulaError(
                f"cv-formula: unknown identifier {node.id!r} "
                f"(known bonds: {sorted(names)})")
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise _CVFormulaError(
                    f"cv-formula: operator {type(node.op).__name__} "
                    "not allowed")
            return op(visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise _CVFormulaError(
                    f"cv-formula: unary {type(node.op).__name__} "
                    "not allowed")
            return op(visit(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise _CVFormulaError(
                    "cv-formula: only direct function calls allowed "
                    "(no attribute access, no nested calls on results)")
            fn = _ALLOWED_FUNCS.get(node.func.id)
            if fn is None:
                raise _CVFormulaError(
                    f"cv-formula: function {node.func.id!r} not allowed "
                    f"(allowed: {sorted(_ALLOWED_FUNCS)})")
            if node.keywords:
                raise _CVFormulaError(
                    "cv-formula: keyword arguments not supported")
            args = [visit(a) for a in node.args]
            return float(fn(*args))
        raise _CVFormulaError(
            f"cv-formula: unsupported AST node {type(node).__name__}")

    return float(visit(tree))


def _formula_partials(expr: str, names: dict[str, float],
                       eps: float = 1e-4) -> dict[str, float]:
    """Numerical ∂(formula)/∂(name) for each name (used to project a target s
    onto target d_i values for `pin-cv-custom` initial guesses)."""
    out: dict[str, float] = {}
    for k in names:
        plus = dict(names); plus[k] = names[k] + eps
        minus = dict(names); minus[k] = names[k] - eps
        out[k] = (_eval_cv_formula(expr, plus) - _eval_cv_formula(expr, minus)) / (2 * eps)
    return out


# ----------------------------------------------------------------------------
# Drag-mode dispatch
# ----------------------------------------------------------------------------
DRAG_MODES = ("pin-both", "pin-onuc", "pin-olg",
              "pin-cv-asymmetric", "pin-cv-custom")


def _parse_bond_flag(token: str, src_atoms) -> tuple[int, int, str, str]:
    """Resolve a ``ATOMA,ATOMB`` --bond token to ``(idx_a, idx_b, lbl_a, lbl_b)``."""
    if "," not in token:
        raise ValueError(
            f"--bond must be ATOMA,ATOMB (got {token!r}); "
            "atom spec is 'name.resname' or PDB serial")
    a_tok, b_tok = token.split(",", 1)
    ia = _atom_token_to_index(a_tok.strip(), src_atoms)
    ib = _atom_token_to_index(b_tok.strip(), src_atoms)
    la = _atom_label(src_atoms[ia])
    lb = _atom_label(src_atoms[ib])
    return ia, ib, la, lb


def _parse_bond_role_flag(token: str) -> tuple[str, str]:
    """``ATOMA,ATOMB:role`` -> (``ATOMA,ATOMB``, role)."""
    if ":" not in token:
        raise ValueError(
            f"--bond-role must be ATOMA,ATOMB:role (got {token!r}); "
            "role ∈ {forming,breaking}")
    body, role = token.rsplit(":", 1)
    role = role.strip().lower()
    if role not in ("forming", "breaking"):
        raise ValueError(
            f"--bond-role role must be 'forming' or 'breaking' (got {role!r})")
    return body.strip(), role


# ----------------------------------------------------------------------------
def _parse_s_grid(s_grid_str: str, s_min: float, s_max: float,
                   n_points: int) -> np.ndarray:
    """Resolve the s-axis grid. Explicit list overrides (s_min, s_max, n)."""
    if s_grid_str:
        try:
            grid = [float(x) for x in s_grid_str.split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError(f"--s-grid parse error: {exc}") from exc
        if not grid:
            raise ValueError("--s-grid is empty")
        return np.array(grid, dtype=float)
    return np.linspace(s_min, s_max, n_points)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="mace-polar-m")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--charge", type=int, default=1)

    # ----- s-axis -----
    ap.add_argument("--s-min", type=float, default=-1.5)
    ap.add_argument("--s-max", type=float, default=1.5)
    ap.add_argument("--n-points", type=int, default=11)
    ap.add_argument(
        "--s-grid", type=str, default="",
        help="Explicit comma-separated list of s values (overrides --s-min/"
             "--s-max/--n-points). Useful for non-uniform sampling near the "
             "saddle, e.g. '-1.5,-1.0,-0.5,0,0.3,0.5,1.0,1.5'.",
    )
    ap.add_argument(
        "--s-fine-around-zero", type=int, default=0, metavar="N",
        help="If >0, after the coarse scan re-scan N additional points in "
             "[s_argmax-Δ, s_argmax+Δ] where Δ is the coarse spacing. "
             "Adaptive refinement around the energy maximum.",
    )

    # ----- sum-target / drag mode -----
    ap.add_argument(
        "--sum-target", default="auto",
        help="d(P-Onuc) + d(P-Olg) sum held fixed across scan in pin-both "
             "mode. Default 'auto' computes from the input geometry "
             "(generalizable to any SN2 reaction). Pass a float like '4.25' "
             "to override (e.g. for PTE literature value).",
    )
    ap.add_argument(
        "--drag-mode", choices=DRAG_MODES, default="pin-both",
        help="What to constrain along the scan. "
             "'pin-both' (DEFAULT, legacy behavior) pins both d(P-Onuc) AND "
             "d(P-Olg) at the per-point targets via FixBondLengths. "
             "'pin-onuc' pins only the FIRST listed bond (== forming P-Onuc "
             "in the legacy 2-bond convention). "
             "'pin-olg' pins only the SECOND listed bond (== breaking P-Olg). "
             "'pin-cv-asymmetric' pins only the difference s=d_break-d_form "
             "via FixInternals(bondcombos), letting both individual distances "
             "relax. 'pin-cv-custom' pins an arbitrary CV defined by "
             "--bond + --cv-formula (multi-bond reactions, e.g. Diels-Alder).",
    )
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=200)

    # ----- legacy single-triplet flags (PTE-friendly) -----
    ap.add_argument("--p-name",   default=None)
    ap.add_argument("--p-res",    default=None)
    ap.add_argument("--nuc-name", default=None)
    ap.add_argument("--nuc-res",  default=None)
    ap.add_argument("--lg-name",  default=None)
    ap.add_argument("--lg-res",   default=None)

    # ----- generic multi-bond flags -----
    ap.add_argument(
        "--bond", action="append", default=[], metavar="ATOMA,ATOMB",
        help="Define a CV component bond. Repeat for multi-bond reactions. "
             "Atom spec is 'name.resname' (e.g. 'P1.SUB,O3.OHX') or 1-based "
             "PDB serial number. Used by --drag-mode pin-cv-custom and as "
             "a complement / replacement for --p-name/--nuc-name/--lg-name.",
    )
    ap.add_argument(
        "--bond-role", action="append", default=[], metavar="ATOMA,ATOMB:ROLE",
        help="Annotate a bond's chemical role: 'forming' or 'breaking'. "
             "Optional metadata, recorded in summary.json. ROLE ∈ "
             "{forming,breaking}. Example: 'C1.LIG,C6.LIG:forming'.",
    )
    ap.add_argument(
        "--cv-formula", default="",
        help="Python arithmetic expression for the CV in pin-cv-custom mode. "
             "Identifiers are 'd_<atomA>_<atomB>' (dots in atom labels become "
             "underscores), built from --bond inputs in declaration order. "
             "Whitelist of helpers: mean, sum, min, max, abs, sqrt, exp, log. "
             "Examples: 'mean(d_C1_C6, d_C2_C5)' for synchronous Diels-Alder; "
             "'(d_O7_SUB - d_O3_OHX) / 2' for SN2-at-P. "
             "If empty in pin-cv-custom mode, defaults to "
             "'sum_of_forming - sum_of_breaking'.",
    )

    # ----- autodetect knobs -----
    ap.add_argument(
        "--autodetect-lg-cutoff", type=float, default=DEFAULT_LG_CUTOFF_A,
        help=f"Distance cutoff (Å) for treating a same-residue O/N/halide as "
             f"a candidate leaving group during reactive-triplet auto-detect. "
             f"Default {DEFAULT_LG_CUTOFF_A}. Bump higher (e.g. 3.0) for very "
             f"loose late-TS poses; tighten (e.g. 2.2) for true reactant "
             f"geometries.",
    )

    # ----- scaffold knobs -----
    ap.add_argument(
        "--scaffold-chain", type=str, default="A",
        help="Chain ID whose CA atoms form the rigid scaffold (default 'A'). "
             "Use 'B' for AChE-scaffolded systems where the bulk protein is "
             "in chain B. The substrate and active-site residues should be in "
             "the OPPOSITE chain (or be passed via --free-residues).",
    )
    ap.add_argument(
        "--free-residues", type=str, default="",
        help="Comma-separated residue ids (in --scaffold-chain) to EXCLUDE "
             "from the CA-rigid scaffold (let them move freely during the "
             "scan). E.g. '131' frees W131 in PTE.",
    )
    ap.add_argument(
        "--cv-spring-k", type=float, default=200.0,
        help="Spring constant (eV/Å^2) for soft pin used as a SAFETY FALLBACK "
             "when FixInternals(bondcombos) fails to converge. Currently "
             "unused; reserved for future Hookean-style fallback.",
    )
    ap.add_argument(
        "--min-bond-length", type=float, default=1.4,
        help="Lower-bound (Å) on per-bond initial-guess distances. Points "
             "where a target distance falls below this are skipped (in "
             "pin-both mode) or clamped (in pin-onuc/pin-olg/pin-cv-* modes) "
             "to avoid pushing atoms inside their van der Waals radii. "
             "Default 1.4 (sensible for C-C, C-O, P-O, S-O); lower for "
             "metal-ligand or hypervalent bonds.",
    )

    args = ap.parse_args()

    free_residues = (
        {int(x) for x in args.free_residues.split(",") if x.strip()}
        if args.free_residues else set()
    )

    args.out.mkdir(parents=True, exist_ok=True)
    input_copy = args.out / args.input.name
    if args.input.resolve() != input_copy.resolve():
        shutil.copy2(args.input, input_copy)
        print(f"copied input → {input_copy}")

    from polish_ts_v2 import write_pdb_from_template, parse_pdb
    from ase.io import read
    from ase.constraints import FixInternals, FixBondLengths
    from ase.optimize import LBFGS
    from quantum_engine.calc import make_calc

    src_atoms, _ = parse_pdb(args.input)
    ca_idx = [i for i, a in enumerate(src_atoms)
              if a.chain == args.scaffold_chain and a.name == "CA"
              and a.resseq not in free_residues]
    print(f"scaffold chain: {args.scaffold_chain!r}; "
          f"{len(ca_idx)} CA atom(s) used in CA-rigid scaffold")
    if free_residues:
        print(f"--free-residues {sorted(free_residues)}: "
              f"{len(free_residues)} residue(s) excluded from CA-rigid scaffold")

    def find(name, res):
        for i, a in enumerate(src_atoms):
            if a.name == name and a.resname == res:
                return i
        raise RuntimeError(f"missing {name} in {res}")

    autodetect_log: list[str] = []

    # ----- Resolve bond list ------------------------------------------------
    # If --bond flags are given, they win. Otherwise fall back to the
    # legacy --p-name/--nuc-name/--lg-name → forming(P-Onuc)+breaking(P-Olg)
    # path. For backward compat with the previous default behavior, when
    # NEITHER --bond nor any --p-name flag is given, we autodetect a triplet.
    bonds: list[dict] = []  # each: {idx: (a,b), label: (la,lb), role: str|None, key: str}
    role_overrides: list[tuple[str, str]] = []
    for tok in args.bond_role:
        body, role = _parse_bond_role_flag(tok)
        role_overrides.append((body, role))

    use_legacy_triplet = (
        not args.bond
        and (
            any(getattr(args, k) for k in ("p_name", "p_res", "nuc_name",
                                            "nuc_res", "lg_name", "lg_res"))
            or args.drag_mode in ("pin-both", "pin-onuc", "pin-olg",
                                   "pin-cv-asymmetric")
        )
    )

    P = ON = OL = None
    if use_legacy_triplet:
        if all(getattr(args, k) for k in ("p_name", "p_res", "nuc_name",
                                            "nuc_res", "lg_name", "lg_res")):
            P  = find(args.p_name,  args.p_res)
            ON = find(args.nuc_name, args.nuc_res)
            OL = find(args.lg_name, args.lg_res)
            autodetect_log.append(
                f"reactive_triplet=user_override "
                f"P={args.p_name}.{args.p_res} "
                f"Nuc={args.nuc_name}.{args.nuc_res} "
                f"LG={args.lg_name}.{args.lg_res}")
        else:
            P, ON, OL, log_lines = _autodetect_reactive_triplet(
                src_atoms, lg_cutoff_a=args.autodetect_lg_cutoff)
            autodetect_log.extend(log_lines)
            if args.p_name and args.p_res:
                P = find(args.p_name, args.p_res)
            if args.nuc_name and args.nuc_res:
                ON = find(args.nuc_name, args.nuc_res)
            if args.lg_name and args.lg_res:
                OL = find(args.lg_name, args.lg_res)

        # Encode the triplet as the canonical bond list
        # (forming P-Onuc, breaking P-Olg). Order matters for cv-formula
        # default and for pin-cv-asymmetric.
        bonds.append({
            "idx": (P, ON),
            "label": (_atom_label(src_atoms[P]), _atom_label(src_atoms[ON])),
            "role": "forming",
            "key": _bond_key(_atom_label(src_atoms[P]),
                              _atom_label(src_atoms[ON])),
        })
        bonds.append({
            "idx": (P, OL),
            "label": (_atom_label(src_atoms[P]), _atom_label(src_atoms[OL])),
            "role": "breaking",
            "key": _bond_key(_atom_label(src_atoms[P]),
                              _atom_label(src_atoms[OL])),
        })

    if args.bond:
        # User-supplied generic bond list — overrides / replaces the
        # legacy triplet. Roles default to None unless --bond-role specified.
        if bonds:
            autodetect_log.append(
                "--bond flags supplied; OVERRIDING legacy triplet bond list")
        bonds = []
        for tok in args.bond:
            ia, ib, la, lb = _parse_bond_flag(tok, src_atoms)
            role = None
            for ro_body, ro_role in role_overrides:
                # match the body string against the original token
                # (ignoring whitespace)
                if ro_body.replace(" ", "") == tok.replace(" ", ""):
                    role = ro_role
                    break
            bonds.append({
                "idx": (ia, ib),
                "label": (la, lb),
                "role": role,
                "key": _bond_key(la, lb),
            })

    if not bonds:
        raise SystemExit(
            "no CV bonds defined. Either give --bond ... flags, "
            "use the legacy --p-name/--nuc-name/--lg-name triplet, or "
            "let auto-detect handle it (the default).")

    print(f"CV bonds ({len(bonds)}):")
    for b in bonds:
        ia, ib = b["idx"]
        la, lb = b["label"]
        d0 = float(np.linalg.norm(
            np.array([src_atoms[ia].x, src_atoms[ia].y, src_atoms[ia].z]) -
            np.array([src_atoms[ib].x, src_atoms[ib].y, src_atoms[ib].z])))
        role_str = f" role={b['role']}" if b['role'] else ""
        print(f"  {b['key']}: {la}#{ia} <-> {lb}#{ib}  d0={d0:.3f}A{role_str}")

    # ----- Resolve CV formula -----------------------------------------------
    # In pin-both / pin-onuc / pin-olg / pin-cv-asymmetric the formula is
    # implicit s = d_break - d_form (or equivalently d(P-Olg)-d(P-Onuc)).
    # In pin-cv-custom user supplies it (or we synthesize from roles).
    cv_formula = args.cv_formula.strip()
    if not cv_formula:
        # synthesize from bond list
        forming = [b for b in bonds if b["role"] == "forming"]
        breaking = [b for b in bonds if b["role"] == "breaking"]
        if forming and breaking:
            f_part = "(" + " + ".join(b["key"] for b in breaking) + ")"
            n_part = "(" + " + ".join(b["key"] for b in forming) + ")"
            cv_formula = f"{f_part} - {n_part}"
        elif len(bonds) == 2 and not (forming or breaking):
            # legacy convention: bonds[0] is forming, bonds[1] is breaking
            cv_formula = f"{bonds[1]['key']} - {bonds[0]['key']}"
        else:
            # arbitrary multi-bond synchronous CV: mean of all
            cv_formula = f"mean({', '.join(b['key'] for b in bonds)})"
    autodetect_log.append(f"cv_formula='{cv_formula}'")
    print(f"cv-formula: {cv_formula}")

    # ----- sum_target resolution (pin-both only) ---------------------------
    coords0 = np.array([[a.x, a.y, a.z] for a in src_atoms])
    if args.drag_mode == "pin-both":
        if P is None or ON is None or OL is None:
            raise SystemExit(
                "pin-both requires the legacy P/Nuc/LG triplet. "
                "For multi-bond CV scans use --drag-mode pin-cv-custom.")
        auto_sum = float(np.linalg.norm(coords0[P] - coords0[ON]) +
                          np.linalg.norm(coords0[P] - coords0[OL]))
        if isinstance(args.sum_target, str) and args.sum_target.lower() == "auto":
            sum_target = auto_sum
            autodetect_log.append(
                f"sum_target={sum_target:.4f}A source=input_geometry")
        else:
            sum_target = float(args.sum_target)
            autodetect_log.append(
                f"sum_target={sum_target:.4f}A source=user_override "
                f"(input_geom would have given {auto_sum:.4f})")
            if abs(sum_target - auto_sum) > 1.0:
                print(f"WARNING: --sum-target {sum_target:.2f} differs from "
                       f"input geometry sum {auto_sum:.2f} by >1.0 Å; input "
                       "pose may not be near the saddle.")
        args.sum_target = sum_target
    else:
        # sum_target is unused in non-pin-both modes; just stash for summary
        if isinstance(args.sum_target, str) and args.sum_target.lower() == "auto":
            args.sum_target = None
        else:
            try:
                args.sum_target = float(args.sum_target)
            except (TypeError, ValueError):
                args.sum_target = None

    print("[autodetect]", *("\n  - " + ln for ln in autodetect_log))

    # ----- s-axis values ---------------------------------------------------
    s_values = _parse_s_grid(args.s_grid, args.s_min, args.s_max, args.n_points)
    grid_descr = (
        f"explicit list (n={len(s_values)})" if args.s_grid
        else f"linspace[{args.s_min:+.2f}, {args.s_max:+.2f}, n={args.n_points}]"
    )
    print(f"scan s grid: {grid_descr}")

    # Initial bond distances from input geometry (for cv-formula targeting)
    def _bond_dists_from_atoms(atoms_pos: np.ndarray) -> dict[str, float]:
        return {
            b["key"]: float(np.linalg.norm(
                atoms_pos[b["idx"][0]] - atoms_pos[b["idx"][1]]))
            for b in bonds
        }

    d0 = _bond_dists_from_atoms(coords0)
    s0 = _eval_cv_formula(cv_formula, d0)
    print(f"initial geometry: {d0}; s0={s0:.4f}")

    # ----- Per-point loop ---------------------------------------------------
    def _run_one_point(k: int, s_target: float) -> dict | None:
        atoms = read(str(args.input))
        pos = atoms.get_positions().copy()

        # Compute the per-bond initial guess for the constraints. Strategy
        # depends on drag_mode:
        target_d: dict[str, float] = {}
        min_d = args.min_bond_length
        if args.drag_mode == "pin-both":
            d_pn = (args.sum_target - s_target) / 2.0
            d_pl = (args.sum_target + s_target) / 2.0
            if d_pn < min_d or d_pl < min_d:
                print(f"point {k}: d_pn={d_pn:.3f} or d_pl={d_pl:.3f} "
                       f"< min-bond-length {min_d}; skipping")
                return None
            target_d[bonds[0]["key"]] = d_pn  # forming P-Onuc
            target_d[bonds[1]["key"]] = d_pl  # breaking P-Olg
        elif args.drag_mode == "pin-onuc":
            # s = d_break - d_form. Pin only d_form (=Onuc); allow d_break
            # to relax. Initial guess: d_form = max(min_d, d0[form] - (s-s0)/2).
            d_pn = max(min_d, d0[bonds[0]["key"]] - (s_target - s0) / 2.0)
            target_d[bonds[0]["key"]] = d_pn
        elif args.drag_mode == "pin-olg":
            d_pl = max(min_d, d0[bonds[1]["key"]] + (s_target - s0) / 2.0)
            target_d[bonds[1]["key"]] = d_pl
        elif args.drag_mode == "pin-cv-asymmetric":
            ds = s_target - s0
            target_d[bonds[0]["key"]] = max(min_d, d0[bonds[0]["key"]] - ds / 2.0)
            target_d[bonds[1]["key"]] = max(min_d, d0[bonds[1]["key"]] + ds / 2.0)
        elif args.drag_mode == "pin-cv-custom":
            partials = _formula_partials(cv_formula, d0)
            denom = sum(p * p for p in partials.values())
            if denom < 1e-12:
                for b in bonds:
                    target_d[b["key"]] = d0[b["key"]]
            else:
                for b in bonds:
                    p = partials[b["key"]]
                    target_d[b["key"]] = max(
                        min_d, d0[b["key"]] + (s_target - s0) * p / denom)
        else:
            raise AssertionError(f"unknown drag-mode {args.drag_mode}")

        # Apply target_d as initial-guess displacement (pull each constrained
        # atom along its current bond vector toward target distance).
        # The first atom of each bond is the "anchor"; the second moves.
        for b in bonds:
            if b["key"] not in target_d:
                continue
            ia, ib = b["idx"]
            v = pos[ib] - pos[ia]
            d_now = float(np.linalg.norm(v))
            if d_now < 1e-9:
                continue
            t = target_d[b["key"]]
            pos[ib] = pos[ia] + (v / d_now) * t

        atoms.set_positions(pos)
        atoms.info["charge"] = args.charge
        atoms.calc = make_calc(args.model, device=args.device, charge=args.charge)

        # Build CA-CA scaffold bonds
        ca_positions = atoms.get_positions()
        scaffold_bonds = []
        for ii in range(len(ca_idx)):
            for jj in range(ii + 1, len(ca_idx)):
                i, j = ca_idx[ii], ca_idx[jj]
                scaffold_bonds.append([
                    float(np.linalg.norm(ca_positions[i] - ca_positions[j])),
                    [i, j],
                ])

        # Constraints by drag mode
        if args.drag_mode == "pin-both":
            constraints = [
                FixInternals(bonds=scaffold_bonds, epsilon=1e-7),
                FixBondLengths([
                    (bonds[0]["idx"][0], bonds[0]["idx"][1]),
                    (bonds[1]["idx"][0], bonds[1]["idx"][1]),
                ]),
            ]
        elif args.drag_mode == "pin-onuc":
            constraints = [
                FixInternals(bonds=scaffold_bonds, epsilon=1e-7),
                FixBondLengths([(bonds[0]["idx"][0], bonds[0]["idx"][1])]),
            ]
        elif args.drag_mode == "pin-olg":
            constraints = [
                FixInternals(bonds=scaffold_bonds, epsilon=1e-7),
                FixBondLengths([(bonds[1]["idx"][0], bonds[1]["idx"][1])]),
            ]
        elif args.drag_mode == "pin-cv-asymmetric":
            # bondcombo: target_value = +1*d(P,Olg) -1*d(P,Onuc) = s
            # (matches the convention s = d_break - d_form throughout)
            ia0, ib0 = bonds[0]["idx"]  # forming
            ia1, ib1 = bonds[1]["idx"]  # breaking
            bondcombo = [s_target, [
                [ia1, ib1, +1.0],
                [ia0, ib0, -1.0],
            ]]
            # Combine with CA scaffold via a single FixInternals
            constraints = [
                FixInternals(bonds=scaffold_bonds,
                              bondcombos=[bondcombo],
                              epsilon=1e-7),
            ]
        elif args.drag_mode == "pin-cv-custom":
            # Build a bondcombo from the cv-formula partial derivatives
            # at the input-geometry point. Linearization is exact for affine
            # formulas and a reasonable surrogate for nonlinear ones near
            # the input pose.
            partials = _formula_partials(cv_formula, d0)
            atom_pairs = []
            for b in bonds:
                ia, ib = b["idx"]
                w = partials[b["key"]]
                if abs(w) < 1e-12:
                    continue
                atom_pairs.append([ia, ib, float(w)])
            if not atom_pairs:
                raise SystemExit(
                    "pin-cv-custom: cv-formula has no derivative wrt any "
                    "--bond at the input geometry; check formula vs bonds.")
            # Compute target value of the linearized CV at this s_target.
            # Linearization: cv ≈ cv0 + Σ partial_i * (d_i - d0_i).
            # Constraint: cv = s_target  ⇒  Σ partial_i * d_i =
            #   s_target - cv0 + Σ partial_i * d0_i.
            cv0 = _eval_cv_formula(cv_formula, d0)
            offset = s_target - cv0 + sum(
                partials[b["key"]] * d0[b["key"]] for b in bonds)
            bondcombo = [offset, atom_pairs]
            constraints = [
                FixInternals(bonds=scaffold_bonds,
                              bondcombos=[bondcombo],
                              epsilon=1e-7),
            ]
        else:
            raise AssertionError(f"unknown drag-mode {args.drag_mode}")

        atoms.set_constraint(constraints)
        opt = LBFGS(atoms, logfile=str(args.out / f"point_{k:02d}.log"))
        converged = opt.run(fmax=args.fmax, steps=args.max_steps)
        e = float(atoms.get_potential_energy())
        fmax_final = float(np.linalg.norm(atoms.get_forces(), axis=1).max())

        # Measure final per-bond distances and the realized CV
        d_final = _bond_dists_from_atoms(atoms.get_positions())
        s_actual = _eval_cv_formula(cv_formula, d_final)

        # legacy fields for backward compat with downstream consumers
        d_pn_final = d_final.get(bonds[0]["key"]) if bonds else None
        d_pl_final = d_final.get(bonds[1]["key"]) if len(bonds) > 1 else None

        pdb_path = args.out / f"point_{k:02d}_s{s_target:+.3f}.pdb"
        extra_remarks = [
            f"REMARK QCB SCAN_POINT {k}",
            f"REMARK QCB DRAG_MODE {args.drag_mode}",
            f"REMARK QCB s_target {s_target:+.4f}",
            f"REMARK QCB s_actual {s_actual:+.4f}",
            f"REMARK QCB ENERGY_eV {e:.4f}",
            f"REMARK QCB FMAX_eV_per_A {fmax_final:.6f}",
            f"REMARK QCB CONVERGED {bool(converged)}",
        ]
        # Legacy SN2-at-P remark fields (preserved so downstream parsers
        # that only know d_P_Onuc / d_P_Olg keep working in pin-both mode).
        if d_pn_final is not None:
            extra_remarks.append(f"REMARK QCB d_P_Onuc {d_pn_final:.4f}")
        if d_pl_final is not None:
            extra_remarks.append(f"REMARK QCB d_P_Olg {d_pl_final:.4f}")
        for b in bonds:
            extra_remarks.append(
                f"REMARK QCB {b['key']} {d_final[b['key']]:.4f}")
        write_pdb_from_template(args.input, atoms.get_positions(), pdb_path,
                                 extra_remarks=extra_remarks)

        result = {
            "point": k,
            "s_target": float(s_target),
            "s_actual": float(s_actual),
            "energy_eV": e,
            "fmax_eV_per_A": fmax_final,
            "converged": bool(converged),
            "pdb": str(pdb_path),
            "bond_distances": {k_: float(v_) for k_, v_ in d_final.items()},
        }
        # Legacy aliases for the SN2-at-P consumers
        if d_pn_final is not None:
            result["d_P_Onuc"] = d_pn_final
        if d_pl_final is not None:
            result["d_P_Olg"] = d_pl_final
        print(f"  E={e:.4f} eV  fmax={fmax_final:.4f}  "
              f"s_actual={s_actual:+.4f}  conv={converged}")
        return result

    # Run the coarse scan first
    results: list[dict] = []
    for k, s in enumerate(s_values):
        print(f"\n--- point {k}: s_target={s:+.3f}  drag-mode={args.drag_mode}")
        r = _run_one_point(k, float(s))
        if r is not None:
            results.append(r)

    # Optional adaptive refinement around argmax energy
    if results and args.s_fine_around_zero > 0:
        energies = [r["energy_eV"] for r in results]
        max_idx_coarse = int(np.argmax(energies))
        s_max_coarse = results[max_idx_coarse]["s_target"]
        # Δ = coarse spacing = mean of |Δs|
        if len(s_values) > 1:
            delta = float(np.mean(np.abs(np.diff(s_values))))
        else:
            delta = 0.5
        N = args.s_fine_around_zero
        fine_s = np.linspace(s_max_coarse - delta, s_max_coarse + delta, N)
        print(f"\n*** ADAPTIVE REFINEMENT: re-scanning {N} points in "
               f"[{fine_s[0]:+.3f}, {fine_s[-1]:+.3f}] around s={s_max_coarse:+.3f}")
        for k_off, s in enumerate(fine_s):
            k = len(s_values) + k_off
            print(f"\n--- fine point {k}: s_target={s:+.3f}")
            r = _run_one_point(k, float(s))
            if r is not None:
                results.append(r)

    if not results:
        print("no scan points completed", file=sys.stderr)
        return 1

    # Find energy max → TS guess
    energies = [r["energy_eV"] for r in results]
    max_idx = int(np.argmax(energies))
    ts_guess = results[max_idx]
    print(f"\n*** ENERGY MAX: point {ts_guess['point']} "
           f"s={ts_guess['s_target']:+.3f} E={ts_guess['energy_eV']:.4f} eV ***")
    print(f"    pdb: {ts_guess['pdb']}")

    warnings_list: list[str] = []
    if max_idx == 0 or max_idx == len(results) - 1:
        warnings_list.append(
            f"WARNING: argmax at endpoint (index {max_idx}/{len(results)-1}); "
            "no interior maximum found. Widen the scan range or shift the "
            "input geometry closer to the expected saddle.")
    diffs = np.diff(energies)
    if np.all(diffs >= -1e-4) or np.all(diffs <= 1e-4):
        warnings_list.append(
            "WARNING: scan profile is monotonic; no interior saddle found.")
    for w in warnings_list:
        print(w)

    min_e = min(energies)
    EV = 23.0605
    profile = [
        {"s": r["s_actual"], "E_rel_kcal": (r["energy_eV"] - min_e) * EV,
         "pdb": r["pdb"]}
        for r in results
    ]

    summary: dict[str, Any] = {
        "input": str(args.input),
        "model": args.model,
        "charge": args.charge,
        "drag_mode": args.drag_mode,
        "cv_formula": cv_formula,
        "sum_target_A": args.sum_target,
        "bonds": [
            {
                "key": b["key"],
                "atom_a": {
                    "index": int(b["idx"][0]),
                    "label": b["label"][0],
                },
                "atom_b": {
                    "index": int(b["idx"][1]),
                    "label": b["label"][1],
                },
                "role": b["role"],
            }
            for b in bonds
        ],
        "autodetect_log": autodetect_log,
        "scan_n_points": len(results),
        "ts_guess_index": max_idx,
        "ts_guess_pdb": ts_guess["pdb"],
        "ts_guess_s": ts_guess["s_actual"],
        "ts_guess_energy_eV": ts_guess["energy_eV"],
        "barrier_R_to_TS_kcal": (
            ts_guess["energy_eV"]
            - min(r["energy_eV"] for r in results[:max_idx + 1])
        ) * EV,
        "barrier_P_to_TS_kcal": (
            ts_guess["energy_eV"]
            - min(r["energy_eV"] for r in results[max_idx:])
        ) * EV,
        "barrier_R_to_TS_kcal_endpoint": (
            ts_guess["energy_eV"] - results[0]["energy_eV"]
        ) * EV,
        "barrier_P_to_TS_kcal_endpoint": (
            ts_guess["energy_eV"] - results[-1]["energy_eV"]
        ) * EV,
        "scan_results": results,
        "energy_profile_kcal": profile,
        "warnings": warnings_list,
    }

    # Legacy fields for downstream consumers that expect SN2-at-P scan output.
    # Only emitted when the run is in fact an SN2-at-P scan with the standard
    # 2-bond setup.
    if (
        len(bonds) == 2
        and bonds[0]["role"] == "forming"
        and bonds[1]["role"] == "breaking"
        and "d_P_Onuc" in ts_guess
        and "d_P_Olg" in ts_guess
    ):
        ia_p, ia_n = bonds[0]["idx"]
        _, ia_l = bonds[1]["idx"]
        summary["reactive_triplet"] = {
            "P":   {"index": int(ia_p), "name": src_atoms[ia_p].name,
                     "resname": src_atoms[ia_p].resname},
            "Nuc": {"index": int(ia_n), "name": src_atoms[ia_n].name,
                     "resname": src_atoms[ia_n].resname},
            "LG":  {"index": int(ia_l), "name": src_atoms[ia_l].name,
                     "resname": src_atoms[ia_l].resname},
        }
        summary["ts_guess_d_P_Onuc"] = ts_guess["d_P_Onuc"]
        summary["ts_guess_d_P_Olg"] = ts_guess["d_P_Olg"]

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote: {args.out / 'summary.json'}")
    print(f"barrier R(s={results[0]['s_actual']:.2f}) → "
           f"TS(s={ts_guess['s_actual']:.2f}) = "
           f"{summary['barrier_R_to_TS_kcal']:.2f} kcal/mol")
    return 0


if __name__ == "__main__":
    sys.exit(main())
