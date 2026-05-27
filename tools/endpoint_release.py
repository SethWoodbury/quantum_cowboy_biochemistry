#!/usr/bin/env python
"""endpoint_release.py — clean up reactant / product endpoints before NEB.

Why this exists
---------------
The classical 1-D scan path produces R-side and P-side structures (e.g.
``point_02_s-0.900.pdb`` and ``point_09_s+1.200.pdb``) that still carry the
FixBondLengths / FixInternals constraints used to slide the system along the
collective variable. If those structures are dropped straight into a NEB
run, the path will start with a **kinked** segment near each endpoint:

    R --- (constrained "R-with-CV-frozen") --- ... --- (constrained P) --- P

NEB then has to undo the artifact during chain optimisation, which is slow
and can mis-tag the climbing image. The fix is to **release** the reactive-
bond constraints at each endpoint, keep the cluster boundary fixed (CA
scaffold, water lattice, whatever the user used in the scan), and relax to a
tighter fmax. The released structures become the new NEB endpoints.

Reaction-agnostic
-----------------
- No PTE-specific atom names. The user supplies the reactive bond list via
  ``--release-bond ATOMA,ATOMB`` (1-based PDB serial OR ``NAME.RESNAME``).
- Boundary preservation is delegated to the existing
  :mod:`quantum_engine.select` grammar and the same ``--fix-preset`` /
  ``--fix`` / ``--free`` flags as the rest of the CLI.
- Distinct from ``qcb opt`` because ``opt`` would simply remove ALL
  constraints; here we surgically remove the reactive-bond constraints
  while keeping the boundary scaffold intact.

CLI surface
-----------
::

    qcb endpoint-release scan_R.pdb \\
        --release-bond P.SUB,O3.OHX --release-bond P.SUB,O7.SUB \\
        --boundary-fix-preset ca-only \\
        --fmax 0.02 \\
        --out R_released.pdb \\
        --charge-ledger ledger.yaml

Public API
----------
``release_endpoint(input_pdb, output_pdb, release_bonds, **knobs)`` runs the
release in-process; suitable for orchestrators that want to chain the call.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quantum_engine.io import load_structure, parse_constraints, build_fix_atoms, write_pdb
from quantum_engine.ops import opt as opt_op
from quantum_engine.ops.charge_ledger import (
    ChargeLedger, append_remarks_to_pdb, inject_into_atoms, resolve_charge_and_spin,
)
from quantum_engine.select import preset_to_specs

log = logging.getLogger("tools.endpoint_release")


# ---------------------------------------------------------------------------
# Atom token resolution (shared idiom with scan_along_s.py)
# ---------------------------------------------------------------------------
def _resolve_atom_token(tok: str, ase_atoms, bt_struct) -> int:
    """Resolve an atom-spec token to a 0-based ASE index.

    Accepted forms:
      - ``NAME.RESNAME`` (first match by atom_name + res_name in the biotite
        template)
      - ``NAME.RESNAME.RESID`` (atom_name + res_name + res_id)
      - bare integer = 1-based PDB serial (= ASE index + 1)
      - ``0:N`` = 0-based ASE index
    """
    s = tok.strip()
    if not s:
        raise ValueError("empty atom token")
    if s.startswith("0:"):
        return int(s[2:])
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s) - 1  # 1-based serial
    if "." in s:
        parts = s.split(".")
        if bt_struct is None:
            raise ValueError(
                f"atom token {tok!r} requires a PDB template (input was non-PDB)"
            )
        if len(parts) == 2:
            name, resname = parts
            mask = (np.asarray(bt_struct.atom_name) == name) & \
                   (np.asarray(bt_struct.res_name) == resname)
        elif len(parts) == 3:
            name, resname, res_id_s = parts
            mask = ((np.asarray(bt_struct.atom_name) == name)
                    & (np.asarray(bt_struct.res_name) == resname)
                    & (np.asarray(bt_struct.res_id) == int(res_id_s)))
        else:
            raise ValueError(
                f"atom token {tok!r}: only NAME.RESNAME or NAME.RESNAME.RESID accepted"
            )
        idxs = np.where(mask)[0]
        if not len(idxs):
            raise ValueError(f"atom token {tok!r} not found in input")
        return int(idxs[0])
    raise ValueError(
        f"atom token {tok!r}: must be 'NAME.RESNAME[.RESID]', integer "
        "(1-based serial), or '0:idx'"
    )


def _parse_release_bonds(specs: Iterable) -> list[tuple[str, str]]:
    """Parse ``--release-bond ATOMA,ATOMB`` specs into raw token pairs.

    Accepts a list whose elements are EITHER:
        - a comma-separated string ``"A,B"`` (CLI form), or
        - a pre-split tuple/list ``("A", "B")`` (Python-API form).
    """
    out: list[tuple[str, str]] = []
    for spec in specs or []:
        if isinstance(spec, (tuple, list)):
            if len(spec) != 2:
                raise ValueError(
                    f"--release-bond {spec!r} must have exactly 2 atom tokens"
                )
            a, b = (str(spec[0]).strip(), str(spec[1]).strip())
        elif isinstance(spec, str):
            if "," not in spec:
                raise ValueError(f"--release-bond {spec!r} must be 'A,B'")
            a, b = (s.strip() for s in spec.split(",", 1))
        else:
            raise TypeError(
                f"--release-bond entries must be str or (a,b) tuple; "
                f"got {type(spec).__name__}"
            )
        if not a or not b:
            raise ValueError(f"--release-bond {spec!r}: empty atom token")
        out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class EndpointReleaseResult:
    status: str
    converged: bool
    fmax_final: float
    energy_eV: float
    n_atoms: int
    n_fixed_atoms: int
    released_bonds: list[tuple[int, int, float, float]]  # (i, j, d_before, d_after)
    output_pdb: Path
    summary_json: Path
    ledger: dict | None = None
    warnings: list[str] = field(default_factory=list)


def release_endpoint(
    input_pdb: str | Path,
    output_pdb: str | Path,
    release_bonds: Iterable[tuple[str, str]] | None = None,
    *,
    boundary_fix_preset: str | None = None,
    fix_specs: Iterable[str] | None = None,
    free_specs: Iterable[str] | None = None,
    model: str = "mace-omol",
    head: str | None = None,
    device: str = "cuda",
    charge_ledger: ChargeLedger | None = None,
    cli_charge: int | None = None,
    cli_spin: int | None = None,
    fmax: float = 0.02,
    max_steps: int = 500,
    optimizer: str = "lbfgs",
    log_release_distances: bool = True,
    extra_remarks: Iterable[str] = (),
) -> EndpointReleaseResult:
    """Relax an endpoint with reactive-bond constraints removed.

    Args:
        input_pdb: scan-output endpoint PDB.
        output_pdb: where to write the released PDB.
        release_bonds: list of ``(atom_a_token, atom_b_token)`` pairs to free.
            These are the bonds whose CV constraint is removed; tokens use
            the same grammar as :func:`_resolve_atom_token`. If ``None``, no
            bond constraints are explicitly released — but the optimizer
            still gets only the ``fix_specs`` / preset, so any FixBondLengths
            already encoded in the file is dropped (ASE doesn't carry them
            through PDB I/O anyway).
        boundary_fix_preset: ``ca-only`` / ``backbone`` / ``backbone-water`` /
            ``none`` — applied to the relaxation. Defaults to no preset.
        fix_specs: extra constraint specs (``residue X``, ``chain B``, ...)
            applied on top of the preset.
        free_specs: subtractive specs that override the fix mask.
        model, head, device: calculator factory args.
        charge_ledger: optional :class:`ChargeLedger`; takes precedence over
            ``cli_charge`` / ``cli_spin``.
        cli_charge, cli_spin: fall-backs when no ledger is supplied.
        fmax: convergence target (eV/Å). Tighter than scan defaults — the
            whole point of release is to remove the residual gradient.
        max_steps: optimizer iteration cap.
        optimizer: ``lbfgs`` / ``bfgs`` / ``fire``.
        log_release_distances: if True, record ``d(A,B)`` before and after
            relaxation for each released bond so the user can confirm the
            geometry actually moved.
        extra_remarks: free-form REMARK lines to append (e.g. lineage info).

    Returns:
        :class:`EndpointReleaseResult`.

    Raises:
        FileNotFoundError, ValueError on bad inputs (fail-loud).
    """
    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)
    if not input_pdb.is_file():
        raise FileNotFoundError(f"endpoint_release: input not found: {input_pdb}")
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    log.info("endpoint_release: input=%s -> %s", input_pdb, output_pdb)

    atoms, bt_struct, charge_hint = load_structure(input_pdb)

    # Resolve charge / spin
    charge, spin, ledger = resolve_charge_and_spin(
        ledger_path=None,
        cli_charge=cli_charge,
        cli_spin=cli_spin,
        pdb_charge_hint=charge_hint,
    )
    if charge_ledger is not None:
        ledger = charge_ledger
        charge = ledger.total
        spin = ledger.spin

    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    if ledger is not None:
        inject_into_atoms(atoms, ledger)

    # Build calculator
    from quantum_engine.calc import make_calc
    atoms.calc = make_calc(model=model, head=head, device=device, charge=charge)

    # Resolve release-bond indices and record pre-release distances
    bonds_raw = _parse_release_bonds(release_bonds or [])
    released: list[tuple[int, int, float, float]] = []  # (i, j, d_before, d_after)
    pos = atoms.get_positions()
    for a_tok, b_tok in bonds_raw:
        i = _resolve_atom_token(a_tok, atoms, bt_struct)
        j = _resolve_atom_token(b_tok, atoms, bt_struct)
        d_before = float(np.linalg.norm(pos[i] - pos[j]))
        released.append((i, j, d_before, d_before))  # placeholder; updated post-relax
        log.info(
            "endpoint_release: releasing bond %s(%d) - %s(%d), d_before=%.4f Å",
            a_tok, i, b_tok, j, d_before,
        )

    # Build boundary constraint
    fix_specs = list(fix_specs or [])
    free_specs = list(free_specs or [])
    excluded: set[str] = set()
    if boundary_fix_preset:
        preset_specs, preset_excluded = preset_to_specs(boundary_fix_preset, None)
        fix_specs = preset_specs + fix_specs
        excluded |= preset_excluded

    constraint = None
    if fix_specs or free_specs:
        fix_mask = parse_constraints(atoms, bt_struct, fix_specs, excluded) if fix_specs \
                   else np.zeros(len(atoms), dtype=bool)
        if free_specs:
            free_mask = parse_constraints(atoms, bt_struct, free_specs, set())
            fix_mask &= ~free_mask
        constraint = build_fix_atoms(fix_mask)
        log.info("endpoint_release: boundary fixed = %d / %d atoms",
                 int(fix_mask.sum()), len(atoms))

    n_fixed = 0
    if constraint is not None and hasattr(constraint, "index"):
        n_fixed = len(constraint.index)

    # Relax
    relax_outdir = output_pdb.parent / f"{output_pdb.stem}_relax"
    res = opt_op.run(
        atoms,
        atoms.calc,
        relax_outdir,
        constraint=constraint,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
    )
    relaxed_atoms = res["atoms"]
    fmax_final = float(np.linalg.norm(relaxed_atoms.get_forces(), axis=1).max())
    energy_eV = float(res.get("energy_eV", relaxed_atoms.get_potential_energy()))
    converged = bool(res.get("converged", fmax_final <= fmax * 1.5))

    # Update post-release distances
    pos_after = relaxed_atoms.get_positions()
    new_released: list[tuple[int, int, float, float]] = []
    for (i, j, d_before, _) in released:
        d_after = float(np.linalg.norm(pos_after[i] - pos_after[j]))
        new_released.append((i, j, d_before, d_after))
        if log_release_distances:
            log.info(
                "endpoint_release: bond %d-%d  d_before=%.4f  d_after=%.4f  Δ=%+.4f Å",
                i, j, d_before, d_after, d_after - d_before,
            )

    # Write released PDB
    write_pdb(relaxed_atoms, bt_struct, output_pdb,
              total_charge=charge, energy_eV=energy_eV)

    # Append ledger remarks + lineage
    if ledger is not None:
        append_remarks_to_pdb(output_pdb, ledger)
    if extra_remarks:
        with open(output_pdb, "r+") as fh:
            text = fh.read()
            extras = "\n".join(f"REMARK QCB ENDPOINT_RELEASE {l}"
                               for l in extra_remarks) + "\n"
            # insert before first ATOM/HETATM/MODEL
            import re
            m = re.search(r"^(?:ATOM|HETATM|MODEL|END)\b", text, re.MULTILINE)
            if m:
                text = text[: m.start()] + extras + text[m.start():]
            else:
                text += extras
            fh.seek(0)
            fh.truncate()
            fh.write(text)

    summary_path = output_pdb.parent / f"{output_pdb.stem}.endpoint_release.json"
    summary = {
        "input_pdb": str(input_pdb),
        "output_pdb": str(output_pdb),
        "model": model,
        "device": device,
        "charge": charge,
        "spin": spin,
        "fmax_final": fmax_final,
        "fmax_target": fmax,
        "max_steps": max_steps,
        "converged": converged,
        "energy_eV": energy_eV,
        "n_atoms": len(relaxed_atoms),
        "n_fixed_atoms": n_fixed,
        "boundary_fix_preset": boundary_fix_preset,
        "released_bonds": [
            {"i": i, "j": j, "d_before_A": d_b, "d_after_A": d_a,
             "delta_A": d_a - d_b}
            for (i, j, d_b, d_a) in new_released
        ],
        "ledger": ledger.to_dict() if ledger is not None else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    log.info(
        "endpoint_release: %s, fmax=%.4f eV/Å, energy=%.4f eV, %d/%d atoms fixed",
        "converged" if converged else "NOT converged",
        fmax_final, energy_eV, n_fixed, len(relaxed_atoms),
    )

    warnings: list[str] = []
    if not converged:
        warnings.append(
            f"endpoint did not converge to fmax={fmax}; current={fmax_final:.4f}"
        )

    return EndpointReleaseResult(
        status="completed" if converged else "not_converged",
        converged=converged,
        fmax_final=fmax_final,
        energy_eV=energy_eV,
        n_atoms=len(relaxed_atoms),
        n_fixed_atoms=n_fixed,
        released_bonds=new_released,
        output_pdb=output_pdb,
        summary_json=summary_path,
        ledger=ledger.to_dict() if ledger is not None else None,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="endpoint_release",
        description=(
            "Release reactive-bond constraints from a scan endpoint and "
            "relax with the cluster boundary preserved. The released PDB "
            "becomes a clean NEB endpoint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Scan-output endpoint PDB (R-side or P-side)")
    p.add_argument("--out", "-o", required=True,
                   help="Output PDB path for the released endpoint")
    p.add_argument(
        "--release-bond", action="append", default=[],
        metavar="ATOMA,ATOMB",
        help=("Bond whose CV constraint is removed. Tokens accept "
              "'NAME.RESNAME', 'NAME.RESNAME.RESID', integer 1-based PDB "
              "serial, or '0:idx' for 0-based ASE index. Repeatable for "
              "multi-bond release. NOTE: ASE doesn't carry FixBondLengths "
              "through PDB I/O, so this flag is mostly for logging the "
              "before/after distances.")
    )
    p.add_argument(
        "--boundary-fix-preset", default=None,
        choices=["ca-only", "backbone", "backbone-water", "none"],
        help="Named boundary preset (same as the rest of qcb).")
    p.add_argument("--fix", nargs="+", default=None,
                   help="Extra constraint specs (see qcb top-level docs).")
    p.add_argument("--free", nargs="+", default=None,
                   help="Constraint specs to subtract from --fix.")
    p.add_argument("--model", default="mace-omol",
                   help="Calculator alias. Default mace-omol.")
    p.add_argument("--head", default=None,
                   help="Head for multi-head models (mace-mh).")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--charge-ledger", default=None,
                   help="Path to a ledger.yaml (validated and propagated).")
    p.add_argument("--charge", type=int, default=None,
                   help="Net charge (overrides PDB hint; ledger wins if given).")
    p.add_argument("--spin", type=int, default=None,
                   help="Spin multiplicity 2S+1 (default: 1 if no ledger).")
    p.add_argument("--fmax", type=float, default=0.02,
                   help="Convergence target (eV/Å). Tighter than scan defaults; "
                        "lower to 0.01 for very polished endpoints, raise to 0.05 "
                        "for noisy ML potentials.")
    p.add_argument("--max-steps", type=int, default=500,
                   help="Optimizer iteration cap.")
    p.add_argument("--optimizer", default="lbfgs",
                   choices=["lbfgs", "bfgs", "fire"])
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ledger = None
    if args.charge_ledger:
        from quantum_engine.ops.charge_ledger import load_ledger
        ledger = load_ledger(args.charge_ledger)
        log.info("loaded ledger from %s (total=%+d, spin=%d)",
                 ledger.source, ledger.total, ledger.spin)

    bonds: list[tuple[str, str]] = []
    for spec in args.release_bond:
        if "," not in spec:
            raise SystemExit(f"--release-bond {spec!r} must be 'A,B'")
        a, b = (s.strip() for s in spec.split(",", 1))
        bonds.append((a, b))

    result = release_endpoint(
        args.input,
        args.out,
        release_bonds=bonds,
        boundary_fix_preset=args.boundary_fix_preset,
        fix_specs=args.fix,
        free_specs=args.free,
        model=args.model,
        head=args.head,
        device=args.device,
        charge_ledger=ledger,
        cli_charge=args.charge,
        cli_spin=args.spin,
        fmax=args.fmax,
        max_steps=args.max_steps,
        optimizer=args.optimizer,
    )
    print(json.dumps({
        "status": result.status,
        "converged": result.converged,
        "fmax_final": result.fmax_final,
        "energy_eV": result.energy_eV,
        "output_pdb": str(result.output_pdb),
        "summary_json": str(result.summary_json),
        "released_bonds": [
            {"i": i, "j": j, "d_before": d_b, "d_after": d_a}
            for (i, j, d_b, d_a) in result.released_bonds
        ],
        "warnings": result.warnings,
    }, indent=2))
    return 0 if result.converged else 1


if __name__ == "__main__":
    sys.exit(main())
