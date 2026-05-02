"""ChimeraX-based protonation: empirical H placement + Gasteiger partial charges.

ChimeraX's `addh` command does atom-type-aware hydrogen addition, including:
- pH-aware ASP/GLU/HIS protonation states (by default neutral pH)
- Sidechain rotamer optimization for H placement
- Geometric clash avoidance

Combined with `addcharge method gasteiger` we get partial charges in PQR
format that downstream code can read.

This module is one OPTION in the consensus protonation system
(see `qcb.prep.protonate_consensus`). It runs in parallel with propka,
pdbfixer, and the hardcoded rules; the consensus arbiter combines
them into a single best protonation state.

Why use ChimeraX specifically:
- Better H placement geometry than reduce/pdbfixer for unusual residues
- Captures aromatic / metal-coordination contexts via atom typing
- Consistent with what the broader structural-bio community uses

Why NOT use ChimeraX exclusively:
- Defaults assume canonical residues — can mis-protonate KCX, covalent
  intermediates, or unusual ligands
- Doesn't compute pKa (uses fixed model values internally)
- Slower than propka or pdbfixer (~20-60s startup per invocation)
- Requires the ChimeraX kernel to be installed

Hence: combine with propka (true pKa) + hardcoded rules (mechanism-specific).
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("quantum_engine.prep.protonate_chimera")


@dataclass
class ChimeraResult:
    """Result of a ChimeraX protonation run."""
    success: bool
    protonated_pdb: Path | None = None        # path to the H-added PDB
    pqr_path: Path | None = None              # path to PQR (charges + radii)
    partial_charges: dict[int, float] = field(default_factory=dict)  # serial → charge
    n_h_added: int = 0
    log: str = ""


def _resolve_kernel(kernel: str | None = None) -> str | None:
    """Resolve ChimeraX kernel path: argument > qcb.config > None."""
    if kernel and os.path.isfile(kernel):
        return kernel
    try:
        from quantum_engine.config import CHIMERA_KERNEL
        if CHIMERA_KERNEL and os.path.isfile(CHIMERA_KERNEL):
            return CHIMERA_KERNEL
    except Exception:
        pass
    return None


def add_hydrogens_chimera(
    input_pdb: str | Path,
    output_pdb: str | Path | None = None,
    kernel: str | None = None,
    method: str = "addh",                     # "addh" or "addh hbond true"
    timeout_s: int = 120,
) -> ChimeraResult:
    """Add hydrogens to a PDB structure using ChimeraX.

    Args:
        input_pdb: input structure path
        output_pdb: where to write the protonated PDB. If None, a temp file is used.
        kernel: path to ChimeraX executable (default: auto-detect via qcb.config)
        method: 'addh' (default) or 'addh hbond true' (uses H-bond optimization)
        timeout_s: kill ChimeraX if it takes longer than this

    Returns:
        ChimeraResult with the protonated PDB path and metadata.
    """
    input_pdb = Path(input_pdb).resolve()
    if output_pdb is None:
        output_pdb = input_pdb.with_suffix(".chimera.pdb")
    else:
        output_pdb = Path(output_pdb).resolve()
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    chimera = _resolve_kernel(kernel)
    if chimera is None:
        log.warning("ChimeraX kernel not found; skipping ChimeraX protonation")
        return ChimeraResult(success=False, log="kernel not found")

    log.info(f"  ChimeraX addh: {input_pdb.name} → {output_pdb.name}")

    cmd_str = "; ".join([
        f"open {input_pdb}",
        method,
        f"save {output_pdb}",
        "close all",
        "exit",
    ])
    cmd = [chimera, "--nogui", "--silent", "--cmd", cmd_str]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        log.error(f"  ChimeraX timed out after {timeout_s}s")
        return ChimeraResult(success=False, log=f"timeout after {timeout_s}s")
    except FileNotFoundError as e:
        log.error(f"  ChimeraX kernel not executable: {e}")
        return ChimeraResult(success=False, log=str(e))

    # Always log return code and any stderr — silent failures are the worst
    combined_log = (
        f"returncode={proc.returncode}\n"
        f"--- stdout (last 1500 chars) ---\n{(proc.stdout or '')[-1500:]}\n"
        f"--- stderr (last 1500 chars) ---\n{(proc.stderr or '')[-1500:]}"
    )

    if proc.returncode != 0:
        log.error(f"  ChimeraX returned {proc.returncode}")
        log.error(f"  stderr (last 500 chars): {(proc.stderr or '').strip()[-500:]}")
        log.error(f"  stdout (last 500 chars): {(proc.stdout or '').strip()[-500:]}")

    if not output_pdb.is_file() or output_pdb.stat().st_size == 0:
        log.error(f"  ChimeraX did not produce {output_pdb}")
        log.error(f"  stderr (last 500 chars): {(proc.stderr or '').strip()[-500:]}")
        log.error(f"  stdout (last 500 chars): {(proc.stdout or '').strip()[-500:]}")
        return ChimeraResult(success=False, log=combined_log)

    # Count H atoms added (rough heuristic)
    n_h_in = _count_h_atoms(input_pdb)
    n_h_out = _count_h_atoms(output_pdb)
    n_h_added = n_h_out - n_h_in
    log.info(f"  ChimeraX added {n_h_added} hydrogens ({n_h_in} → {n_h_out})")

    return ChimeraResult(
        success=True,
        protonated_pdb=output_pdb,
        n_h_added=n_h_added,
        log=combined_log,
    )


def add_hydrogens_with_charges(
    input_pdb: str | Path,
    output_pdb: str | Path | None = None,
    pqr_path: str | Path | None = None,
    charge_method: str = "gasteiger",         # gasteiger | am1-bcc | mmff (require ChimeraX charge plugins)
    kernel: str | None = None,
    ligand_charges: dict[str, int] | None = None,
    timeout_s: int = 180,
) -> ChimeraResult:
    """Add hydrogens AND compute partial charges in one ChimeraX session.

    Outputs both a protonated PDB and a PQR file (atomic coords + charge + radius).

    Args:
        input_pdb: input structure
        output_pdb: H-added PDB path (default: <stem>.chimera.pdb)
        pqr_path: PQR output path (default: <stem>.chimera.pqr)
        charge_method: ChimeraX `addcharge method` argument
        ligand_charges: optional {residue_name: net_charge} for non-standard ligands.
                       Passed as `setattr` commands before charge calculation.
        kernel: ChimeraX path
        timeout_s: command timeout
    """
    input_pdb = Path(input_pdb).resolve()
    if output_pdb is None:
        output_pdb = input_pdb.with_suffix(".chimera.pdb")
    else:
        output_pdb = Path(output_pdb).resolve()
    if pqr_path is None:
        pqr_path = output_pdb.with_suffix(".pqr")
    else:
        pqr_path = Path(pqr_path).resolve()

    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    chimera = _resolve_kernel(kernel)
    if chimera is None:
        log.warning("ChimeraX kernel not found; skipping addh+charges")
        return ChimeraResult(success=False, log="kernel not found")

    cmds = [f"open {input_pdb}", "addh"]

    # Set custom ligand charges before addcharge (so it knows ligand totals)
    if ligand_charges:
        for resname, charge in ligand_charges.items():
            cmds.append(f"setattr r netCharge {charge} :{resname}")

    cmds.extend([
        f"save {output_pdb}",
        f"addcharge method {charge_method}",
        f"save {pqr_path} format pdb pqr true",
        "close all",
        "exit",
    ])

    cmd_str = "; ".join(cmds)
    log.info(f"  ChimeraX addh+addcharge ({charge_method}): {input_pdb.name}")

    try:
        proc = subprocess.run(
            [chimera, "--nogui", "--silent", "--cmd", cmd_str],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        log.error(f"  ChimeraX timed out after {timeout_s}s")
        return ChimeraResult(success=False, log=f"timeout after {timeout_s}s")
    except FileNotFoundError as e:
        log.error(f"  ChimeraX kernel not executable: {e}")
        return ChimeraResult(success=False, log=str(e))

    combined_log = (
        f"returncode={proc.returncode}\n"
        f"--- stdout (last 1500 chars) ---\n{(proc.stdout or '')[-1500:]}\n"
        f"--- stderr (last 1500 chars) ---\n{(proc.stderr or '')[-1500:]}"
    )

    if proc.returncode != 0:
        log.error(f"  ChimeraX returned {proc.returncode}")
        log.error(f"  stderr (last 500 chars): {(proc.stderr or '').strip()[-500:]}")
        log.error(f"  stdout (last 500 chars): {(proc.stdout or '').strip()[-500:]}")

    if not output_pdb.is_file():
        log.error(f"  ChimeraX did not produce {output_pdb}")
        log.error(f"  stderr (last 500 chars): {(proc.stderr or '').strip()[-500:]}")
        log.error(f"  stdout (last 500 chars): {(proc.stdout or '').strip()[-500:]}")
        return ChimeraResult(success=False, log=combined_log)

    partial_charges = parse_pqr_charges(pqr_path) if pqr_path.is_file() else {}
    n_h_added = _count_h_atoms(output_pdb) - _count_h_atoms(input_pdb)

    log.info(f"  ChimeraX done: {n_h_added} H added, {len(partial_charges)} partial charges")
    return ChimeraResult(
        success=True,
        protonated_pdb=output_pdb,
        pqr_path=pqr_path if pqr_path.is_file() else None,
        partial_charges=partial_charges,
        n_h_added=n_h_added,
        log=combined_log,
    )


def parse_pqr_charges(pqr_path: str | Path) -> dict[int, float]:
    """Parse a PQR file (PDB + charge + radius). Returns {atom_serial: charge}."""
    out: dict[int, float] = {}
    with open(pqr_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                serial = int(line[6:11])
                charge = float(line[54:62].strip())
            except ValueError:
                continue
            out[serial] = charge
    return out


def _count_h_atoms(pdb_path: str | Path) -> int:
    """Count hydrogen atoms in a PDB file."""
    n = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                element = line[76:78].strip() if len(line) >= 78 else line[12:16].strip()[0]
                if element == "H" or element == "D":
                    n += 1
    return n
