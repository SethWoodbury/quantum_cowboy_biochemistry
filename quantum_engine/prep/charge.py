"""
Net formal-charge calculation for extracted active-site clusters.

Computes the total charge of an active-site model by summing standard
amino-acid formal charges (adjusted for pH-dependent protonation states
via PROPKA) and a user-supplied ligand charge.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

from quantum_engine.prep.protonate import get_pka_dict

# Standard formal charges of ionizable amino acids at neutral pH
# (default assumption before pKa correction)
DEFAULT_CHARGES: dict[str, int] = {
    "ASP": -1,   # deprotonated carboxylate
    "GLU": -1,   # deprotonated carboxylate
    "LYS": +1,   # protonated amine
    "ARG": +1,   # protonated guanidinium
    "HIS":  0,   # neutral (Nε-H tautomer)
    "CYS":  0,   # neutral thiol
    "TYR":  0,   # neutral phenol
}

# Charges when the protonation state is flipped from the default
FLIPPED_CHARGES: dict[str, int] = {
    "ASP":  0,   # protonated (neutral)
    "GLU":  0,   # protonated (neutral)
    "LYS":  0,   # deprotonated (neutral)
    "ARG":  0,   # deprotonated (neutral, very rare)
    "HIS": +1,   # protonated (imidazolium)
    "CYS": -1,   # deprotonated (thiolate)
    "TYR": -1,   # deprotonated (tyrosinate)
}


def _unique_residues(
    atom_array: AtomArray,
) -> list[tuple[str, int, str, str]]:
    """Return unique (chain_id, res_id, ins_code, res_name) tuples."""
    seen: set[tuple[str, int, str]] = set()
    residues: list[tuple[str, int, str, str]] = []
    for i in range(len(atom_array)):
        key = (
            atom_array.chain_id[i],
            int(atom_array.res_id[i]),
            atom_array.ins_code[i],
        )
        if key not in seen:
            seen.add(key)
            residues.append((
                atom_array.chain_id[i],
                int(atom_array.res_id[i]),
                atom_array.ins_code[i],
                atom_array.res_name[i].strip(),
            ))
    return residues


def _write_temp_pdb(atom_array: AtomArray) -> Path:
    """Write an AtomArray to a temporary PDB file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    pdb_file = PDBFile()
    pdb_file.set_structure(atom_array)
    pdb_file.write(tmp.name)
    tmp.close()
    return Path(tmp.name)


def calculate_net_charge(
    atom_array: AtomArray,
    ligand_charge: int = 0,
    pH: float = 7.0,
    pdb_path: str | Path | None = None,
    verbose: bool = False,
) -> int:
    """Calculate the formal net charge of an active-site cluster.

    The charge is the sum of:

    1. Formal charges of standard amino acids, adjusted for pH-dependent
       protonation states predicted by PROPKA.
    2. A user-supplied ligand charge.
    3. Terminal group charges (+1 for N-terminus, -1 for C-terminus) are
       **not** added because extracted clusters are typically capped and
       do not have free termini.

    If *pdb_path* is provided, PROPKA runs on that file (which should be
    the full protein, not the extracted cluster, for accurate pKa
    prediction).  Otherwise a temporary PDB is written from *atom_array*.

    Args:
        atom_array: Biotite AtomArray of the active-site cluster.
        ligand_charge: Formal charge of the ligand(s) / cofactor(s).
        pH: pH at which protonation states are evaluated.
        pdb_path: Optional path to the *full* protein PDB for more
            accurate PROPKA predictions.  Falls back to writing the
            cluster to a temporary PDB.
        verbose: If True, print per-residue charge contributions.

    Returns:
        Integer formal net charge of the cluster.
    """
    # Run PROPKA on the full protein (preferred) or the cluster
    if pdb_path is not None:
        pka_dict = get_pka_dict(pdb_path, pH=pH)
    else:
        tmp_pdb = _write_temp_pdb(atom_array)
        try:
            pka_dict = get_pka_dict(tmp_pdb, pH=pH)
        finally:
            tmp_pdb.unlink(missing_ok=True)

    # Enumerate unique residues in the cluster
    residues = _unique_residues(atom_array)

    total_charge = 0
    for chain_id, res_id, ins_code, res_name in residues:
        if res_name not in DEFAULT_CHARGES:
            continue

        default_q = DEFAULT_CHARGES[res_name]
        flipped_q = FLIPPED_CHARGES[res_name]

        # Look up PROPKA result
        pka_key = (chain_id, res_id)
        if pka_key in pka_dict:
            info = pka_dict[pka_key]
            pka_value = info["pKa"]

            if pka_value is not None:
                protonated = pka_value >= pH

                # Determine whether protonation matches the default
                # For acids (ASP, GLU, CYS, TYR): default = deprotonated
                #   protonated=True means flipped
                # For bases (LYS, ARG): default = protonated
                #   protonated=False means flipped
                # For HIS: default = neutral
                #   protonated=True means flipped (+1)
                if res_name in ("ASP", "GLU", "CYS", "TYR"):
                    charge = flipped_q if protonated else default_q
                elif res_name in ("LYS", "ARG"):
                    charge = default_q if protonated else flipped_q
                elif res_name == "HIS":
                    charge = flipped_q if protonated else default_q
                else:
                    charge = default_q
            else:
                charge = default_q
        else:
            # No PROPKA data -- use default
            charge = default_q

        if verbose:
            print(f"  {chain_id}:{res_name}{res_id} -> charge {charge:+d}")

        total_charge += charge

    total_charge += ligand_charge

    if verbose:
        print(f"  Ligand charge: {ligand_charge:+d}")
        print(f"  Total net charge: {total_charge:+d}")

    return total_charge
