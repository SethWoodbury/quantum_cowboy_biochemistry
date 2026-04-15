"""
Protonation state assignment and pKa prediction via PROPKA.

Uses PROPKA3 to predict pKa values for all ionizable residues in a PDB
structure, then determines protonation states at a given pH.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# Residues whose protonation state is pH-sensitive
IONIZABLE_RESIDUES = {"ASP", "GLU", "HIS", "LYS", "CYS", "ARG", "TYR"}

# Standard model pKa values (Stryer / biochemistry textbooks)
MODEL_PKA: dict[str, float] = {
    "ASP": 3.65,
    "GLU": 4.25,
    "HIS": 6.00,
    "CYS": 8.18,
    "TYR": 10.07,
    "LYS": 10.54,
    "ARG": 12.48,
}


def get_pka_dict(
    pdb_path: str | Path,
    pH: float = 7.0,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Calculate pKa for every ionizable group in a PDB file using PROPKA.

    Args:
        pdb_path: Path to a PDB file.
        pH: Reference pH for determining protonation states.

    Returns:
        Dictionary keyed by ``(chain_id, res_num)`` with values::

            {
                'pKa': float,        # PROPKA-predicted pKa
                'resn': str,         # 3-letter residue name
                'protonated': bool,  # True when pKa >= pH (acid stays protonated)
            }

        For acid residues (ASP, GLU, CYS, TYR), ``protonated`` is True when
        pKa >= pH (the side-chain remains neutral / protonated).  For base
        residues (HIS, LYS, ARG), ``protonated`` is True when pKa >= pH (the
        side-chain carries a positive charge).
    """
    import propka.input
    import propka.lib
    import propka.molecular_container
    import propka.parameters

    in_file = Path(pdb_path).resolve()
    if not in_file.exists():
        raise FileNotFoundError(f"PDB file not found: {in_file}")

    propka_opts = propka.lib.loadOptions([str(in_file)])
    propka_params = propka.input.read_parameter_file(
        propka_opts.parameters, propka.parameters.Parameters()
    )
    propka_molc = propka.molecular_container.MolecularContainer(
        propka_params, propka_opts
    )
    propka.input.read_molecule_file(propka_opts.input_pdb, propka_molc)
    propka_molc.calculate_pka()

    resis_dict: dict[tuple[str, int], dict[str, Any]] = {}
    for group in propka_molc.conformations["AVR"].groups:
        pka_value = group.pka_value
        resis_dict[(group.atom.chain_id, group.atom.res_num)] = {
            "pKa": pka_value,
            "resn": group.atom.res_name,
            "protonated": (pka_value >= pH) if pka_value is not None else None,
        }

    return resis_dict


def assign_protonation_states(
    pdb_path: str | Path,
    pH: float = 7.0,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Identify residues whose protonation state differs from the default at *pH*.

    This focuses on the five residues most commonly affected by pH in enzyme
    active sites: HIS, ASP, GLU, LYS, and CYS.

    At neutral pH the "default" assumptions are:

    * ASP / GLU  -- deprotonated (negative charge)
    * LYS        -- protonated   (positive charge)
    * HIS        -- neutral (Nε protonated by convention)
    * CYS        -- neutral thiol

    Any residue whose PROPKA pKa shifts it *away* from these defaults at the
    requested pH is returned.

    Args:
        pdb_path: Path to a PDB file.
        pH: Target pH for protonation-state evaluation.

    Returns:
        Dictionary keyed by ``(chain_id, res_num)`` with values::

            {
                'pKa': float,
                'resn': str,
                'protonated': bool,
                'note': str,   # human-readable description of the change
            }
    """
    target_residues = {"HIS", "ASP", "GLU", "LYS", "CYS"}
    pka_dict = get_pka_dict(pdb_path, pH=pH)

    changes: dict[tuple[str, int], dict[str, Any]] = {}

    for key, info in pka_dict.items():
        resn = info["resn"].strip()
        if resn not in target_residues:
            continue

        pka = info["pKa"]
        if pka is None:
            continue

        protonated = pka >= pH

        # Determine whether this deviates from the "standard" pH 7 expectation
        note: str | None = None
        if resn in ("ASP", "GLU"):
            # Normally deprotonated (pKa ~ 3-4). Flag if pKa shifted above pH.
            if protonated:
                note = f"{resn} pKa={pka:.2f} >= pH={pH:.1f}: stays protonated (neutral)"
        elif resn == "LYS":
            # Normally protonated (pKa ~ 10.5). Flag if pKa shifted below pH.
            if not protonated:
                note = f"LYS pKa={pka:.2f} < pH={pH:.1f}: deprotonated (neutral)"
        elif resn == "HIS":
            # Normally neutral at pH 7 (pKa ~ 6). Flag if pKa shifted above pH.
            if protonated:
                note = f"HIS pKa={pka:.2f} >= pH={pH:.1f}: protonated (+1 charge)"
        elif resn == "CYS":
            # Normally protonated thiol (pKa ~ 8). Flag if pKa shifted below pH.
            if not protonated:
                note = f"CYS pKa={pka:.2f} < pH={pH:.1f}: deprotonated (thiolate, -1)"

        if note is not None:
            changes[key] = {
                "pKa": pka,
                "resn": resn,
                "protonated": protonated,
                "note": note,
            }

    return changes
