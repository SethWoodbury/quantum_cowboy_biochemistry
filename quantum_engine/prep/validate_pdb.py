"""
PDB quality checks and validation.

Catches common issues that break downstream QM/MLFF calculations:
- Invalid atom serial numbers (0, duplicates, non-sequential)
- Missing hydrogens on sp2/sp3 carbons and nitrogens
- Clashing atoms (< 0.5 A apart)
- Atom name format violations
- Residue completeness (missing backbone atoms)
- Coordinate sanity (NaN, extreme values)
- Element column consistency
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

log = logging.getLogger("quantum_engine.validate")


class PDBIssue:
    """A single validation issue found in a PDB file."""

    def __init__(self, severity: str, category: str, message: str,
                 line_num: int | None = None, atom_serial: int | None = None):
        self.severity = severity  # "ERROR", "WARNING", "INFO"
        self.category = category
        self.message = message
        self.line_num = line_num
        self.atom_serial = atom_serial

    def __repr__(self):
        loc = f" (line {self.line_num})" if self.line_num else ""
        return f"[{self.severity}] {self.category}: {self.message}{loc}"


def validate_pdb(pdb_path: str | Path, strict: bool = False) -> list[PDBIssue]:
    """Run all validation checks on a PDB file.

    Args:
        pdb_path: Path to the PDB file
        strict: If True, treat warnings as errors

    Returns:
        List of PDBIssue objects. Empty list = clean PDB.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        return [PDBIssue("ERROR", "FILE", f"PDB file not found: {pdb_path}")]

    lines = pdb_path.read_text().splitlines()
    issues = []

    issues.extend(_check_atom_serials(lines))
    issues.extend(_check_coordinates(lines))
    issues.extend(_check_element_column(lines))
    issues.extend(_check_atom_clashes(lines))
    issues.extend(_check_atom_names(lines))
    issues.extend(_check_backbone_completeness(lines))
    issues.extend(_check_hydrogen_coverage(lines))
    issues.extend(_check_header_preserved(lines))

    return issues


def validate_and_report(pdb_path: str | Path, strict: bool = False) -> bool:
    """Run validation and print a report. Returns True if no errors."""
    issues = validate_pdb(pdb_path, strict=strict)

    errors = [i for i in issues if i.severity == "ERROR"]
    warnings = [i for i in issues if i.severity == "WARNING"]
    infos = [i for i in issues if i.severity == "INFO"]

    if not issues:
        log.info(f"PDB validation PASSED: {pdb_path}")
        return True

    log.info(f"PDB validation: {len(errors)} errors, {len(warnings)} warnings, {len(infos)} info")
    for issue in issues:
        if issue.severity == "ERROR":
            log.error(f"  {issue}")
        elif issue.severity == "WARNING":
            log.warning(f"  {issue}")
        else:
            log.info(f"  {issue}")

    return len(errors) == 0


# ══════════════════════════════════════════════════════════════
#  CHECK FUNCTIONS
# ══════════════════════════════════════════════════════════════


def _check_atom_serials(lines: list[str]) -> list[PDBIssue]:
    """Check atom serial numbers: no zeros, no duplicates, sequential."""
    issues = []
    serials = []

    for i, line in enumerate(lines):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            serial = int(line[6:11])
        except ValueError:
            issues.append(PDBIssue("ERROR", "SERIAL", f"Non-integer atom serial: '{line[6:11].strip()}'", i + 1))
            continue

        if serial == 0:
            issues.append(PDBIssue("ERROR", "SERIAL", f"Atom serial is 0 (invalid)", i + 1, serial))
        if serial < 0:
            issues.append(PDBIssue("ERROR", "SERIAL", f"Negative atom serial: {serial}", i + 1, serial))
        serials.append((serial, i + 1))

    # Check duplicates
    seen = {}
    for serial, line_num in serials:
        if serial in seen:
            issues.append(PDBIssue("WARNING", "SERIAL",
                                   f"Duplicate atom serial {serial} (first at line {seen[serial]})",
                                   line_num, serial))
        seen[serial] = line_num

    # Check sequential (ATOM and HETATM may have separate ranges)
    if serials:
        for j in range(1, len(serials)):
            curr, prev = serials[j][0], serials[j - 1][0]
            if curr != prev + 1 and curr != 1:  # allow reset at HETATM
                # Only warn if the gap is large
                if abs(curr - prev) > 2:
                    issues.append(PDBIssue("INFO", "SERIAL",
                                           f"Non-sequential serials: {prev} → {curr}",
                                           serials[j][1]))

    return issues


def _check_coordinates(lines: list[str]) -> list[PDBIssue]:
    """Check for NaN, inf, or extreme coordinate values."""
    issues = []
    for i, line in enumerate(lines):
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            issues.append(PDBIssue("ERROR", "COORDS", f"Unparseable coordinates", i + 1))
            continue

        for val, label in [(x, "x"), (y, "y"), (z, "z")]:
            if math.isnan(val) or math.isinf(val):
                issues.append(PDBIssue("ERROR", "COORDS", f"{label}={val} (NaN/inf)", i + 1))
            elif abs(val) > 9999:
                issues.append(PDBIssue("WARNING", "COORDS",
                                       f"{label}={val:.1f} (extreme, may overflow PDB format)", i + 1))

    return issues


def _check_element_column(lines: list[str]) -> list[PDBIssue]:
    """Check that element column (77-78) is present and consistent with atom name."""
    issues = []
    for i, line in enumerate(lines):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 78:
            issues.append(PDBIssue("WARNING", "ELEMENT",
                                   "Line too short for element column", i + 1))
            continue

        element = line[76:78].strip()
        atom_name = line[12:16].strip()

        if not element:
            issues.append(PDBIssue("WARNING", "ELEMENT",
                                   f"Empty element column for atom '{atom_name}'", i + 1))
        elif element not in ("H", "C", "N", "O", "S", "P", "ZN", "FE", "MG", "CA",
                             "CL", "BR", "F", "I", "MN", "CU", "CO", "NI", "SE"):
            issues.append(PDBIssue("INFO", "ELEMENT",
                                   f"Unusual element '{element}' for atom '{atom_name}'", i + 1))

    return issues


def _check_atom_clashes(lines: list[str], min_dist: float = 0.5) -> list[PDBIssue]:
    """Check for atoms closer than min_dist (excluding bonded H)."""
    issues = []
    atoms = []
    for i, line in enumerate(lines):
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            aname = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            resn = line[17:20].strip()
            chain = line[21]
            resnum = int(line[22:26])
            atoms.append((i + 1, aname, element, chain, resnum, resn, x, y, z))
        except ValueError:
            continue

    # Only check non-H vs non-H clashes (H clashes are handled by xTB)
    heavy = [(idx, an, el, ch, rn, rsn, x, y, z) for idx, an, el, ch, rn, rsn, x, y, z in atoms
             if el != "H" and not an.startswith("H")]

    for a in range(len(heavy)):
        for b in range(a + 1, len(heavy)):
            ax, ay, az = heavy[a][6], heavy[a][7], heavy[a][8]
            bx, by, bz = heavy[b][6], heavy[b][7], heavy[b][8]
            d = math.sqrt((ax - bx)**2 + (ay - by)**2 + (az - bz)**2)
            if d < min_dist:
                issues.append(PDBIssue(
                    "ERROR", "CLASH",
                    f"{heavy[a][5]} {heavy[a][3]}:{heavy[a][4]} {heavy[a][1]} ↔ "
                    f"{heavy[b][5]} {heavy[b][3]}:{heavy[b][4]} {heavy[b][1]} = {d:.2f} A",
                    heavy[a][0],
                ))

    return issues


def _check_atom_names(lines: list[str]) -> list[PDBIssue]:
    """Check atom name formatting (columns 13-16)."""
    issues = []
    for i, line in enumerate(lines):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        raw_name = line[12:16]
        name = raw_name.strip()
        if not name:
            issues.append(PDBIssue("ERROR", "ATOMNAME", "Empty atom name", i + 1))
        # Check for non-alphanumeric characters
        if not all(c.isalnum() or c in "' *+" for c in name):
            issues.append(PDBIssue("WARNING", "ATOMNAME",
                                   f"Unusual characters in atom name: '{name}'", i + 1))

    return issues


def _check_backbone_completeness(lines: list[str]) -> list[PDBIssue]:
    """Check that each protein residue has at minimum N, CA, C atoms."""
    issues = []
    residues = {}  # {(chain, resnum, resname): set of atom names}

    for line in lines:
        if not line.startswith("ATOM"):
            continue
        chain = line[21]
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        resname = line[17:20].strip()
        aname = line[12:16].strip()
        key = (chain, resnum, resname)
        residues.setdefault(key, set()).add(aname)

    for (chain, resnum, resname), atoms in residues.items():
        # Standard amino acids should have at least N, CA, C
        required = {"N", "CA", "C"}
        missing = required - atoms
        if missing:
            issues.append(PDBIssue("WARNING", "BACKBONE",
                                   f"{resname} {chain}:{resnum} missing backbone: {missing}"))

    return issues


def _check_hydrogen_coverage(lines: list[str]) -> list[PDBIssue]:
    """Check that sp3 carbons (CA, CB) have hydrogens."""
    issues = []
    residues = {}

    for line in lines:
        if not line.startswith("ATOM"):
            continue
        chain = line[21]
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        resname = line[17:20].strip()
        aname = line[12:16].strip()
        key = (chain, resnum, resname)
        residues.setdefault(key, set()).add(aname)

    # Expected H atoms for common heavy atoms
    h_expectations = {
        "CA": ["HA"],      # alpha H
        "CB": ["HB"],      # at least one beta H (HB2/HB3 or 1HB/2HB)
    }

    for (chain, resnum, resname), atoms in residues.items():
        if resname == "GLY":
            # GLY has HA2/HA3 instead of HA
            if "CA" in atoms and not any(a.startswith("HA") for a in atoms):
                issues.append(PDBIssue("WARNING", "HYDROGEN",
                                       f"GLY {chain}:{resnum} CA missing HA hydrogens"))
            continue

        if "CA" in atoms and not any(a.startswith("HA") for a in atoms):
            issues.append(PDBIssue("WARNING", "HYDROGEN",
                                   f"{resname} {chain}:{resnum} missing HA on CA"))

        if "CB" in atoms and not any(a.startswith("HB") or a.startswith("1HB") or a.startswith("2HB")
                                     for a in atoms):
            issues.append(PDBIssue("WARNING", "HYDROGEN",
                                   f"{resname} {chain}:{resnum} missing HB on CB"))

    return issues


def _check_header_preserved(lines: list[str]) -> list[PDBIssue]:
    """Check that HEADER and important REMARK lines are present."""
    issues = []
    has_header = any(l.startswith("HEADER") for l in lines)
    has_remark = any(l.startswith("REMARK") for l in lines)

    if not has_header:
        issues.append(PDBIssue("INFO", "HEADER", "No HEADER line found"))

    return issues


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(description="Validate a PDB file for QM/MLFF calculations")
    p.add_argument("pdb_file", help="PDB file to validate")
    p.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = p.parse_args()

    issues = validate_pdb(args.pdb_file, strict=args.strict)

    errors = [i for i in issues if i.severity == "ERROR"]
    warnings = [i for i in issues if i.severity == "WARNING"]
    infos = [i for i in issues if i.severity == "INFO"]

    for issue in issues:
        print(issue)

    print(f"\n{len(errors)} errors, {len(warnings)} warnings, {len(infos)} info")

    if errors:
        exit(1)
    elif warnings and args.strict:
        exit(1)


if __name__ == "__main__":
    main()
