"""Semiempirical (xTB / g-xTB) energy functions as ASE calculators.

This is the "traditional-QC" plug for the energy-function axis: it exposes
GFN0/1/2-xTB, GFN-FF, and g-xTB through the SAME ``ase.Calculator`` interface the
MLFFs use, so any optimizer / saddle / path method drives them identically.

Resolution order per call:
  1. ``xtb-python`` (``xtb.ase.calculator.XTB``) if importable — clean native
     ASE calc with charge/uhf support; preferred.
  2. subprocess to the vendored binary (``site.XTB_BIN`` for GFN levels /
     ``site.GXTB_BIN`` for g-xTB) writing an xyz, running ``--grad``, and parsing
     the Turbomole-format ``energy`` + ``gradient`` files.

Charge-awareness caveat (enforced): **GFN2-xTB is not charge-state-aware** for
metals — :func:`make_qc_calc` warns (default) or raises (``forbid_metal_gfn2``)
when a metal element is present, so the di-Zn site isn't silently mis-modeled.
Use a charge-aware MLFF (mace-omol / mace-polar / uma) for metal/charged TSs;
xTB is for organic substrates / cheap sanity checks.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from quantum_engine.logging_utils import get_logger

log = get_logger("calc.qc")

# eV / Bohr conversions (CODATA)
_HARTREE_EV = 27.211386245988
_BOHR_A = 0.529177210903

# method alias -> (binary key, extra flags)
_METHODS = {
    "gfn0-xtb": ("xtb", ["--gfn", "0"]),
    "gfn1-xtb": ("xtb", ["--gfn", "1"]),
    "gfn2-xtb": ("xtb", ["--gfn", "2"]),
    "gfn-ff":   ("xtb", ["--gfnff"]),
    "gfnff":    ("xtb", ["--gfnff"]),
    "g-xtb":    ("gxtb", []),
    "xtb":      ("xtb", ["--gfn", "2"]),   # bare 'xtb' == GFN2
}

# metals where GFN2's lack of charge/spin awareness is dangerous
_METALS = frozenset({
    "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
})


def _binary(key: str):
    from quantum_engine import site  # noqa: PLC0415
    b = getattr(site, "GXTB_BIN", None) if key == "gxtb" else getattr(site, "XTB_BIN", None)
    return b


def _lib_dirs(key: str):
    from quantum_engine import site  # noqa: PLC0415
    return getattr(site, "GXTB_LIB_DIRS", []) if key == "gxtb" else getattr(site, "XTB_LIB_DIRS", [])


def make_qc_calc(method: str, charge: int = 0, spin: int = 1, *,
                 atoms=None, forbid_metal_gfn2: bool = False, **kwargs):
    """Build an ASE calculator for an xTB-family ``method``.

    spin is the multiplicity (2S+1); xTB's ``--uhf`` takes the number of unpaired
    electrons = spin - 1. Warns/forbids GFN2 + metal (see module docstring).
    """
    m = method.lower()
    if m not in _METHODS:
        raise ValueError(f"unknown semiempirical method {method!r}; "
                         f"valid: {sorted(_METHODS)}")
    if m.startswith("gfn2") or m == "xtb":
        elems = set()
        if atoms is not None:
            elems = set(getattr(atoms, "get_chemical_symbols", lambda: [])())
        metals = elems & _METALS
        if metals:
            msg = (f"GFN2-xTB is not charge/spin-state-aware for metals "
                   f"({sorted(metals)}); results for the metal site are unreliable. "
                   f"Use a charge-aware MLFF (mace-omol/mace-polar/uma) for "
                   f"metal/charged transition states; xTB is for organic sanity only.")
            if forbid_metal_gfn2:
                raise ValueError(msg)
            log.warning(msg)
        elif atoms is None:
            log.debug("GFN2-xTB selected; remember it is not charge-state-aware "
                      "for metals (cannot check without an atoms object here).")

    # Prefer xtb-python's native ASE calculator if available (GFN levels only).
    bin_key, flags = _METHODS[m]
    if bin_key == "xtb":
        try:
            from xtb.ase.calculator import XTB  # noqa: PLC0415
        except (ImportError, ModuleNotFoundError) as exc:
            log.debug("xtb-python unavailable (%s); using subprocess xtb", exc)
        else:
            # xtb-python method names: 'GFN2-xTB' / 'GFN1-xTB' / 'GFN0-xTB' / 'GFN-FF'
            method_name = ("GFN-FF" if "--gfnff" in flags
                           else f"GFN{flags[flags.index('--gfn') + 1]}-xTB")
            return XTB(method=method_name, **kwargs)  # constructor errors propagate

    return XTBProcessCalculator(method=m, charge=charge, spin=spin, **kwargs)


try:
    from ase.calculators.calculator import Calculator, all_changes
    _ASE_OK = True
except Exception:  # noqa: BLE001 — ASE not importable (pure-routing tests)
    Calculator = object       # type: ignore[assignment,misc]
    all_changes = []          # type: ignore[assignment]
    _ASE_OK = False


class XTBProcessCalculator(Calculator):  # type: ignore[misc]
    """ASE calculator that shells out to the xTB / g-xTB binary per evaluation.

    Parses the Turbomole-format ``energy`` + ``gradient`` files written by
    ``xtb ... --grad``. Energies returned in eV, forces in eV/Å.
    """
    implemented_properties = ["energy", "forces"]

    def __init__(self, method: str = "gfn2-xtb", charge: int = 0, spin: int = 1,
                 accuracy: float = 1.0, **kwargs):
        if not _ASE_OK:
            raise ImportError("ASE is required for XTBProcessCalculator")
        super().__init__(**kwargs)
        self.method = method.lower()
        self.charge = int(charge)
        self.uhf = max(0, int(spin) - 1)
        self.accuracy = accuracy
        self._bin_key, self._flags = _METHODS[self.method]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        binary = _binary(self._bin_key)
        if not binary or not os.path.isfile(binary):
            raise FileNotFoundError(
                f"{self._bin_key} binary not found for method {self.method!r} "
                f"(checked site.{'GXTB_BIN' if self._bin_key == 'gxtb' else 'XTB_BIN'}={binary!r})")
        env = dict(os.environ)
        libs = [d for d in (_lib_dirs(self._bin_key) or []) if d]
        if libs:
            env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
        # per-call charge/spin via xtb's .CHRG/.UHF or flags
        with tempfile.TemporaryDirectory() as td:
            from ase.io import write as ase_write  # noqa: PLC0415
            xyz = os.path.join(td, "mol.xyz")
            ase_write(xyz, a)
            cmd = [binary, "mol.xyz", *self._flags, "--grad",
                   "--chrg", str(self.charge), "--uhf", str(self.uhf), "--acc",
                   str(self.accuracy)]
            proc = subprocess.run(cmd, cwd=td, env=env, capture_output=True, text=True)
            e, grad = self._parse(td)
            if e is None:
                raise RuntimeError(
                    f"xtb did not produce an energy (rc={proc.returncode}). "
                    f"stderr tail:\n{proc.stderr[-800:]}")
        self.results["energy"] = e * _HARTREE_EV
        if grad is not None:
            self.results["forces"] = -grad * (_HARTREE_EV / _BOHR_A)

    @staticmethod
    def _parse(td):
        """Return (energy_hartree, gradient[N,3] in Hartree/Bohr) from xtb output."""
        e = None
        epath = os.path.join(td, "energy")
        if os.path.isfile(epath):
            lines = [ln for ln in open(epath).read().splitlines() if ln.strip()
                     and not ln.startswith("$")]
            if lines:
                # Turbomole $energy data line: "<cycle> <Etot> [<E1e> <E2e>]".
                # Take the total-energy field (index 1), or index 0 if single-column.
                fields = lines[-1].split()
                tok = fields[1] if len(fields) >= 2 else fields[0]
                e = float(tok.replace("D", "E"))
        grad = None
        gpath = os.path.join(td, "gradient")
        if os.path.isfile(gpath):
            txt = open(gpath).read().splitlines()
            vecs = []
            for ln in txt:
                parts = ln.split()
                # gradient component lines: 3 floats (may use 'D' exponent)
                if len(parts) == 3:
                    try:
                        vecs.append([float(p.replace("D", "E")) for p in parts])
                    except ValueError:
                        continue
            if vecs:
                grad = np.array(vecs, dtype=float)
        return e, grad


__all__ = ["make_qc_calc", "XTBProcessCalculator"]
