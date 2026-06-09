"""Phase 6: ORCA QM-native engine — parser, input-gen, engine registry/routing.

No live ORCA (not in the container): the parser is tested with canned ORCA 4.x
output, input-gen by inspecting the written files, and the engine runner with
execute=False (prepare-only) or a monkeypatched run_orca.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from ase import Atoms

from quantum_engine.qm import orca
from quantum_engine.qm.engine import ENGINES, make_engine, register_engine, run_orca_engine
from quantum_engine.reaction_spec import ReactionSpec, ResolvedReaction, RunContext


CANNED_OUT = """
... preamble ...
FINAL SINGLE POINT ENERGY     -100.123456
... iterations ...
FINAL SINGLE POINT ENERGY     -100.234567

-----------------------
VIBRATIONAL FREQUENCIES
-----------------------

Scaling factor for frequencies =  1.000000000

   0:         0.00 cm**-1
   1:         0.00 cm**-1
   2:         0.00 cm**-1
   3:         0.00 cm**-1
   4:         0.00 cm**-1
   5:         0.00 cm**-1
   6:      -523.45 cm**-1
   7:       412.33 cm**-1
   8:       980.10 cm**-1

------------
NORMAL MODES
------------

---------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  O      0.000000    0.000000    0.000000
  C      1.160000    0.000000    0.000000
  O      2.320000    0.000000    0.000000

------
TIMING
"""


def _co2():
    return Atoms("OCO", positions=[[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]])


def _resolved():
    return ResolvedReaction(forming=[(0, 1)], breaking=[(1, 2)], reactive=[0, 1, 2],
                            cv_bond_difference=None, spec=ReactionSpec())


# ---- parser ----
def test_parse_final_energy_takes_last():
    assert orca.parse_final_energy(CANNED_OUT) == pytest.approx(-100.234567)
    assert orca.parse_final_energy("no energy here") is None


def test_parse_frequencies_and_imag():
    freqs = orca.parse_frequencies(CANNED_OUT)
    assert -523.45 in freqs and 412.33 in freqs
    assert orca.n_imaginary(freqs) == 1            # only -523.45 < -10
    assert orca.imaginary_frequency(freqs) == pytest.approx(-523.45)


def test_parse_last_geometry():
    geo = orca.parse_last_geometry(CANNED_OUT)
    assert geo is not None
    syms, pos = geo
    assert syms == ["O", "C", "O"]
    assert pos[1] == [1.16, 0.0, 0.0]


# ---- input generation ----
def test_write_nebts_input(tmp_path):
    r = tmp_path / "r.xyz"; p = tmp_path / "p.xyz"
    from ase.io import write as ase_write
    ase_write(str(r), _co2(), format="xyz")
    ase_write(str(p), _co2(), format="xyz")
    inp = orca.write_orca_nebts_input(r, p, tmp_path / "nebts.inp",
                                      charge=-1, multiplicity=1, n_images=6)
    text = inp.read_text()
    assert "NEB-TS" in text and "Freq" in text
    # ORCA 4.1.1 endpoint keyword is NEB_END_XYZFILE — NOT 'product'
    assert "%neb" in text and "NEB_END_XYZFILE" in text and "NImages 6" in text
    assert "product" not in text.lower().replace("nebts_product", "")
    assert "* xyz -1 1" in text
    assert (tmp_path / "nebts_product.xyz").exists()   # product endpoint written


def test_write_ts_freq_input(tmp_path):
    g = tmp_path / "g.xyz"
    from ase.io import write as ase_write
    ase_write(str(g), _co2(), format="xyz")
    inp = orca.write_orca_input(g, tmp_path / "optts.inp", job_type="ts+freq")
    assert "OptTS Freq" in inp.read_text()


# ---- engine registry ----
def test_engine_registry_has_orca():
    assert "orca" in ENGINES.names()
    assert make_engine("orca") is run_orca_engine
    with pytest.raises(ValueError, match="Unknown engine"):
        make_engine("gaussian")


def test_register_engine_extensibility():
    register_engine("dummy", lambda *a, **k: {"status": "ok"})
    try:
        assert make_engine("dummy")() == {"status": "ok"}
    finally:
        ENGINES.unregister("dummy")
    assert "dummy" not in ENGINES.names()


# ---- engine runner (no live ORCA) ----
def test_orca_engine_prepare_only_reactant_product(tmp_path):
    ctx = RunContext(engine="orca", model="b3lyp/def2-SVP", charge=-1, multiplicity=1)
    res = run_orca_engine(_resolved(), ctx, entry="reactant-product",
                          reactant=_co2(), product=_co2(), outdir=tmp_path,
                          execute=False)
    assert res["status"] == "prepared" and res["engine"] == "orca"
    from pathlib import Path
    assert Path(res["outputs"]["input"]).exists()
    assert (tmp_path / "gates.json").exists()


def test_orca_engine_reactant_only_errors(tmp_path):
    ctx = RunContext(engine="orca", model="b3lyp/def2-SVP")
    with pytest.raises(ValueError, match="needs both endpoints"):
        run_orca_engine(_resolved(), ctx, entry="reactant-only",
                        reactant=_co2(), outdir=tmp_path, execute=False)


def test_orca_engine_executes_and_gates(tmp_path, monkeypatch):
    """With a monkeypatched run_orca (n_imag=1), the engine reports converged."""
    def fake_run(inp, **kw):
        return {"returncode": 0, "out_path": str(tmp_path / "x.out"),
                "energy_eV": -2723.5, "frequencies_cm": [-500.0, 300.0],
                "n_imag": 1, "imag_freq_cm": -500.0,
                "geometry": (["O", "C", "O"],
                             [[-1.1, 0, 0], [0, 0, 0], [1.1, 0, 0]])}
    monkeypatch.setattr(orca, "run_orca", fake_run)
    ctx = RunContext(engine="orca", model="b3lyp/def2-SVP", charge=0, multiplicity=1)
    res = run_orca_engine(_resolved(), ctx, entry="ts-guess", ts_guess=_co2(),
                          outdir=tmp_path, execute=True)
    assert res["status"] == "converged" and res["n_imag"] == 1
    assert res["imag_freq_cm"] == -500.0
    assert isinstance(res["ts"], Atoms) and len(res["ts"]) == 3


def test_qm_method_basis_from_model_alias():
    from quantum_engine.qm.engine import _qm_method_basis
    ctx = RunContext(model="wB97X-D3/def2-TZVP")
    assert _qm_method_basis(ctx, None, None) == ("wB97X-D3", "def2-TZVP")
    # explicit args win
    assert _qm_method_basis(ctx, "PBE0", "def2-SVP") == ("PBE0", "def2-SVP")


# ---- ASE-ORCA calculator route (mode A) ----
def test_make_qm_calc_constructs_orca_ase_calc():
    from quantum_engine.qm.calc import make_qm_calc
    c = make_qm_calc("B3LYP", "def2-SVP", charge=-2, mult=3,
                     orca_bin="/net/software/orca/orca_4_1_1_linux_x86-64_openmpi313/orca")
    assert type(c).__name__ == "ORCA"
    assert c.parameters.get("charge") == -2 and c.parameters.get("mult") == 3
    assert "B3LYP" in c.parameters.get("orcasimpleinput")
    assert "engrad" in c.parameters.get("orcasimpleinput")
