"""Known-outcome regression benchmarks (xTB, in-container; slow-marked).

These exercise the WHOLE pipeline end-to-end on small reactions with known
outcomes — qc_calc (xTB as an ASE calc) → saddle cascade → partial Hessian →
the n_imag/imag-freq/overlap gate → the ts_entry orchestrator. They are the
end-to-end layer above the fast unit tests.

xTB (GFN2) is qualitative — frequencies/barriers are method-soft; the
quantitative layer is the user's charge-aware-MLFF (mace-polar / uma) GPU runs.
So HCN asserts the strong, well-separated isomerization mode tightly, while the
charged SN2 case asserts the pipeline MECHANICS (a first-order saddle is found
with the imaginary mode on the reactive atoms, and the −1 charge propagates).

Run: `pytest tests/test_benchmarks.py -m slow` inside the quantum_chem container.
The larger set (Diels-Alder, Pt(PH3)2+H2, the di-Zn hydrolysis model) is wired in
`benchmarks/` for the user's GPU sbatch — see benchmarks/README.md.
"""
from __future__ import annotations

import math
import os
import tempfile

import pytest

pytest.importorskip("pysisyphus")
from ase import Atoms  # noqa: E402
from ase.io import read  # noqa: E402

from quantum_engine.reaction_spec import ReactionSpec, RunContext  # noqa: E402
from quantum_engine.ops import ts_entry  # noqa: E402

pytestmark = pytest.mark.slow


def _hcn_ts_guess():
    import pysisyphus  # noqa: PLC0415
    gl = os.path.join(os.path.dirname(pysisyphus.__file__), "geom_library")
    return read(os.path.join(gl, "hcn_iso_ts_opt_xtb.xyz"))   # order: C(0) H(1) N(2)


def test_benchmark_hcn_isomerization():
    """HCN ↔ HNC: a clean 3-atom first-order saddle with a strong imaginary mode
    (~−1100..−1450 cm⁻¹ across methods). The whole pipeline must return n_imag=1
    and PASS the gate."""
    spec = ReactionSpec.from_dict({
        "forming_bonds": [["0:1", "0:2"]],     # H–N forms
        "breaking_bonds": [["0:1", "0:0"]],    # H–C breaks
        "reactive_atoms": ["0:0", "0:1", "0:2"]})
    ctx = RunContext(charge=0, spin=1, model="gfn2-xtb", device="cpu")
    with tempfile.TemporaryDirectory() as td:
        res = ts_entry.run(spec, ctx, entry="ts-guess", ts_guess=_hcn_ts_guess(),
                           outdir=td, rigor="standard", validate=False)
    assert res["status"] == "converged"
    assert res["n_imag"] == 1
    assert res["imag_freq_cm"] < -1000.0        # the isomerization mode
    assert res["gates_overall"] == "PASS"


def test_benchmark_sn2_charge_handling():
    """Cl⁻ + CH₃Cl symmetric SN2 (charge −1): validates the CHARGED pipeline
    mechanics — a first-order saddle is found with the imaginary mode on the
    reactive atoms and the −1 charge propagates to xTB. (GFN2's imaginary
    frequency is method-soft; the quantitative value is an MLFF/DFT job.)"""
    h = [[1.07 * math.cos(a), 1.07 * math.sin(a), 0.0]
         for a in (0, 2 * math.pi / 3, 4 * math.pi / 3)]
    ts = Atoms("CH3Cl2", positions=[[0, 0, 0]] + h + [[0, 0, 2.35], [0, 0, -2.35]])
    spec = ReactionSpec.from_dict({
        "forming_bonds": [["0:0", "0:5"]],     # C–Cl(5) forms
        "breaking_bonds": [["0:0", "0:4"]],    # C–Cl(4) breaks
        "reactive_atoms": ["0:0", "0:4", "0:5"]})
    ctx = RunContext(charge=-1, spin=1, model="gfn2-xtb", device="cpu")
    with tempfile.TemporaryDirectory() as td:
        res = ts_entry.run(spec, ctx, entry="ts-guess", ts_guess=ts, outdir=td,
                           rigor="standard", validate=False)
    assert res["status"] == "converged"
    assert res["n_imag"] == 1                   # exactly one (significant) imag mode
    assert res["imag_freq_cm"] is not None and res["imag_freq_cm"] < 0
    assert res["gates_overall"] == "PASS"
