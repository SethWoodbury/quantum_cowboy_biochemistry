from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "protonate_consensus_under_test",
    ROOT / "quantum_engine" / "prep" / "protonate_consensus.py",
)
assert spec and spec.loader
protonate_consensus = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = protonate_consensus
spec.loader.exec_module(protonate_consensus)

consensus_protonate = protonate_consensus.consensus_protonate
_infer_states_from_pdb = protonate_consensus._infer_states_from_pdb


def _atom(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    seq: int,
    x: float,
    y: float,
    z: float,
    *,
    record: str = "ATOM",
    element: str | None = None,
) -> str:
    if len(name) >= 4:
        atom_field = name[:4]
    else:
        atom_field = f" {name:<3}"
    element = element or name[0]
    return (
        f"{record:<6}{serial:>5} {atom_field} "
        f"{resname:>3} {chain}{seq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"  1.00  0.00          {element:>2}  \n"
    )


def test_consensus_reconciles_rule_only_hydrogens(tmp_path: Path) -> None:
    input_pdb = tmp_path / "input.pdb"
    output_pdb = tmp_path / "output.pdb"
    input_pdb.write_text(
        "".join(
            [
                _atom(1, "CG", "HIS", "A", 254, 0.000, 0.000, 0.000, element="C"),
                _atom(2, "ND1", "HIS", "A", 254, 1.000, 0.000, 0.000, element="N"),
                _atom(3, "CD2", "HIS", "A", 254, -0.500, 1.000, 0.000, element="C"),
                _atom(4, "CE1", "HIS", "A", 254, 0.600, 1.200, 0.000, element="C"),
                _atom(5, "NE2", "HIS", "A", 254, -0.200, 2.000, 0.000, element="N"),
                _atom(6, "HD1", "HIS", "A", 254, 1.700, -0.700, 0.000, element="H"),
                _atom(7, "NZ", "KCX", "A", 169, 5.000, 0.000, 0.000, record="HETATM", element="N"),
                _atom(8, "CX", "KCX", "A", 169, 6.300, 0.000, 0.000, record="HETATM", element="C"),
                _atom(9, "OQ1", "KCX", "A", 169, 6.900, 0.900, 0.000, record="HETATM", element="O"),
                _atom(10, "OQ2", "KCX", "A", 169, 6.900, -0.900, 0.000, record="HETATM", element="O"),
                _atom(11, "HQ2", "KCX", "A", 169, 7.800, -1.200, 0.000, record="HETATM", element="H"),
                "END\n",
            ]
        )
    )

    result = consensus_protonate(
        input_pdb=input_pdb,
        output_pdb=output_pdb,
        methods=("rules",),
        rules={
            "HIS:A:254": "doubly_protonated",
            "KCX:A:169": "deprotonated",
        },
        parallel=False,
    )

    assert result.consensus_states[("A", 254, "HIS")] == "doubly_protonated"
    assert result.consensus_states[("A", 169, "KCX")] == "deprotonated"

    output_text = output_pdb.read_text()
    assert " HE2 HIS A 254" in output_text
    assert " HQ2 KCX A 169" not in output_text

    inferred = _infer_states_from_pdb(output_pdb)
    assert inferred[("A", 254, "HIS")] == "doubly_protonated"
    assert inferred[("A", 169, "KCX")] == "deprotonated"
