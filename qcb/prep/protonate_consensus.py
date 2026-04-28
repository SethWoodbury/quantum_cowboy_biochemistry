"""Consensus protonation: combine ChimeraX, propka, pdbfixer, and hardcoded rules.

Why consensus
-------------
Every single protonation method has known failure modes:

| Method     | Strength                           | Fails on                              |
|------------|-----------------------------------|---------------------------------------|
| pdbfixer   | Fast, full residue templates       | Unusual protonation states, ligands   |
| propka     | Quantitative pKa from electrostatic | Non-standard residues, pKa edge cases |
| ChimeraX   | Empirical H placement, geometry    | Slow, kernel may not be available     |
| Hardcoded  | Mechanistic knowledge (KCX, metals)| Brittle, requires curation per system |

Combining them via a consensus arbiter is more robust than any single one:

1. Each method runs INDEPENDENTLY → easy to parallelize, isolate failures
2. For each ionizable residue, we collect a "vote" from each method
3. Conflict-resolution rules (configurable):
   - Hardcoded rules (mechanism-specific) ALWAYS win
   - Otherwise: majority vote
   - Tie: prefer propka (best pKa physics)
   - All disagreed: flag in audit log + use propka

4. Audit log records: per-residue, what each method said, what consensus chose,
   and why. This is invaluable for debugging "why did this residue end up like that?"

Usage
-----
    from qcb.prep.protonate_consensus import consensus_protonate
    result = consensus_protonate(
        "input.pdb",
        pH=7.0,
        ligand_charges={"YYL": -1, "ZN": 2},
        methods=["chimera", "propka", "pdbfixer", "rules"],
        rules={"HIS:254": "neutral", "LYS:139": "carbamylated"},
    )
    print(result.consensus_states)   # {(chain, resnum, resname): "protonated|neutral|deprotonated"}
    print(result.disagreements)      # list of residues where methods disagreed
    print(result.protonated_pdb)     # final structure path
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

log = logging.getLogger("qcb.prep.protonate_consensus")


@dataclass
class MethodResult:
    """Per-method protonation output."""
    method_name: str
    success: bool
    states: dict[tuple[str, int, str], str] = field(default_factory=dict)
    # key: (chain, resnum, resname); value: 'protonated' | 'neutral' | 'deprotonated' | 'unknown'
    protonated_pdb: Path | None = None
    log: str = ""


@dataclass
class ConsensusResult:
    """Final output of consensus_protonate."""
    consensus_states: dict[tuple[str, int, str], str]
    method_results: dict[str, MethodResult]
    disagreements: list[dict]                     # residues where methods disagreed
    audit_log: list[str]                          # one line per residue decision
    protonated_pdb: Path | None = None            # final structure file
    net_charge: int | None = None


# ═══════════════════════════════════════════════════════════════════
# Per-method runners (each returns a MethodResult)
# ═══════════════════════════════════════════════════════════════════

def _run_propka_method(
    input_pdb: Path, pH: float, **kwargs
) -> MethodResult:
    """propka pKa prediction → protonation state at the given pH.

    `qcb.prep.protonate.get_pka_dict` returns:
      {(chain, resid): {'pKa': float, 'resn': '3LETTER', 'protonated': bool}}
    """
    try:
        from qcb.prep.protonate import get_pka_dict
        pka = get_pka_dict(str(input_pdb), pH=pH)
        states: dict[tuple[str, int, str], str] = {}
        for (chain, resid), info in pka.items():
            resname = info.get("resn") or info.get("res_name") or info.get("residue") or "UNK"
            pka_val = info.get("pKa")
            if pka_val is None:
                states[(chain, resid, resname)] = "unknown"
                continue
            # Henderson-Hasselbalch decision
            states[(chain, resid, resname)] = _hh_decision(resname, pka_val, pH)
        return MethodResult("propka", True, states=states)
    except Exception as e:
        log.warning(f"  propka method failed: {e}")
        return MethodResult("propka", False, log=str(e))


def _run_chimera_method(
    input_pdb: Path, outdir: Path, ligand_charges: dict | None, **kwargs
) -> MethodResult:
    """ChimeraX addh + Gasteiger → protonated PDB; states inferred by H presence."""
    try:
        from qcb.prep.protonate_chimera import add_hydrogens_with_charges
        chim = add_hydrogens_with_charges(
            input_pdb,
            output_pdb=outdir / "chimera_protonated.pdb",
            pqr_path=outdir / "chimera_protonated.pqr",
            ligand_charges=ligand_charges,
        )
        if not chim.success:
            return MethodResult("chimera", False, log=chim.log)
        states = _infer_states_from_pdb(chim.protonated_pdb)
        return MethodResult(
            "chimera", True, states=states,
            protonated_pdb=chim.protonated_pdb,
            log=f"{chim.n_h_added} H added, {len(chim.partial_charges)} charges",
        )
    except Exception as e:
        log.warning(f"  ChimeraX method failed: {e}")
        return MethodResult("chimera", False, log=str(e))


def _run_pdbfixer_method(
    input_pdb: Path, outdir: Path, pH: float, **kwargs
) -> MethodResult:
    """pdbfixer addMissingHydrogens at the given pH."""
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        fixer = PDBFixer(filename=str(input_pdb))
        fixer.addMissingHydrogens(pH=pH)
        out = outdir / "pdbfixer_protonated.pdb"
        with open(out, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        states = _infer_states_from_pdb(out)
        return MethodResult("pdbfixer", True, states=states, protonated_pdb=out)
    except Exception as e:
        log.warning(f"  pdbfixer method failed: {e}")
        return MethodResult("pdbfixer", False, log=str(e))


def _run_rules_method(
    input_pdb: Path, rules: dict[str, str] | None, **kwargs
) -> MethodResult:
    """Hardcoded mechanism-specific rules.

    `rules` is a dict mapping "RESNAME:RESID" or "RESNAME:CHAIN:RESID" to
    a state name. Examples:
        {"HIS:254": "doubly_protonated", "LYS:139": "carbamylated", "ASP:233": "neutral"}
    """
    states: dict[tuple[str, int, str], str] = {}
    if not rules:
        return MethodResult("rules", True, states=states, log="no rules supplied")

    # Parse rules into normalized form
    for spec, state in rules.items():
        parts = spec.split(":")
        if len(parts) == 2:
            resname, resid = parts[0], int(parts[1])
            chain = "*"  # any chain
        elif len(parts) == 3:
            resname, chain, resid = parts[0], parts[1], int(parts[2])
        else:
            log.warning(f"  rules: ignoring malformed spec '{spec}'")
            continue
        states[(chain, resid, resname)] = state

    return MethodResult("rules", True, states=states,
                       log=f"{len(states)} rules supplied")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

# Default model pKa values (used in fallback HH decisions)
_MODEL_PKA = {"ASP": 3.65, "GLU": 4.25, "HIS": 6.00,
              "CYS": 8.18, "TYR": 10.07, "LYS": 10.54, "ARG": 12.48}


def _hh_decision(resname: str, pka: float, pH: float) -> str:
    """Henderson-Hasselbalch protonation decision.

    Acidic side chains (ASP/GLU/CYS/TYR): protonated if pH < pKa else deprotonated.
    Basic side chains (LYS/ARG): protonated if pH < pKa else neutral.
    HIS: protonated (HIP) if pH < pKa, else neutral (random tautomer HID/HIE).
    """
    acid = resname in {"ASP", "GLU", "CYS", "TYR"}
    base = resname in {"LYS", "ARG"}
    if resname == "HIS":
        return "doubly_protonated" if pH < pka else "neutral"
    if acid:
        return "protonated" if pH < pka else "deprotonated"
    if base:
        return "protonated" if pH < pka else "neutral"
    return "unknown"


def _infer_states_from_pdb(pdb_path: Path) -> dict[tuple[str, int, str], str]:
    """Infer protonation state per residue by inspecting which H atoms are present.

    Heuristics (not exhaustive — adequate for consensus voting):
      ASP: HD2 present → protonated; absent → deprotonated
      GLU: HE2 present → protonated; absent → deprotonated
      HIS: HD1 + HE2 → doubly_protonated (HIP); only HE2 → HIE (neutral);
           only HD1 → HID (neutral)
      LYS: HZ3 present → protonated; absent → neutral
      ARG: HE + 2 HH → protonated (default in proteins)
      CYS: HG present → protonated; absent → deprotonated
      TYR: HH present → protonated; absent → deprotonated
    """
    by_residue: dict[tuple[str, int, str], set[str]] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21:22].strip() or "A"
            try:
                res_id = int(line[22:26].strip())
            except ValueError:
                continue
            key = (chain_id, res_id, res_name)
            by_residue.setdefault(key, set()).add(atom_name)

    states: dict[tuple[str, int, str], str] = {}
    for key, atoms in by_residue.items():
        rn = key[2]
        if rn == "ASP":
            states[key] = "protonated" if "HD2" in atoms else "deprotonated"
        elif rn == "GLU":
            states[key] = "protonated" if "HE2" in atoms else "deprotonated"
        elif rn == "HIS":
            hd1 = "HD1" in atoms
            he2 = "HE2" in atoms
            if hd1 and he2:
                states[key] = "doubly_protonated"
            elif he2:
                states[key] = "neutral"  # HIE
            elif hd1:
                states[key] = "neutral"  # HID
            else:
                states[key] = "unknown"
        elif rn == "LYS":
            states[key] = "protonated" if "HZ3" in atoms or "HZ1" in atoms else "neutral"
        elif rn == "CYS":
            states[key] = "protonated" if "HG" in atoms else "deprotonated"
        elif rn == "TYR":
            states[key] = "protonated" if "HH" in atoms else "deprotonated"
        # Other residues: skip (not pH-sensitive)
    return states


# ═══════════════════════════════════════════════════════════════════
# The arbiter
# ═══════════════════════════════════════════════════════════════════

def _arbitrate_residue(
    key: tuple[str, int, str],
    method_states: dict[str, str],         # {method_name: state}
    priority: list[str] = None,
) -> tuple[str, str]:
    """Decide a single residue's state given the method votes.

    Returns: (consensus_state, rationale_string)

    Rules:
    1. If 'rules' (hardcoded) provided a state → use it. (Mechanism wins.)
    2. Else if all methods (excluding 'rules') agree → use that state.
    3. Else majority (excluding 'rules' and 'unknown')
    4. Tie or no majority → use propka if available, else first non-unknown
    """
    # 1. Hardcoded rule wins
    if "rules" in method_states and method_states["rules"] != "unknown":
        return method_states["rules"], "hardcoded rule"

    # Filter out unknowns + missing 'rules' (already handled)
    votes = {m: s for m, s in method_states.items() if s != "unknown" and m != "rules"}
    if not votes:
        return "unknown", "no method produced a state"

    # 2. Unanimous agreement
    unique_states = set(votes.values())
    if len(unique_states) == 1:
        return next(iter(unique_states)), f"unanimous: {list(votes)}"

    # 3. Majority vote
    from collections import Counter
    cnt = Counter(votes.values())
    top_state, top_count = cnt.most_common(1)[0]
    if top_count > len(votes) / 2:
        winners = [m for m, s in votes.items() if s == top_state]
        return top_state, f"majority {top_count}/{len(votes)}: {winners}"

    # 4. Tie — prefer propka
    priority = priority or ["propka", "chimera", "pdbfixer"]
    for m in priority:
        if m in votes:
            return votes[m], f"tie broken by priority ({m})"
    return next(iter(votes.values())), "tie, no priority match"


def consensus_protonate(
    input_pdb: str | Path,
    output_pdb: str | Path | None = None,
    pH: float = 7.0,
    methods: Iterable[str] = ("chimera", "propka", "pdbfixer", "rules"),
    ligand_charges: dict[str, int] | None = None,
    rules: dict[str, str] | None = None,
    parallel: bool = True,
    method_priority: list[str] | None = None,
) -> ConsensusResult:
    """Run multiple protonation methods in parallel and arbitrate.

    Args:
        input_pdb: input structure
        output_pdb: where to write the FINAL consensus structure
        pH: reference pH
        methods: which methods to run. Subset of {"chimera", "propka", "pdbfixer", "rules"}.
        ligand_charges: {residue_name: net_charge} hints for non-standard ligands
        rules: {"RESNAME:RESID": "state", ...} hardcoded overrides (always win)
        parallel: run methods concurrently (default: True)
        method_priority: for tie-breaking (default: propka > chimera > pdbfixer)

    Returns: ConsensusResult with per-residue decisions, audit log, and the
             final protonated PDB (taken from the highest-priority method that
             actually produced a structure file).
    """
    input_pdb = Path(input_pdb).resolve()
    if output_pdb is None:
        output_pdb = input_pdb.with_suffix(".consensus.pdb")
    else:
        output_pdb = Path(output_pdb).resolve()
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    # Working dir for intermediate per-method outputs
    workdir = output_pdb.parent / f".{input_pdb.stem}_protonate_work"
    workdir.mkdir(exist_ok=True)

    method_runners = {
        "propka":   _run_propka_method,
        "chimera":  _run_chimera_method,
        "pdbfixer": _run_pdbfixer_method,
        "rules":    _run_rules_method,
    }
    methods = [m for m in methods if m in method_runners]
    log.info(f"Consensus protonation: methods = {methods}, pH = {pH}")

    # Run all methods (optionally in parallel)
    method_results: dict[str, MethodResult] = {}
    runner_kwargs = dict(
        input_pdb=input_pdb, outdir=workdir, pH=pH,
        ligand_charges=ligand_charges, rules=rules,
    )

    def _safe_run(m):
        try:
            return method_runners[m](**runner_kwargs)
        except Exception as e:
            log.error(f"  method '{m}' raised: {e}")
            return MethodResult(m, False, log=str(e))

    if parallel and len(methods) > 1:
        with ThreadPoolExecutor(max_workers=len(methods)) as ex:
            futures = {ex.submit(_safe_run, m): m for m in methods}
            for fut in futures:
                m = futures[fut]
                method_results[m] = fut.result()
    else:
        for m in methods:
            method_results[m] = _safe_run(m)

    # Arbitrate per residue
    all_keys: set[tuple[str, int, str]] = set()
    for r in method_results.values():
        all_keys.update(r.states.keys())

    consensus_states: dict[tuple[str, int, str], str] = {}
    audit: list[str] = []
    disagreements: list[dict] = []

    # Drop wildcard "*" keys from the iteration set (they're rule-only entries
    # whose intent is to apply across any chain). They get merged into per-chain
    # decisions via the wildcard lookup below.
    iter_keys = {k for k in all_keys if k[0] != "*"}

    for key in sorted(iter_keys):
        votes = {m: r.states.get(key, "unknown") for m, r in method_results.items()
                 if r.success}
        # Rules votes: lookup with both exact and wildcard chain
        if "rules" in method_results and method_results["rules"].success:
            rs = method_results["rules"].states
            wildcard_key = ("*", key[1], key[2])
            if wildcard_key in rs:
                votes["rules"] = rs[wildcard_key]
            elif key in rs:
                votes["rules"] = rs[key]

        state, rationale = _arbitrate_residue(key, votes, method_priority)
        consensus_states[key] = state

        line = (f"  {key[2]} {key[0]}:{key[1]:>4} → {state:<22s} "
                f"({rationale}; votes={votes})")
        audit.append(line)

        if len(set(s for s in votes.values() if s != "unknown")) > 1:
            disagreements.append({
                "residue": f"{key[2]} {key[0]}:{key[1]}",
                "votes": votes,
                "consensus": state,
                "rationale": rationale,
            })

    # Pick the FINAL protonated PDB from the highest-priority successful method
    priority_for_pdb = ["chimera", "pdbfixer"]  # methods that produce a PDB
    final_pdb_source = None
    for m in priority_for_pdb:
        r = method_results.get(m)
        if r and r.success and r.protonated_pdb and r.protonated_pdb.is_file():
            import shutil
            shutil.copy2(str(r.protonated_pdb), str(output_pdb))
            final_pdb_source = m
            log.info(f"  Final structure copied from '{m}': {output_pdb}")
            break

    if final_pdb_source is None:
        log.warning("  No method produced a usable PDB; copying input as fallback")
        import shutil
        shutil.copy2(str(input_pdb), str(output_pdb))

    # Print audit summary
    log.info("=" * 60)
    log.info("Consensus protonation summary:")
    for line in audit[:20]:
        log.info(line)
    if len(audit) > 20:
        log.info(f"  ... and {len(audit) - 20} more residues (see ConsensusResult.audit_log)")
    if disagreements:
        log.info(f"  *** {len(disagreements)} residue(s) had method disagreements (see .disagreements)")

    return ConsensusResult(
        consensus_states=consensus_states,
        method_results=method_results,
        disagreements=disagreements,
        audit_log=audit,
        protonated_pdb=output_pdb,
    )
