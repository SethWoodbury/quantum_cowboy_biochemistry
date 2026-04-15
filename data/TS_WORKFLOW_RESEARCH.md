# TS Search Workflow Research (April 2026)

## Recommended Pipeline for Enzyme Active Sites

```
Reactant + Product (or TS guess)
    |
    v
Geodesic Interpolation (internal coords, better than IDPP)
    |
    v
CI-NEB or FSM with MACE-OMol25/UMA    [MLFF path finding]
    |
    v
Sella TS refinement (internal coords)  [MLFF saddle point]
    |
    v
Frequency check (1 imaginary mode)     [MLFF validation]
    |
    v
IRC forward + reverse                  [MLFF connectivity check]
    |
    v  
FIRE + LBFGS relaxation of endpoints   [get proper reactant/product]
    |
    v
DFT TS optimization (few steps)        [optional DFT confirmation]
    |
    v
DFT frequency analysis                 [final validation]
```

This workflow achieves 94-96% reduction in DFT gradient evaluations.

## When to Use What

| Situation | Method |
|-----------|--------|
| Have TS guess from DFT | Sella (internal=True) directly |
| Have endpoints, want path | CI-NEB with geodesic interpolation |
| Large system >300 atoms | Dimer method (gradient-only, no Hessian) |
| Want TS guess without endpoints | FSM (pyGSM) or dimer |
| Need throughput screening | Quick NEB (mode=quick) + FSM |
| Publication quality | Full: NEB → Sella → IRC → freq → DFT SP |

## Available Tools (installed in deps/)

| Tool | Package | Use |
|------|---------|-----|
| Sella + IRC | `sella` | TS refinement + IRC validation |
| CI-NEB | `ase.mep.neb` | Path optimization |
| Dimer | `ase.mep.dimer` | Gradient-only TS search |
| Geodesic interp | `geodesic_interpolate` | Better NEB initial paths |
| pyGSM (FSM) | `pyGSM` | Growing/freezing string method |
| FIRE | `ase.optimize` | Robust pre-relaxation |
| LBFGS | `ase.optimize` | Fast near-minimum relaxation |
| Vibrations | `ase.vibrations` | Frequency analysis |

## Key Findings

1. **FSM achieves 88-90% success vs 63-71% for CI-NEB with MLIPs** (benchmark 2026)
2. **Sella with internal coords is best for molecular/enzyme systems**
3. **Dimer method: only needs gradients**, good for >200 atoms where Hessian is expensive
4. **FIRE→LBFGS only for IRC endpoints**, NOT for TS guess (destroys saddle point)
5. **BA-Sella (Bonds-Aware)**: ~97% success with 20 restarts (2025 paper)
6. **geodesic interpolation >> IDPP** for initial NEB paths

## Monte Carlo Sampling

| Package | Purpose |
|---------|---------|
| `grand` | GCMC water insertion/deletion in active sites |
| `BLUES` | NCMC sidechain rotamer sampling |
| `CREST` | Conformer-rotamer ensemble via xTB |
| `ASE-MC` (2025) | MC with any ASE calculator (including MACE) |

## Models for Enzyme TS Search

Ranked for Zn-centered PTE active sites:
1. **UMA-Medium** — best for transition-metal transfer (benchmark 2026)
2. **MACE-POLAR-1** — best physics for charged metal pockets
3. **MACE-OMol25 (XL)** — strongest overall TS-search result, mature

All support charge/spin. UMA needs FairChem calculator (different interface from MACE).
