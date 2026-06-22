# opaa_theozyme — key structures (deliverables)

Clean, stem-named copies of the pipeline's important outputs. Stem `opaa_theozyme` derives
from the initial theozyme input. (Source paths in parentheses.)

| file | what it is |
|---|---|
| `opaa_theozyme_ca_frozen_relaxed.pdb`            | CA-frozen relaxed cluster / TS-region guess (relax_minimize/relax/relaxed.pdb) |
| `opaa_theozyme_minimized_reactant.pdb`           | true reactant minimum (endpoints/reactant/reactant_min.pdb) |
| `opaa_theozyme_minimized_product.pdb`            | true product minimum  (endpoints/product/product_min.pdb) |
| `opaa_theozyme_scan_ts_guess.pdb`                | 1-D scan TS guess (under-estimates the barrier; see notes) |
| `opaa_theozyme_neb_ts_guess_n11.pdb`             | CI-NEB (11-image) climbing image |
| `opaa_theozyme_neb_ts_guess_pentacoordinate.pdb` | **BEST TS structure** — CI-NEB (25-image) symmetric pentacoordinate (O3–P 1.85 / P–O7 1.83 Å) |

## Honest status (mace-polar-m)
Concerted SN2-at-P, symmetric pentacoordinate TS, reaction exothermic ~14.5 kcal/mol. The barrier
region is a **flat ridge** (~1 kcal relaxed scan / ~6–8 kcal constrained NEB) with **no Hessian-verified
first-order saddle** at this MLFF level (soft/peripheral imaginary modes only). **Carry**
`opaa_theozyme_neb_ts_guess_pentacoordinate.pdb` **to DFT or QM/MM** for a quantitative barrier + a
verified saddle. See `docs/ts_search_pitfalls_and_methods.md` and notebook STEP 8b.
