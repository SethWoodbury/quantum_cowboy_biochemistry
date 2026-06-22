# Extending the pipeline

Every pluggable axis of the TS pipeline is a **registry**. Adding a new energy
function, optimizer, saddle method, path method, IRC backend, or QM engine is a
one-liner plus a small adapter — **no edits to the orchestrator or the CLI
dispatch**. This is the primary design constraint: the pipeline stays easy to
extend as the field moves.

The stable contracts new code targets are:
- the ASE `Calculator` interface (energy functions),
- `ReactionSpec` / `RunContext` (the reaction + run state),
- the result dicts (`OptResult`, the saddle/path/IRC result shapes).

Register at import time (e.g. in your own module, or a plugin you import before
running). Below is a copy-paste template for each plug point.

---

## 1. A new energy function (MLFF / semiempirical / QM-as-ASE)

Registry: `quantum_engine.calc.factory.ENERGY_FAMILIES` (a `PredicateRegistry`).
A family is `(label, predicate-over-model-alias, builder)`; first matching
predicate wins. `mace` is the implicit fallback, so a new family with a specific
predicate is never shadowed.

```python
from quantum_engine.calc.factory import register_energy

def _build_myff(model, *, model_path, registry_path, head, device,
                default_dtype, charge, spin):
    from myff.ase import MyFFCalculator        # your calculator
    return MyFFCalculator(model_path or "myff-default",
                          device=device, charge=charge or 0, spin=spin or 1)

register_energy("myff", lambda m: m.lower().startswith("myff-"), _build_myff)
# now: make_calc("myff-small", charge=-1) routes here.
```

The builder MUST return an ASE-compatible `Calculator`. Put model weight paths in
`quantum_engine/site.py` (`MACE_MODELS`) so `make_calc("myff-small")` resolves a
local file; the builder receives it as `model_path`.

### Worked example: SO3LR — a model whose deps conflict with the main container

When a model's stack can't share the main container (here: SO3LR is JAX, the main
image is torch; UMA/eSEN are the same story with fairchem-core), it runs in a
**sidecar** apptainer image. The builder stays in-process but, when the backend
isn't importable, raises an `ImportError` that names the sidecar + the exact
`apptainer exec` line — so the same alias works whether you're inside the sidecar
or not. This is the actual `so3lr` wiring (`calc/factory.py`), reusable verbatim:

```python
import glob, os
from quantum_engine.calc.factory import register_energy

_SO3LR_SIDECAR_GLOB = "/net/software/containers/users/woodbuse/quantum_chem/so3lr-*.sif"

def _build_so3lr(model, *, model_path, registry_path, head, device,
                 default_dtype, charge, spin):
    try:
        from so3lr import So3lrCalculator          # JAX — only importable in the sidecar
    except ImportError as exc:
        sif = next(iter(sorted(glob.glob(_SO3LR_SIDECAR_GLOB), reverse=True)), None)
        hint = (f"re-run inside it:  apptainer exec --nv --bind /home --bind /net {sif} python ..."
                if sif else "build it:  apptainer build --fakeroot deps/so3lr_sidecar.def")
        raise ImportError(f"SO3LR ({model!r}) needs the JAX sidecar — {hint}") from exc
    import numpy as np
    # a SO3LR "model" is a DIRECTORY (workdir w/ params.pkl); site.py registers the
    # params.pkl, so derive the workdir. model_path is None → bare alias (bundled copy).
    target = os.path.dirname(model_path) if model_path and os.path.isfile(model_path) else model.lower()
    return So3lrCalculator(model=target, lr_cutoff=100.0, dtype=np.float64)

register_energy("so3lr", lambda m: m.lower().startswith("so3lr"), _build_so3lr)
```

The three artifacts that complete a sidecar model — copy this shape for the next one:
1. **builder** with the try-import → actionable-`ImportError` pattern above (in `calc/factory.py`);
2. **`site.py` aliases** → the weight path (a *file*; for directory-based models register the
   `params.pkl` inside the workdir so `os.path.isfile` resolution + "missing on disk" errors work);
3. **`deps/<name>_sidecar.def`** build recipe (model on `deps/uma_sidecar.def` / `deps/so3lr_sidecar.def`;
   use `%files` to copy a staged source tree in so the build doesn't depend on `/net` being bound).

Routing/error tests need no GPU or the real backend — see `tests/test_so3lr_routing.py`
(family dispatch, alias registry, non-shadowing, and the actionable sidecar `ImportError`).

---

## 2. A new minimizer

Registry: `quantum_engine.opt.factory.OPTIMIZERS`. A backend is an `Optimizer`
subclass (implements `.run(atoms, calculator) -> OptResult`); register a zero-arg
`factory_fn` that returns the class (lazily, so a missing dep doesn't break import).

```python
from quantum_engine.opt.factory import register_optimizer
from quantum_engine.opt.base import Optimizer, OptResult

class MyMinimizer(Optimizer):
    name = "my-min"
    def run(self, atoms, calculator=None):
        atoms = self._attach_calc(atoms, calculator)
        ...                                   # drive the relaxation
        return OptResult(status="converged", converged=True, atoms=atoms, ...)

register_optimizer("my-min", lambda: MyMinimizer, aliases=("mine",))
# set requires_torch_sim=True if it needs a torch/MLFF model (GPU fast path).
```

`make_optimizer("my-min", fmax=..., max_steps=...)` builds it;
`list_backends(available_only=True)` drops it if its deps don't import.

---

## 3. A new saddle (TS) optimizer

Registry: `quantum_engine.ops.saddle.SADDLE_OPTIMIZERS`. A backend is a *runner*
with the uniform signature below, returning the standard `saddle.run` dict
(`status`/`converged`/`atoms`/`energy_eV`/`backend`/`outputs`).

```python
from quantum_engine.ops.saddle import register_saddle

def _run_mysaddle(atoms, *, calculator, outdir, constraint, fmax, max_steps,
                  initial_mode_vector, eigh_drivers, **extra):
    ...
    return {"status": "converged", "converged": True, "atoms": atoms,
            "energy_eV": float(atoms.get_potential_energy()),
            "backend": "my-saddle", "outputs": {}}

register_saddle("my-saddle", _run_mysaddle, aliases=("ms",))
# saddle.run(atoms, backend="my-saddle", ...) and the auto-cascade can use it.
```

---

## 4. A new path (double-ended) method

Registry: `quantum_engine.ops.path_search.PATH_METHODS`. A runner takes a
*per-image* `calculator_fn` (a fresh calc per image) and returns the path dict
(`status`/`images`/`ts`/`ts_idx`/`energies_eV`/`barrier_*_kcal`/`outputs`).

```python
from quantum_engine.ops.path_search import register_path

def _run_mypath(reactant, product, calculator_fn, *, outdir, constraint, charge,
                **kwargs):
    ...
    return {"status": "converged", "images": images, "ts": images[ts_idx],
            "ts_idx": ts_idx, "energies_eV": energies, "outputs": {}}

register_path("my-path", _run_mypath, aliases=("mp",))
# path_search.run("my-path", R, P, calc_fn, ...) — endpoints are validated first.
```

---

## 5. A new IRC backend

Registry: `quantum_engine.ops.irc.IRC_METHODS`. A runner shares `irc.run`'s
signature and returns its dict (`status`/`ts`/`reactant`/`product`/`imag_freq_cm`/…).

```python
from quantum_engine.ops.irc import register_irc

def _run_myirc(atoms, calculator=None, outdir=".", constraint=None,
               refine_ts=True, **kwargs):
    ...
    return {"status": "converged", "ts": atoms, "reactant": r, "product": p,
            "imag_freq_cm": imag, "outputs": {}}

register_irc("my-irc", _run_myirc)
# irc.run(atoms, method="my-irc", ...)
```

---

## 6. A new QM-native engine (whole-step gateway)

Registry: `quantum_engine.qm.engine.ENGINES`. An engine routes the whole TS step
to a package's own optimizer (NEB-TS/OptTS for ORCA). A runner implements:

```python
from quantum_engine.qm.engine import register_engine

def run_myqm_engine(reaction, ctx, *, entry, outdir, reactant=None, product=None,
                    ts_guess=None, execute=True, **kwargs):
    # entry in {"ts-guess","reactant-product","reactant-only"}
    # write the package input, optionally run it (execute=False = prepare-only),
    # parse energy / frequencies / TS geometry, gate, return:
    return {"status": "converged", "engine": "myqm", "ts": ts_atoms,
            "energy_eV_ts": e_ev, "imag_freq_cm": imag, "n_imag": n,
            "outputs": {...}}

register_engine("myqm", run_myqm_engine)
# RunContext(engine="myqm") → ts_entry routes the whole step to it.
```

---

## 7. A new TS-guess proposer (generative / heuristic)

Registry: `quantum_engine.ops.ts_propose.TS_PROPOSERS`. A *proposer* suggests a
TS **guess** directly from a reactant + product, **skipping path search** — the
integration point for generative TS models (diffusion / flow-matching:
OA-ReactDiff, React-OT, AEFM, …). The guess then feeds the SAME canonical gate
(`ts_entry` → refine → partial-Hessian → IRC), so a proposer only *suggests* a TS;
the gate stays the acceptance authority.

```python
from quantum_engine.ops.ts_propose import register_ts_proposer

@register_ts_proposer("my-model", aliases=("mine",))
def _my_proposer(reactant, product, *, charge, spin, atom_map, outdir, **kw):
    # reactant & product arrive atom-order-consistent (mapped i→i); the model
    # may condition on charge/spin. Return ONE 3D guess (+ optional confidence).
    ts = my_model.predict_ts(reactant, product, charge=charge, spin=spin)
    return {"ts_guess": ts, "confidence": 0.9, "status": "converged", "outputs": {}}
```

Use it: `ts_entry.run(spec, ctx, entry="reactant-product", reactant=R, product=P,
proposer="my-model")` — the proposer replaces path search; everything downstream
is unchanged. A `ts_guess` of `None` / a non-`converged` status fails the run
cleanly (no crash). The built-in `midpoint` proposer (geodesic midpoint of R→P)
is a dependency-free reference + baseline. Note generative models are typically
trained on gas-phase neutral organics (Transition1x) — out of domain for
metal/charged active sites; the `atom_map` endpoint safeguard is reused here.
Built-in: `midpoint` (reference); `react-ot` (gated → `deps/reactot_sidecar.def`).

## 8. A new TS-guess refiner (ML structure refinement)

Registry: `quantum_engine.ops.ts_refine.TS_REFINERS`. A *refiner* is the mirror of
a proposer: it takes an EXISTING TS guess and returns a **better** one
(`(ts_guess) -> ts_guess`), where a proposer makes a guess from scratch
(`(R, P) -> ts_guess`). Separate axes keep each contract clean and let you compose
them: `path search / proposer → refiner → saddle+Hessian+IRC gate` (e.g. the
React-OT proposer → AEFM refiner chain). A refiner is a learned structure prior,
**not** an optimizer — it computes no Hessian and guarantees no saddle — so the QM
gate after it stays the authority; the refiner only improves the guess fed in.

```python
from quantum_engine.ops.ts_refine import register_ts_refiner

@register_ts_refiner("my-refiner", aliases=("mine",))
def _my_refiner(ts_guess, *, charge, spin, reactant, product, outdir, **kw):
    # reactant/product are optional context (ignored by guess-only refiners).
    better = my_model.refine(ts_guess)            # ONE improved 3D guess
    return {"ts_guess": better, "confidence": 0.9, "status": "converged", "outputs": {}}
```

Use it: `ts_entry.run(..., refiner="my-refiner")` (optionally with `proposer=` for
the chain). The refiner stage is **non-critical**: if it's unavailable or fails,
the run logs a WARN gate and falls back to the un-refined guess — it can never
break the pipeline. Built-in: `identity` (a no-op baseline — there is no sound
dependency-free "refine toward a saddle"); `aefm` (gated → `deps/aefm_sidecar.def`).
Same domain caveat as proposers (CHNO gas-phase organics; guarded in the adapter).

## Where the registries live (quick reference)

| Axis            | Registry object                              | Register fn        | Factory / dispatch                |
|-----------------|----------------------------------------------|--------------------|-----------------------------------|
| Energy function | `calc.factory.ENERGY_FAMILIES`               | `register_energy`  | `make_calc(model, ...)`           |
| Minimizer       | `opt.factory.OPTIMIZERS`                      | `register_optimizer` | `make_optimizer(backend, ...)`  |
| Saddle          | `ops.saddle.SADDLE_OPTIMIZERS`               | `register_saddle`  | `make_saddle_optimizer` / `saddle.run` |
| Path            | `ops.path_search.PATH_METHODS`               | `register_path`    | `make_path_method` / `path_search.run` |
| TS proposer     | `ops.ts_propose.TS_PROPOSERS`                | `register_ts_proposer` | `make_ts_proposer` / `ts_entry` (proposer=) |
| TS refiner      | `ops.ts_refine.TS_REFINERS`                  | `register_ts_refiner` | `make_ts_refiner` / `ts_entry` (refiner=) |
| IRC             | `ops.irc.IRC_METHODS`                         | `register_irc`     | `make_irc` / `irc.run`            |
| QM engine       | `qm.engine.ENGINES`                          | `register_engine`  | `make_engine` / `ts_entry` (ctx.engine) |

All lookups are case-insensitive and alias-aware; an unknown name lists the
available choices. See `tests/test_registry.py` and the per-axis test files for
worked examples.
