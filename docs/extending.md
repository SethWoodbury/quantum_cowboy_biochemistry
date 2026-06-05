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

## Where the registries live (quick reference)

| Axis            | Registry object                              | Register fn        | Factory / dispatch                |
|-----------------|----------------------------------------------|--------------------|-----------------------------------|
| Energy function | `calc.factory.ENERGY_FAMILIES`               | `register_energy`  | `make_calc(model, ...)`           |
| Minimizer       | `opt.factory.OPTIMIZERS`                      | `register_optimizer` | `make_optimizer(backend, ...)`  |
| Saddle          | `ops.saddle.SADDLE_OPTIMIZERS`               | `register_saddle`  | `make_saddle_optimizer` / `saddle.run` |
| Path            | `ops.path_search.PATH_METHODS`               | `register_path`    | `make_path_method` / `path_search.run` |
| IRC             | `ops.irc.IRC_METHODS`                         | `register_irc`     | `make_irc` / `irc.run`            |
| QM engine       | `qm.engine.ENGINES`                          | `register_engine`  | `make_engine` / `ts_entry` (ctx.engine) |

All lookups are case-insensitive and alias-aware; an unknown name lists the
available choices. See `tests/test_registry.py` and the per-axis test files for
worked examples.
