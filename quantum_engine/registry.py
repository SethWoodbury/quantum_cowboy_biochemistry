"""A tiny, generic plugin registry — the extensibility backbone.

Every pluggable axis of the pipeline (energy functions, minimizers, saddle
optimizers, path methods, IRC engines, QM engines) is a :class:`Registry`. Adding
a new one as the field evolves is a ONE-LINER + a small adapter — no edits to the
orchestrator or CLI dispatch:

    from quantum_engine.opt.factory import OPTIMIZERS

    @OPTIMIZERS.register("my-fancy-min", aliases=("fancy",))
    class MyOptimizer(Optimizer): ...

    # or imperatively
    OPTIMIZERS.register("my-fancy-min", MyOptimizer)

Lookups are case-insensitive and alias-aware; ``names()`` is what an unknown-name
error lists, so new plugins are immediately discoverable.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable


class Registry:
    """Named registry of plugins of one ``kind`` (e.g. "optimizer")."""

    def __init__(self, kind: str):
        self.kind = kind
        self._reg: dict[str, Any] = {}
        self._alias: dict[str, str] = {}

    def register(self, name: str, obj: Any = None, *,
                 aliases: Iterable[str] = (), overwrite: bool = False):
        """Register ``obj`` under ``name`` (+ optional ``aliases``).

        Usable directly (``reg.register("x", thing)``) or as a decorator
        (``@reg.register("x")``). Refuses to clobber unless ``overwrite=True``.
        """
        key = name.lower()

        def _do(o: Any) -> Any:
            if key in self._reg and not overwrite:
                raise ValueError(
                    f"{self.kind} {name!r} is already registered "
                    f"(pass overwrite=True to replace)")
            self._reg[key] = o
            for a in aliases:
                self._alias[a.lower()] = key
            return o

        return _do(obj) if obj is not None else _do

    def unregister(self, name: str) -> None:
        key = self._resolve(name)
        self._reg.pop(key, None)
        for a, tgt in list(self._alias.items()):
            if tgt == key:
                self._alias.pop(a, None)

    def _resolve(self, name: str) -> str:
        key = str(name).lower()
        return self._alias.get(key, key)

    def get(self, name: str) -> Any:
        key = self._resolve(name)
        if key not in self._reg:
            raise KeyError(
                f"unknown {self.kind} {name!r}; available: {self.names()}")
        return self._reg[key]

    def names(self) -> list[str]:
        return sorted(self._reg)

    def aliases(self) -> dict[str, str]:
        return dict(self._alias)

    def __contains__(self, name: str) -> bool:
        return self._resolve(name) in self._reg

    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    def __iter__(self):
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._reg)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {len(self._reg)} entries: {self.names()})"


class PredicateRegistry:
    """Registry that dispatches by the FIRST matching predicate (for families).

    Used where the key isn't an exact name but a pattern over the model alias
    (e.g. "starts with 'uma-'" → UMA builder). Order of registration = priority.
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: list[tuple[str, Callable[[str], bool], Any]] = []

    def register(self, label: str, predicate: Callable[[str], bool], builder: Any) -> Any:
        self._entries.append((label, predicate, builder))
        return builder

    def unregister(self, label: str) -> None:
        """Remove all entries registered under ``label`` (no-op if absent)."""
        self._entries = [e for e in self._entries if e[0] != label]

    def match(self, name: str):
        for label, pred, builder in self._entries:
            try:
                if pred(name):
                    return label, builder
            except Exception:  # noqa: BLE001 — a bad predicate shouldn't break dispatch
                continue
        return None, None

    def labels(self) -> list[str]:
        return [lbl for lbl, _, _ in self._entries]

    def __repr__(self) -> str:
        return f"PredicateRegistry({self.kind!r}, families: {self.labels()})"


__all__ = ["Registry", "PredicateRegistry"]
