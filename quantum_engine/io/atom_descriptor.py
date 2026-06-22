"""Flexible atom-descriptor resolution — identify an atom by many conventions.

Turn a human-friendly descriptor into a 0-based atom index, trying several
naming conventions and REQUIRING a unique match (else a clear, listing error).
Built so a user can refer to HETATM cofactors / ligand atoms by a readable code
instead of memorising raw indices.

Supported descriptors (case-insensitive on chain/resname/atom):

  raw 0-based index            ``260``      or ``"260"``
  RESNAME-ATOM                 ``"OHX-O3"``, ``"SUB-P1"``    (resname must be unique)
  <Chain><ResNo>-ATOM          ``"N998-O3"``, ``"A519-ZN"``  (chain letter + residue number)
  <RESNAME><ResNo>-ATOM        ``"ZN519-ZN"``                (resname + residue number)
  CHAIN:RESID:ATOM             ``"A:519:ZN"``                (explicit, always unambiguous)

Resolution tries EVERY interpretation and unions the matching atoms; a single
unique atom wins. Ambiguity (e.g. ``"ZN-ZN"`` with two zincs) or no match raises
a ``ValueError`` that lists the candidates and how to qualify them. This is the
single source of truth for atom addressing across the CLI (``monitor``,
``reaction-spec``) and the protonator's HETATM check.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

# letters-then-digits prefix, e.g. N998 (chain N, 998), ZN519 (resname ZN, 519)
_PREFIX_RE = re.compile(r"^([A-Za-z]+)(\d+)$")
_INT_RE = re.compile(r"-?\d+")


def _norm(s) -> str:
    return str(s).strip().upper()


@dataclass(frozen=True)
class AtomTable:
    """Per-atom identity columns (parallel lists); row index == 0-based atom index."""
    chains: list[str]
    resids: list[int]
    resnames: list[str]
    names: list[str]

    def __len__(self) -> int:
        return len(self.names)

    # ---- adapters ----
    @classmethod
    def from_biotite(cls, bt) -> "AtomTable":
        """Build from a biotite ``AtomArray`` (chain_id / res_id / res_name / atom_name)."""
        return cls(
            chains=[str(c) for c in bt.chain_id],
            resids=[int(r) for r in bt.res_id],
            resnames=[str(r) for r in bt.res_name],
            names=[str(a) for a in bt.atom_name],
        )

    @classmethod
    def from_atoms(cls, atoms) -> "AtomTable":
        """Build from protonator-style ``Atom`` objects (.chain/.resid/.resname/.name)."""
        return cls(
            chains=[(getattr(a, "chain", "") or "") for a in atoms],
            resids=[int(getattr(a, "resid", 0)) for a in atoms],
            resnames=[str(getattr(a, "resname", "")) for a in atoms],
            names=[str(getattr(a, "name", "")) for a in atoms],
        )


def _match(table: AtomTable, *, chain: Optional[str] = None,
           resid=None, resname: Optional[str] = None,
           name: Optional[str] = None) -> set[int]:
    """Atom rows matching every supplied (non-None) field. Empty if no field given."""
    if chain is None and resid is None and resname is None and name is None:
        return set()
    rid = None
    if resid is not None:
        try:
            rid = int(resid)
        except (ValueError, TypeError):
            return set()
    out: set[int] = set()
    for i in range(len(table)):
        if chain is not None and _norm(table.chains[i]) != _norm(chain):
            continue
        if rid is not None and int(table.resids[i]) != rid:
            continue
        if resname is not None and _norm(table.resnames[i]) != _norm(resname):
            continue
        if name is not None and _norm(table.names[i]) != _norm(name):
            continue
        out.add(i)
    return out


def _candidates(token: str, table: AtomTable) -> set[int]:
    """Union of atom rows matching ``token`` under every supported convention."""
    raw = token.strip()
    cands: set[int] = set()

    # explicit triple CHAIN:RESID:ATOM (e.g. A:519:ZN) OR RESNAME:RESID:ATOM (e.g. SUB:1:P1)
    if ":" in raw:
        parts = raw.split(":")
        if len(parts) == 3:
            a, rid, at = (p.strip() for p in parts)
            cands |= _match(table, chain=a, resid=rid, name=at)
            cands |= _match(table, resname=a, resid=rid, name=at)

    # dash forms: PREFIX-ATOM
    if "-" in raw:
        prefix, _, atom = raw.partition("-")
        prefix, atom = prefix.strip(), atom.strip()
        if atom:
            # (a) RESNAME-ATOM
            cands |= _match(table, resname=prefix, name=atom)
            # (b/c) <letters><digits>-ATOM → chain+resid AND resname+resid
            m = _PREFIX_RE.match(prefix)
            if m:
                letters, digits = m.group(1), int(m.group(2))
                cands |= _match(table, chain=letters, resid=digits, name=atom)
                cands |= _match(table, resname=letters, resid=digits, name=atom)
    return cands


def _check_range(i: int, table: AtomTable, raw: str) -> int:
    if 0 <= i < len(table):
        return i
    raise ValueError(f"atom reference {raw!r} -> index {i} out of range [0,{len(table)})")


def resolve_atom(token, table: AtomTable, *, bare_int: str = "index") -> int:
    """Resolve ANY atom reference to a 0-based index — the one canonical resolver.

    Accepted forms (case-insensitive on chain/resname/atom):
      ``0:N``                explicit 0-based ASE index
      ``serial:N``           1-based PDB serial (N-th atom in file order; index N-1)
      ``N`` (bare int)       0-based index if ``bare_int='index'`` (default), else
                             1-based serial when ``bare_int='serial'``
      ``RESNAME-ATOM``       e.g. OHX-O3, SUB-P1 (unique resname)
      ``<Chain><ResNo>-ATOM``  e.g. A519-ZN, N998-O3
      ``<RESNAME><ResNo>-ATOM``  e.g. ZN519-ZN
      ``CHAIN:RESID:ATOM``   e.g. A:519:ZN     ``RESNAME:RESID:ATOM``  e.g. SUB:1:P1

    Tries every interpretation and requires a UNIQUE atom. Raises ``ValueError``
    on empty/out-of-range/no-match/ambiguous (the message lists candidates).
    """
    if bare_int not in ("index", "serial"):
        raise ValueError("bare_int must be 'index' or 'serial'")
    raw = str(token).strip()
    if not raw:
        raise ValueError("empty atom reference")
    low = raw.lower()

    if low.startswith("0:") and _INT_RE.fullmatch(raw[2:]):
        return _check_range(int(raw[2:]), table, raw)
    if low.startswith("serial:") and _INT_RE.fullmatch(raw[7:]):
        return _check_range(int(raw[7:]) - 1, table, raw)
    if _INT_RE.fullmatch(raw):
        i = int(raw) if bare_int == "index" else int(raw) - 1
        return _check_range(i, table, raw)

    cands = _candidates(raw, table)
    if len(cands) == 1:
        return next(iter(cands))
    if not cands:
        raise ValueError(
            f"atom reference {token!r} matched no atom. Use RESNAME-ATOM "
            "(e.g. OHX-O3), <Chain><ResNo>-ATOM (A519-ZN), <RESNAME><ResNo>-ATOM "
            "(ZN519-ZN), CHAIN:RESID:ATOM (A:519:ZN), serial:N, 0:N, or an index.")
    cand = sorted(cands)
    shown = ", ".join(
        f"#{i}({table.chains[i]}{table.resids[i]}:{table.resnames[i]}:{table.names[i]})"
        for i in cand)
    raise ValueError(
        f"atom reference {token!r} is AMBIGUOUS — matches {len(cand)} atoms: "
        f"{shown}. Qualify it with a residue number or chain (e.g. ZN519-ZN).")


def resolve_atom_descriptor(token, table: AtomTable) -> int:
    """Back-compat alias for :func:`resolve_atom` (bare int = 0-based index)."""
    return resolve_atom(token, table, bare_int="index")


def best_descriptor(table: AtomTable, i: int) -> Optional[str]:
    """Return the most concise descriptor that UNIQUELY identifies atom ``i``.

    Tries RESNAME-ATOM, then <RESNAME><ResNo>-ATOM, then <Chain><ResNo>-ATOM, then
    CHAIN:RESID:ATOM. Returns ``None`` if NONE of them is unique (i.e. the atom is
    only addressable by its raw index — duplicate chain+resid+name).
    """
    ch, rid, rn, nm = (table.chains[i], table.resids[i],
                       table.resnames[i], table.names[i])
    for desc in (f"{rn}-{nm}", f"{rn}{rid}-{nm}", f"{ch}{rid}-{nm}",
                 f"{ch}:{rid}:{nm}"):
        try:
            if resolve_atom_descriptor(desc, table) == i:
                return desc
        except ValueError:
            continue
    return None


def resolve_many(tokens: Iterable, table: AtomTable, *, bare_int: str = "index") -> list[int]:
    """Resolve a sequence of atom references to 0-based indices (order preserved)."""
    return [resolve_atom(t, table, bare_int=bare_int) for t in tokens]


__all__ = ["AtomTable", "resolve_atom", "resolve_atom_descriptor", "best_descriptor",
           "resolve_many"]
