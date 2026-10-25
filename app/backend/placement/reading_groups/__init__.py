"""Per-question reading overrides, one module per subject area.

Each module exposes GROUPS: a list of Group. A group names the question ids it
covers, the key terms a verified page must contain, and exactly two readings.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass

from ..resources import Reading


@dataclass(frozen=True)
class Group:
    ids: tuple[int, ...]
    terms: tuple[str, ...]
    readings: tuple[Reading, Reading]


def G(ids, terms, first, second):
    """first and second are (title, url, publisher, why) tuples."""
    def mk(t):
        return Reading(t[0], t[1], t[2], "", t[3])
    if isinstance(ids, int):
        ids = (ids,)
    elif isinstance(ids, range):
        ids = tuple(ids)
    return Group(tuple(ids), tuple(terms.split()) if isinstance(terms, str) else tuple(terms), (mk(first), mk(second)))


def all_groups() -> list[Group]:
    groups: list[Group] = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        groups.extend(importlib.import_module(f"{__name__}.{info.name}").GROUPS)
    return groups


def overrides() -> dict[int, tuple[Reading, Reading]]:
    out: dict[int, tuple[Reading, Reading]] = {}
    for g in all_groups():
        for i in g.ids:
            if i in out:
                raise ValueError(f"question {i} has two override groups")
            out[i] = g.readings
    from ._aliases import ALIASES
    for alias, target in ALIASES.items():
        if alias in out:
            raise ValueError(f"question {alias} is both an alias and a group member")
        if target not in out:
            raise ValueError(f"alias {alias} points at {target}, which has no readings")
        out[alias] = out[target]
    return out


def CAT(prefix: str):
    """Return (title, url, publisher, why) for a verified catalog entry whose title starts with prefix."""
    from ..resources import CATALOG
    found = [r for r in CATALOG if r.title.startswith(prefix)]
    if len(found) != 1:
        raise KeyError(f"catalog entry for {prefix!r}: {len(found)} matches")
    r = found[0]
    return (r.title, r.url, r.publisher, r.why)
