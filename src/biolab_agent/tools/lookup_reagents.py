"""Optimized reagent catalog lookup used by ``OptimizedAgent``.

The baseline agent keeps using ``tools/reagents.py``. This module gives the
optimized agent its own catalog lookup boundary, so catalog behavior can evolve
without changing the reference tool.
"""

from __future__ import annotations

import csv
import functools
import json
import re
from pathlib import Path
from typing import Any

from biolab_agent.config import load_settings


@functools.lru_cache(maxsize=1)
def _optimized_catalog() -> list[dict[str, str]]:
    settings = load_settings()
    path = Path(settings.biolab_data_dir) / "reagents" / "catalog.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@functools.lru_cache(maxsize=1)
def _abbrev_map() -> dict[str, str]:
    settings = load_settings()
    path = Path(settings.biolab_data_dir) / "reagents" / "abbrev_map.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

_TRAILING_CONTEXT_RE = re.compile(r"\s*[,(].*$")
_PERCENT_WORD_RE = re.compile(r"(\d+)\s*percent\b", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _format_normalize(name: str) -> str:
    text = _MD_LINK_RE.sub(r"\1", name)
    text = _PERCENT_WORD_RE.sub(r"\1%", text)
    text = _TRAILING_CONTEXT_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lower()

@functools.lru_cache(maxsize=1)
def _catalog_index() -> dict[str, dict[str, str]]:
    """Map every normalized alias of every catalog row → that row.

    Built once at startup. A single catalog entry may be registered under
    several keys (raw name, format-normalized, abbrev-expanded variants).
    """
    index: dict[str, dict[str, str]] = {}
    for row in _optimized_catalog():
        key = _format_normalize(row["name"])
        if key:
            index.setdefault(key, row)

    return index


def _token_expand(text: str, abbrev_map: dict[str, str]) -> str | None:
    tokens = text.split()
    result: list[str] = []
    changed = False
    for tok in tokens:
        clean = tok.strip(".,;:-()")
        expansion = abbrev_map.get(clean) or abbrev_map.get(tok)
        if expansion:
            result.append(expansion)
            changed = True
        else:
            result.append(tok)
    return " ".join(result) if changed else None


@functools.lru_cache(maxsize=1)
def _reverse_abbrev_map() -> dict[str, list[str]]:
    """Map normalized long forms back to useful aliases seen in abbrev_map.

    ``abbrev_map.json`` mostly maps abbreviation -> long form, e.g.
    ``etoh -> ethanol``. Catalog rows may contain the abbreviation while the
    user asks with the long form, so lookup needs to try the reverse direction
    too. Very short aliases are skipped to avoid broad accidental matches.
    """
    reverse: dict[str, list[str]] = {}
    for alias, expansion in _abbrev_map().items():
        alias_norm = _format_normalize(alias)
        expansion_norm = _format_normalize(expansion)
        if len(alias_norm) < 3 or not expansion_norm:
            continue
        reverse.setdefault(expansion_norm, []).append(alias_norm)
    return {
        expansion: sorted(set(aliases), key=lambda alias: (len(alias), alias))
        for expansion, aliases in reverse.items()
    }


def _reverse_token_variants(text: str) -> list[str]:
    reverse = _reverse_abbrev_map()
    tokens = text.split()
    variants: list[str] = []
    for idx, token in enumerate(tokens):
        for alias in reverse.get(token, [])[:6]:
            candidate = [*tokens]
            candidate[idx] = alias
            variants.append(" ".join(candidate))
    return variants


def normalize_reagent_name(name: str) -> list[str]:
    """Return ordered candidate strings for name.

    The lookup tool will iterate these and return the first catalog hit.
    """
    amap = _abbrev_map()
    base = _format_normalize(name)
    seen: set[str] = set()
    candidates: list[str] = []

    def _add(s: str) -> None:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            candidates.append(s)

    _add(base)

    if phrase := amap.get(base):
        _add(phrase)

    if expanded := _token_expand(base, amap):
        _add(expanded)
        if phrase2 := amap.get(expanded):
            _add(phrase2)

    for variant in _reverse_token_variants(base):
        _add(variant)

    return candidates


def _to_lookup_result(requested_name: str, row: dict[str, str] | None) -> dict[str, Any]:
    """Return an explicit catalog decision instead of an ambiguous null.

    The baseline tool returns ``None`` for absence. The optimized tool uses a
    richer observation so the LLM sees the same decision a human would see:
    the requested item was searched, and the catalog either did or did not
    contain a matching record.
    """
    if row is None:
        return {
            "requested_name": requested_name,
            "found": False,
            "name": None,
            "record": None,
        }

    record = {
        "name": row["name"],
        "cas": row.get("cas") or None,
        "vendor": row.get("vendor") or None,
        "sku": row.get("sku") or None,
        "concentration": row.get("concentration") or None,
        "hazard": row.get("hazard") or None,
        "notes": row.get("notes") or None,
    }
    return {
        "requested_name": requested_name,
        "found": True,
        "name": record["name"],
        "record": record,
    }


def lookup_reagent(name: str) -> dict[str, Any]:
    """Return the first catalog match for name, with abbreviation expansion.

    Search order for each candidate from normalize_reagent_name():
      1. Exact key hit in the pre-built index (fastest).
      2. Candidate is a substring of an index key (e.g. "ethanol" inside
         "ethanol 80%").
      3. An index key is a substring of the candidate (reverse direction).
    Returns ``{"found": false}`` only after all candidates are exhausted.
    """
    requested_name = name.strip()
    if not requested_name:
        return _to_lookup_result(name, None)

    index = _catalog_index()
    candidates = normalize_reagent_name(requested_name)

    for candidate in candidates:
        # 1. exact key
        if candidate in index:
            return _to_lookup_result(requested_name, index[candidate])

        # 2 & 3. token-set scan over index keys (order-independent)
        cand_tokens = set(candidate.split())
        for key, row in index.items():
            key_tokens = set(key.split())
            if cand_tokens <= key_tokens or key_tokens <= cand_tokens:
                return _to_lookup_result(requested_name, row)

    return _to_lookup_result(requested_name, None)
