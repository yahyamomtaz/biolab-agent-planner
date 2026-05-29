"""lookup_reagent tool  -  substring search against ``data/reagents/catalog.csv``."""

from __future__ import annotations

import csv
import functools
from pathlib import Path

from biolab_agent.config import load_settings
from biolab_agent.schemas import ReagentRecord


@functools.lru_cache(maxsize=1)
def _catalog() -> list[dict[str, str]]:
    settings = load_settings()
    path = Path(settings.biolab_data_dir) / "reagents" / "catalog.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def lookup_reagent(name: str) -> ReagentRecord | None:
    """Return the first catalog entry whose name contains ``name`` (case-insensitive)."""
    needle = name.strip().lower()
    if not needle:
        return None
    for row in _catalog():
        if needle in row["name"].lower():
            return ReagentRecord(
                name=row["name"],
                cas=row.get("cas") or None,
                vendor=row.get("vendor") or None,
                sku=row.get("sku") or None,
                concentration=row.get("concentration") or None,
                hazard=row.get("hazard") or None,
                notes=row.get("notes") or None,
            )
    return None


lookup_reagent_spec = {
    "type": "function",
    "function": {
        "name": "lookup_reagent",
        "description": (
            "Look up a reagent, labware item, or pipette in the local catalog "
            "and return its metadata (vendor, notes) if present. Returns null "
            "if the item is not in the catalog  -  do not invent data when null."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Reagent / labware name to search for (substring match).",
                },
            },
            "required": ["name"],
        },
    },
}

reagents = lookup_reagent

reagents_spec = {
    "type": "function",
    "function": {
        "name": "reagents",
        "description": (
            "Look up a reagent, labware item, or pipette in the local catalog "
            "and return its metadata (vendor, notes) if present. Returns null "
            "if the item is not in the catalog  -  do not invent data when null."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Reagent / labware name to search for (substring match).",
                },
            },
            "required": ["name"],
        },
    },
}
