"""compose_protocol tool  -  validate + normalize a structured protocol object."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProtocolDraft(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str = Field(..., min_length=3, max_length=240)
    categories: list[str] = Field(default_factory=list)
    labware: list[str] = Field(default_factory=list)
    pipettes: list[str] = Field(default_factory=list)
    reagents: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("title", mode="before")
    @classmethod
    def title_no_placeholders(cls, v: str) -> str:
        if "ITEM_" in v:
            raise ValueError(
                f"unfilled placeholder {v!r} — replace with real values from the query"
            )
        return v

    @field_validator("labware", "pipettes", "reagents", "categories", mode="before")
    @classmethod
    def lists_no_placeholders(cls, v: list) -> list:
        for item in v:
            if isinstance(item, str) and "ITEM_" in item:
                raise ValueError(
                    f"unfilled placeholder {item!r} — replace with real values from the query"
                )
        return v


def compose_protocol(
    title: str,
    labware: list[str],
    pipettes: list[str],
    reagents: list[str],
    categories: list[str] | None = None,
    notes: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Validate and return the protocol as a JSON-serializable dict.

    Stripping blanks is deliberate: the upstream LLM sometimes emits empty
    strings which would otherwise inflate the list counts the harness checks.
    """
    draft = ProtocolDraft(
        title=title.strip(),
        labware=[x.strip() for x in labware if x and x.strip()],
        pipettes=[x.strip() for x in pipettes if x and x.strip()],
        reagents=[x.strip() for x in reagents if x and x.strip()],
        categories=[x.strip() for x in (categories or []) if x and x.strip()],
        notes=notes.strip() if notes else None,
    )
    return draft.model_dump()


compose_protocol_spec = {
    "type": "function",
    "function": {
        "name": "compose_protocol",
        "description": (
            "Validate and return a structured protocol object. Call this at the "
            "end, once you know the title, required labware, pipettes, and "
            "reagents. The result populates the structured output field."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short protocol name."},
                "labware": {"type": "array", "items": {"type": "string"}},
                "pipettes": {"type": "array", "items": {"type": "string"}},
                "reagents": {"type": "array", "items": {"type": "string"}},
                "categories": {"type": "array", "items": {"type": "string"}, "default": []},
                "notes": {"type": "string"},
            },
            "required": ["title", "labware", "pipettes", "reagents"],
        },
    },
}
