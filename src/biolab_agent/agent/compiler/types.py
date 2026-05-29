"""Typed data passed between Planner, Executor, and Renderers.

These are the public contracts of the pipeline.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Names exposed to the LLM planner. Tied to ToolRegistry at construction time;
# kept as a Literal so a mistyped tool name fails at schema validation.
ToolName = Literal[
    "segment_wells",
    "retrieve_protocol",
    "lookup_reagent",
    "compose_protocol",
]

TaskKind = Literal[
    "cell_count",
    "retrieval",
    "design",
    "catalog",
    "composite",
    "other",
]


class PlanStep(BaseModel):
    """One tool call the executor must run.

    ``extra="ignore"``: small models (medgemma:4b) often add chain-of-thought
    fields like ``"thought"`` or ``"observation"``. Rejecting those wastes
    retries on cosmetic violations; we silently drop them and keep the
    structural fields we actually need.
    """

    model_config = ConfigDict(extra="ignore")

    id: int = Field(ge=0)
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    depends_on: list[int] = Field(default_factory=list)
    foreach_image_id: bool = False


class Plan(BaseModel):
    """Output of the Planner. See PlanStep for the extra="ignore" rationale."""

    model_config = ConfigDict(extra="ignore")

    task_kind: TaskKind
    steps: list[PlanStep]
    success_criteria: str = ""


class RenderedAnswer(BaseModel):
    """What a Renderer returns to the agent."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    structured: dict[str, Any] = Field(default_factory=dict)
    citations: list[tuple[str, str]] = Field(default_factory=list)
