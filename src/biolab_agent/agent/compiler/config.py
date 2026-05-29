"""Runtime knobs for the planner agent.

Everything that used to be a module-level magic constant in
``planner.py`` is here, as a Pydantic model so values are validated
once at startup instead of being scattered across the code.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PlannerConfig(BaseModel):
    """Tuneable limits for the plan/execute/render pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_plan_steps: int = Field(8, ge=1, le=32)
    max_citations: int = Field(6, ge=1, le=64)
    planner_num_predict: int = Field(800, ge=64)
    renderer_num_predict: int = Field(600, ge=64)
    planner_temperature: float = Field(0.0, ge=0.0, le=2.0)
    renderer_temperature: float = Field(0.1, ge=0.0, le=2.0)
    llm_top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_parallel_calls: int = Field(1, ge=1, le=16)
    max_replans: int = Field(1, ge=0, le=4)
