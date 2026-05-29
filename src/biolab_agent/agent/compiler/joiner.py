"""Joiner

  1. Finalize
  2. Replan

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from biolab_agent.agent.compiler.executor import ExecutionResult
from biolab_agent.agent.compiler.types import Plan

JoinerAction = Literal["finalize", "replan"]


@dataclass(slots=True, frozen=True)
class JoinerDecision:
    """The Joiner's verdict for one Plan/ExecutionResult pair.

    ``feedback`` is non-empty only when ``action == "replan"`` — it's the
    failure summary the Planner receives so it can avoid the same
    mistake next round.
    """

    action: JoinerAction
    feedback: str = ""


class Joiner:
    """Decide whether the Executor's output is enough to finalize.

    Triggers replan when the run produced no useful evidence. Safe-by-
    default: any Plan that produced the expected kind of observation
    (cell_counts for cell_count tasks, citations for retrieval, etc.)
    finalizes immediately.
    """

    def decide(self, plan: Plan, result: ExecutionResult) -> JoinerDecision:
        # Hard failures — no executable steps, or every step errored.
        if not result.trace:
            return JoinerDecision(
                "replan",
                "previous plan emitted no executable steps; ensure steps[] is non-empty",
            )
        if not any(t.ok for t in result.trace):
            first_errors = [t.error or "?" for t in result.trace if not t.ok][:2]
            return JoinerDecision(
                "replan",
                f"every step in the previous plan failed; sample errors: {first_errors}. "
                "Pick different tool arguments or a different tool.",
            )

        # compose_protocol was planned but failed — most likely unfilled ITEM_*
        # placeholders. Replan with concrete feedback so the model replaces them
        # with real values from the query even if other steps succeeded.
        compose_was_planned = any(s.tool == "compose_protocol" for s in plan.steps)
        compose_failed = any(
            not t.ok and t.tool == "compose_protocol" for t in result.trace
        )
        if compose_was_planned and compose_failed and not result.composed:
            errors = [
                t.error for t in result.trace
                if not t.ok and t.tool == "compose_protocol" and t.error
            ][:1]
            return JoinerDecision(
                "replan",
                f"compose_protocol failed: {errors[0] if errors else 'validation error'}. "
                "Replace every ITEM_* placeholder with real values extracted from the user query.",
            )

        # Kind-specific empty-result triggers. Each one means "the tool
        # ran but produced nothing the renderer can use".
        if plan.task_kind == "cell_count" and not result.cell_counts:
            return JoinerDecision(
                "replan",
                "previous cell_count plan produced no per-well counts; ensure "
                "segment_wells is called with foreach_image_id=true.",
            )
        if plan.task_kind == "retrieval" and not result.citations:
            return JoinerDecision(
                "replan",
                "previous retrieval plan returned no citations; try different "
                "keywords in retrieve_protocol.args.query.",
            )
        if plan.task_kind == "design" and not result.composed:
            return JoinerDecision(
                "replan",
                "previous design plan produced no composed protocol; "
                "ensure compose_protocol is the only step and its args "
                "include title, labware, pipettes, reagents.",
            )
        if plan.task_kind == "catalog" and result.catalog_observation is None:
            return JoinerDecision(
                "replan",
                "previous catalog plan produced no lookup_reagent observation.",
            )
        if (
            plan.task_kind == "composite"
            and not result.citations
            and not result.cell_counts
            and not result.composed
        ):
            return JoinerDecision(
                "replan",
                "previous composite plan produced neither counts, citations, "
                "nor a composed protocol; ensure both steps are emitted and "
                "the second step depends_on the first.",
            )

        return JoinerDecision("finalize")
