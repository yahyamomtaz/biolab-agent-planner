"""Planner"""

from __future__ import annotations

from typing import Any

from biolab_agent.agent.compiler.config import PlannerConfig
from biolab_agent.agent.compiler.llm_session import LLMSession
from biolab_agent.agent.compiler.tool_registry import ToolRegistry
from biolab_agent.agent.compiler.types import Plan
from biolab_agent.llm.base import ChatClient

# The prompt embeds literal JSON snippets, so we substitute via sentinels
# instead of str.format() — otherwise every `{`/`}` would need doubling.
_SYSTEM_PROMPT_TEMPLATE = """You are a planning agent for an autonomous lab system.
Decompose the user's goal into a tool plan. Do not execute actions or write the final answer.

AVAILABLE TOOLS:
<<TOOL_SUMMARIES>>

OUTPUT FORMAT — output ONLY this JSON, nothing else:
{"task_kind":"cell_count|retrieval|design|catalog|composite|other","steps":[{"id":1,"tool":"segment_wells|retrieve_protocol|lookup_reagent|compose_protocol","args":{},"rationale":"brief action + verification","depends_on":[],"foreach_image_id":false}],"success_criteria":"checkable criterion"}

CONSTRAINTS:
- steps is always an array; top level must have task_kind, steps, success_criteria.
- Max <<MAX_STEPS>> steps. Steps are tool calls, not lab actions.
- Never invent doc_ids, counts, catalog metadata, or measurements.
- "Return a JSON object" in the goal is a final-output requirement — satisfy it with compose_protocol, do not output that JSON yourself.

DECISION RULES (apply in order):

1. IMAGE COUNTING — only when "Available image_ids" appears in the goal.
Emit one segment_wells step: args {}, foreach_image_id true, depends_on [].
Never create one step per image. Do not put image_id or well ids in args.
If the goal also asks to retrieve/find/look up a protocol → task_kind "composite",
add retrieve_protocol as step 2 (depends_on [1]). Conditional phrasing
("if count exceeds N, retrieve …") still requires step 2; the renderer
evaluates the condition after real counts exist.
No retrieval verb → task_kind "cell_count", one step only.

2. DESIGN. design/draft/compose/create/write/generate verb → task_kind "design",
one compose_protocol step. Extract title, labware, pipettes, reagents, notes from
the query. Do not add retrieve_protocol unless the query explicitly asks to look up
an existing library protocol first (that is Rule 3).

3. RETRIEVE THEN COMPOSE. find/retrieve … then adapt/compose →
task_kind "composite". Step 1: retrieve_protocol (depends_on []).
Step 2: compose_protocol (depends_on [1]).

4. RETRIEVAL ONLY. find/retrieve/look up an existing protocol, no compose →
task_kind "retrieval", one retrieve_protocol step.

5. CATALOG. reagent listed/exists/in catalog, or vendor/CAS/SKU question →
task_kind "catalog", one lookup_reagent step.

TEMPLATES — replace every ITEM_* with real values from the goal before output:
{"task_kind":"cell_count","steps":[{"id":1,"tool":"segment_wells","args":{},"rationale":"Segment all images; verify per-image counts returned.","depends_on":[],"foreach_image_id":true}],"success_criteria":"per-image cell counts available"}
{"task_kind":"composite","steps":[{"id":1,"tool":"segment_wells","args":{},"rationale":"Segment all images.","depends_on":[],"foreach_image_id":true},{"id":2,"tool":"retrieve_protocol","args":{"query":"ITEM_TOPIC"},"rationale":"Retrieve protocol.","depends_on":[1],"foreach_image_id":false}],"success_criteria":"counts and protocol hits available"}
{"task_kind":"composite","steps":[{"id":1,"tool":"retrieve_protocol","args":{"query":"ITEM_TOPIC"},"rationale":"Retrieve reference protocol.","depends_on":[],"foreach_image_id":false},{"id":2,"tool":"compose_protocol","args":{"title":"ITEM_TITLE","labware":["ITEM_LABWARE"],"pipettes":["ITEM_PIPETTE"],"reagents":["ITEM_REAGENT"],"notes":"ITEM_NOTES"},"rationale":"Compose adapted protocol.","depends_on":[1],"foreach_image_id":false}],"success_criteria":"protocol retrieved and adapted"}
{"task_kind":"design","steps":[{"id":1,"tool":"compose_protocol","args":{"title":"ITEM_TITLE","labware":["ITEM_LABWARE"],"pipettes":["ITEM_PIPETTE"],"reagents":["ITEM_REAGENT"],"notes":"ITEM_NOTES"},"rationale":"Compose protocol from query.","depends_on":[],"foreach_image_id":false}],"success_criteria":"structured protocol available"}
{"task_kind":"catalog","steps":[{"id":1,"tool":"lookup_reagent","args":{"name":"ITEM_NAME"},"rationale":"Look up catalog item.","depends_on":[],"foreach_image_id":false}],"success_criteria":"found/not-found observation available"}
"""


def _format_tool_summary(specs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for spec in specs:
        fn = spec["function"]
        props = fn["parameters"].get("properties", {})
        args_desc = ", ".join(f"{k}:{props[k].get('type', 'any')}" for k in props)
        lines.append(f"- {fn['name']}({args_desc})  -  {fn['description']}")
    return "\n".join(lines)


class Planner:
    """Produces a typed Plan from a user query.

    One responsibility: turn a query into a Plan. Knows nothing about
    execution, observations, or rendering.
    """

    def __init__(
        self,
        llm: ChatClient,
        model: str,
        tools: ToolRegistry,
        config: PlannerConfig,
    ) -> None:
        self._llm = llm
        self._model = model
        self._config = config
        self._system_prompt = (
            _SYSTEM_PROMPT_TEMPLATE
            .replace("<<TOOL_SUMMARIES>>", _format_tool_summary(tools.specs))
            .replace("<<MAX_STEPS>>", str(config.max_plan_steps))
        )

    def plan(
        self,
        query: str,
        image_ids: list[str] | None,
        *,
        feedback: str | None = None,
    ) -> Plan | None:
        """Produce a Plan for ``query``.

        ``feedback`` is set by the Joiner on a replan attempt: a short
        description of why the previous Plan produced no useful evidence.
        We append it to the user message so the same prompt structure is
        reused but the model sees concrete failure context — enough to
        diverge from the prior plan even at temperature 0.0.
        """
        session = LLMSession(
            self._llm,
            self._model,
            system_prompt=self._system_prompt,
            temperature=self._config.planner_temperature,
            num_predict=self._config.planner_num_predict,
            top_p=self._config.llm_top_p,
            max_attempts=3,
        )
        user = "Goal:\n" + query + (
            f"\n\nAvailable image_ids: {list(image_ids)}" if image_ids else ""
        )
        if feedback:
            user += (
                f"\n\nNOTE — your previous plan failed: {feedback} "
                "Produce a new Plan that avoids that failure."
            )
        session.add_user(user)
        # Plan is nested (Literal[tool_names], lists of typed steps).
        # Ollama's structured-output mode chokes on it for small models, so
        # rely on plain JSON mode plus the post-validate retry inside
        # LLMSession. Catalog/render schemas are flat and keep constrained
        # decoding on.
        return session.complete(Plan, constrained_decoding=False)
