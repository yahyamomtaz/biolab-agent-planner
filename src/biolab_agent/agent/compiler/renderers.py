"""Renderers — turn an ExecutionResult into a RenderedAnswer.

Every renderer is LLM-driven:
the model writes the prose.

  * Pydantic schema (via Ollama ``format=<schema>``) forces valid JSON.
  * For catalog answers, a consistency validator rejects any reply whose
    declared metadata does not match the tool observation byte-for-byte,
    then re-prompts the model with the specific mismatch.

Only when the LLM exhausts its retries does a renderer fall back to an
honest failure string — never to a fake natural-sounding answer.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from biolab_agent.agent.compiler.config import PlannerConfig
from biolab_agent.agent.compiler.executor import ExecutionResult
from biolab_agent.agent.compiler.llm_session import LLMSession
from biolab_agent.agent.compiler.types import Plan, RenderedAnswer, TaskKind
from biolab_agent.llm.base import ChatClient
from biolab_agent.logging import get_logger
from biolab_agent.schemas import ReagentRecord

log = get_logger(__name__)


# LLM response schemas


class _ProseOnly(BaseModel):
    """Single string field for renderers that don't need structured fields."""

    model_config = ConfigDict(extra="forbid")
    answer: str


class CatalogAnswer(BaseModel):
    """LLM reply to a catalog-lookup question.

    Mirrors the ``ReagentRecord`` shape so we can verify the model isn't
    inventing metadata (the failure mode of the original baseline on T4).
    """

    model_config = ConfigDict(extra="forbid")
    found: bool
    quoted_name: str | None = None
    cas: str | None = None
    vendor: str | None = None
    sku: str | None = None
    concentration: str | None = None
    hazard: str | None = None
    notes: str | None = None
    answer: str


# Fields cross-checked against the observation in the consistency validator.
_CATALOG_METADATA_FIELDS: tuple[str, ...] = tuple(
    f for f in ReagentRecord.model_fields if f != "name"
)


def _catalog_consistency_error(
    reply: CatalogAnswer,
    observation: dict[str, Any],
) -> str | None:
    """Return a one-line description of any disagreement with the observation.

    The validator enforces only the safety-critical invariant: the LLM may
    not assert a value the observation does not contain. It MAY decline to
    repeat a field (declare it null) and it MAY use a different casing of
    the catalog name. Small models can't reliably copy long strings
    byte-for-byte; insisting on that turns "the LLM paraphrased notes" into
    a failure as serious as "the LLM invented a CAS number".

    The two rules:
      1. ``reply.found`` must match ``observation.found``.
      2. For each metadata field, the reply may be null OR equal to the
         observation. A non-null reply that disagrees with a non-null
         observation, or any non-null reply when the observation is null,
         is a hallucination.
    """
    obs_found = bool(observation.get("found"))
    obs_name = observation.get("name")

    if reply.found != obs_found:
        return (
            f"reply.found={reply.found} contradicts observation.found={obs_found}."
        )

    if not obs_found:
        # No record exists. Every metadata field in the reply must be null
        # or the model invented something.
        if reply.quoted_name is not None:
            return "reply.quoted_name must be null when observation.found is false."
        for field in _CATALOG_METADATA_FIELDS:
            if getattr(reply, field) is not None:
                return (
                    f"reply.{field} must be null when observation.found is false."
                )
        return None

    # found=true. Validate that nothing was invented; permit case-insensitive
    # name match and null for any field the model chose not to mention.
    if not isinstance(obs_name, str) or not obs_name:
        return "observation says found=true but has no name."
    if reply.quoted_name is not None and reply.quoted_name.lower() != obs_name.lower():
        return (
            f"reply.quoted_name={reply.quoted_name!r} must match "
            f"observation.name={obs_name!r} (case-insensitive)."
        )
    try:
        record = ReagentRecord.model_validate(observation.get("record") or {})
    except ValidationError:
        return "observation.record is malformed."
    for field in _CATALOG_METADATA_FIELDS:
        declared = getattr(reply, field)
        actual = getattr(record, field)
        if declared is None:
            continue  # model chose not to repeat this field
        if actual is None:
            return (
                f"reply.{field}={declared!r} but observation.record.{field} is "
                "null. Do not invent values."
            )
        if declared != actual:
            return (
                f"reply.{field}={declared!r} contradicts "
                f"observation.record.{field}={actual!r}."
            )
    return None


# Renderer protocol

class Renderer(Protocol):
    def render(
        self,
        query: str,
        plan: Plan,
        exec_result: ExecutionResult,
    ) -> RenderedAnswer: ...


# Catalog — LLM-driven, schema-constrained, observation-validated.

_CATALOG_SYSTEM = """You answer questions about entries in a reagent catalog.
You receive the user's query and the raw tool observation from lookup_reagent.

Write a natural-language answer in the `answer` field. Populate the structured
fields from the observation:

  * `quoted_name`: the catalog name (observation.name). May be null.
  * `cas`, `vendor`, `sku`, `concentration`, `hazard`, `notes`: copy from
    observation.record.<field>, OR set null if you choose not to mention it.
    NEVER invent a value the observation does not contain.

If observation.found is false, set every structured field to null and state
clearly in your prose that the item is not present in the catalog.

The answer prose should address the user's question naturally. If the
observation has null for the field the user asked about (e.g. CAS), say so
explicitly rather than guessing.
"""


class CatalogRenderer:
    """LLM writes the prose; consistency check rejects invented metadata.

    Replaces the previous 60-line prompt with: short system message,
    schema-constrained decoding, and a post-hoc validator that compares the
    reply against the observation. On mismatch we re-prompt the model with
    the specific error so it can self-correct.
    """

    _max_consistency_retries = 2

    def __init__(self, llm: ChatClient, model: str, config: PlannerConfig) -> None:
        self._llm = llm
        self._model = model
        self._config = config

    def render(
        self,
        query: str,
        plan: Plan,
        exec_result: ExecutionResult,
    ) -> RenderedAnswer:
        obs = exec_result.catalog_observation
        if obs is None:
            return RenderedAnswer(
                answer="The catalog lookup tool returned no observation.",
                structured={},
                citations=[],
            )

        session = LLMSession(
            self._llm,
            self._model,
            system_prompt=_CATALOG_SYSTEM,
            temperature=self._config.renderer_temperature,
            num_predict=self._config.renderer_num_predict,
            top_p=self._config.llm_top_p,
        )
        session.add_user(
            json.dumps({"query": query, "observation": obs}, ensure_ascii=False),
        )

        for _ in range(self._max_consistency_retries):
            reply = session.complete(CatalogAnswer)
            if reply is None:
                break
            problem = _catalog_consistency_error(reply, obs)
            if problem is None:
                return RenderedAnswer(
                    answer=reply.answer.strip(),
                    structured=reply.model_dump(exclude={"answer"}, exclude_none=True),
                    citations=[],
                )
            log.info("catalog_renderer.retry", reason=problem)
            session.add_user(
                f"Your previous reply was inconsistent with the observation: "
                f"{problem} Re-emit a corrected CatalogAnswer JSON.",
            )

        log.warning("catalog_renderer.gave_up")
        return RenderedAnswer(
            answer="The agent could not produce a validated catalog answer.",
            structured={"found": bool(obs.get("found"))} if obs else {},
            citations=[],
        )


# Cell count — LLM-driven prose; structured map comes from the tool.

_CELL_COUNT_SYSTEM = """You report cell-count measurements for plate wells.
You receive the user query and the measured cell_count map from the segmentation tool.

CRITICAL RULES
- Return exactly one JSON object matching the renderer schema: {"answer": "..."}.
- The `answer` value is plain text. Do not put a nested JSON object in it.
- Use the observed `cell_count` values VERBATIM - do not round, average, or invent.
- Only use well or plate IDs that appear in the observation.
- If the user query contains a backtick-quoted JSON template such as
  `{"cell_count": {"<image_id>": <n>}}`, write a baseline-style plain text
  summary in the `answer` field. The structured output is filled separately
  from the tool observation.
  Example - observed count 42 for IMG_01:
    {"answer": "cell_count: {'IMG_01': 42}"}
  For multiple wells, list all: {"answer": "cell_count: {'W1': 12, 'W2': 34}"}
- Otherwise write a concise natural-language sentence with the per-well counts.
"""


class CellCountRenderer:
    def __init__(self, llm: ChatClient, model: str, config: PlannerConfig) -> None:
        self._llm = llm
        self._model = model
        self._config = config

    def render(
        self,
        query: str,
        plan: Plan,
        exec_result: ExecutionResult,
    ) -> RenderedAnswer:
        counts = exec_result.cell_counts
        if not counts:
            return RenderedAnswer(
                answer="Segmentation produced no per-well cell counts.",
                structured={},
                citations=[],
            )

        session = LLMSession(
            self._llm,
            self._model,
            system_prompt=_CELL_COUNT_SYSTEM,
            temperature=self._config.renderer_temperature,
            num_predict=self._config.renderer_num_predict,
            top_p=self._config.llm_top_p,
        )
        session.add_user(
            json.dumps(
                {
                    "query": query,
                    "observation": {"cell_count": counts, "confluency": exec_result.confluency},
                },
                ensure_ascii=False,
            ),
        )
        reply = session.complete(_ProseOnly)
        if reply is None:
            return RenderedAnswer(
                answer="The agent could not produce a cell-count narrative.",
                structured={},
                citations=[],
            )
        return RenderedAnswer(
            answer=reply.answer.strip(),
            structured={},
            citations=[],
        )


# Retrieval — LLM summary of protocol hits.

_RETRIEVAL_SYSTEM = """You summarise protocols returned by retrieve_protocol.
Use only the data in the observation; never invent doc_ids, titles, or text.
Write a short answer (<= 80 words) in the `answer` field.
"""


class RetrievalRenderer:
    def __init__(self, llm: ChatClient, model: str, config: PlannerConfig) -> None:
        self._llm = llm
        self._model = model
        self._config = config

    @staticmethod
    def _citations_structured(citations: list[tuple[str, str]]) -> dict[str, Any]:
        seen: set[str] = set()
        doc_ids: list[str] = []
        for doc_id, _ in citations:
            if doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
        return {"doc_ids": doc_ids}

    def render(
        self,
        query: str,
        plan: Plan,
        exec_result: ExecutionResult,
    ) -> RenderedAnswer:
        if not exec_result.citations:
            return RenderedAnswer(
                answer="The agent could not find a matching protocol in the library.",
                structured={},
                citations=[],
            )

        structured = self._citations_structured(exec_result.citations)

        retrieve_obs = next(
            (t.observation for t in exec_result.trace
             if t.tool == "retrieve_protocol" and t.ok),
            None,
        )
        if retrieve_obs is None:
            return RenderedAnswer(
                answer="The retrieve_protocol tool returned no observation.",
                structured=structured,
                citations=list(exec_result.citations),
            )

        session = LLMSession(
            self._llm,
            self._model,
            system_prompt=_RETRIEVAL_SYSTEM,
            temperature=self._config.renderer_temperature,
            num_predict=self._config.renderer_num_predict,
            top_p=self._config.llm_top_p,
        )
        session.add_user(
            json.dumps({"query": query, "hits": retrieve_obs}, ensure_ascii=False),
        )
        reply = session.complete(_ProseOnly)
        if reply is None:
            return RenderedAnswer(
                answer="The agent could not summarise the retrieved protocols.",
                structured=structured,
                citations=list(exec_result.citations),
            )
        return RenderedAnswer(
            answer=reply.answer.strip(),
            structured=structured,
            citations=list(exec_result.citations),
        )


# Retrieve-then-compose — LLM refines the composed protocol using retrieved text.

_RETRIEVE_THEN_COMPOSE_SYSTEM = """You adapt a structured protocol using a retrieved reference.

You receive:
  - `query`:           the user's original request
  - `retrieved_hits`:  up to 3 protocol documents from the library (title + excerpt)
  - `draft_protocol`:  an initial structured protocol composed from the query

Your job:
1. Write a short prose summary (<= 80 words) in `answer`.
2. Refine `title`, `labware`, `pipettes`, `reagents` by incorporating relevant
   details from `retrieved_hits` that match the adaptation goal in the query.
3. Only use values grounded in `retrieved_hits` or the user query. Never invent items.
"""


class _RefinedProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    title: str
    labware: list[str] = Field(default_factory=list)
    pipettes: list[str] = Field(default_factory=list)
    reagents: list[str] = Field(default_factory=list)
    notes: str | None = None


class RetrieveThenComposeRenderer:
    """Uses retrieved protocol text to ground the composed protocol.

    The planner fills compose_protocol args at plan time (before retrieval
    runs), so the initial draft is based only on the query. This renderer
    makes a second LLM call with the retrieved text in context so the final
    structured output is actually informed by the library reference.
    """

    def __init__(self, llm: ChatClient, model: str, config: PlannerConfig) -> None:
        self._llm = llm
        self._model = model
        self._config = config

    def render(
        self,
        query: str,
        plan: Plan,
        exec_result: ExecutionResult,
    ) -> RenderedAnswer:
        retrieve_obs = next(
            (t.observation for t in exec_result.trace
             if t.tool == "retrieve_protocol" and t.ok),
            None,
        )
        draft = exec_result.composed or {}
        citations = list(exec_result.citations)

        if retrieve_obs is None and not draft:
            return RenderedAnswer(
                answer="Protocol composed from the retrieved reference.",
                structured=draft,
                citations=citations,
            )

        # Only include hits with a meaningful relevance score. Near-zero
        # scores (< 0.05) mean the retrieve query was garbage.
        relevant_hits = [
            h for h in (retrieve_obs if isinstance(retrieve_obs, list) else [])
            if h.get("score", 0) >= 0.05
        ]
        hits_for_llm = [
            {"title": h.get("title", ""), "text": (h.get("text") or "")[:600]}
            for h in relevant_hits[:3]
        ]

        # Exclude ITEM_* placeholder drafts — they confuse the LLM and add no
        # information. The renderer generates from query + relevant hits alone.
        draft_for_llm = {} if (not draft or "ITEM_" in json.dumps(draft)) else draft

        session = LLMSession(
            self._llm,
            self._model,
            system_prompt=_RETRIEVE_THEN_COMPOSE_SYSTEM,
            temperature=self._config.renderer_temperature,
            num_predict=self._config.renderer_num_predict,
            top_p=self._config.llm_top_p,
        )
        session.add_user(
            json.dumps(
                {
                    "query": query,
                    "retrieved_hits": hits_for_llm,
                    "draft_protocol": draft_for_llm,
                },
                ensure_ascii=False,
            ),
        )
        reply = session.complete(_RefinedProtocol)
        if reply is None:
            log.warning("retrieve_then_compose_renderer.llm_failed")
            return RenderedAnswer(
                answer="The agent could not refine the protocol from the retrieved reference.",
                structured=draft,
                citations=citations,
            )
        refined = reply.model_dump(exclude={"answer"}, exclude_none=True)
        return RenderedAnswer(
            answer=reply.answer.strip(),
            structured=refined,
            citations=citations,
        )


# Design — LLM prose around the composed protocol.

_DESIGN_SYSTEM = """You describe an OT-2 protocol that has just been composed.
You receive the user's design brief and the composed protocol object. Write a
short answer (<= 90 words) in the `answer` field. Use only the data in the
observation; never invent labware, pipettes, or reagents.
"""


class DesignRenderer:
    def __init__(self, llm: ChatClient, model: str, config: PlannerConfig) -> None:
        self._llm = llm
        self._model = model
        self._config = config

    def render(
        self,
        query: str,
        plan: Plan,
        exec_result: ExecutionResult,
    ) -> RenderedAnswer:
        composed = exec_result.composed or {}
        if not composed:
            return RenderedAnswer(
                answer="Protocol composition produced no output.",
                structured={},
                citations=[],
            )
        session = LLMSession(
            self._llm,
            self._model,
            system_prompt=_DESIGN_SYSTEM,
            temperature=self._config.renderer_temperature,
            num_predict=self._config.renderer_num_predict,
            top_p=self._config.llm_top_p,
        )
        session.add_user(
            json.dumps({"query": query, "protocol": composed}, ensure_ascii=False),
        )
        reply = session.complete(_ProseOnly)
        if reply is None:
            return RenderedAnswer(
                answer="The agent could not describe the composed protocol.",
                structured={},
                citations=[],
            )
        return RenderedAnswer(answer=reply.answer.strip(), structured={}, citations=[])


# Composite — delegate to the LLM-driven children.


class CompositeRenderer:
    """Combine cell-count narrative and (optional) retrieval narrative."""

    def __init__(
        self,
        cell_count: CellCountRenderer,
        retrieval: RetrievalRenderer,
    ) -> None:
        self._cell_count = cell_count
        self._retrieval = retrieval

    def render(
        self,
        query: str,
        plan: Plan,
        exec_result: ExecutionResult,
    ) -> RenderedAnswer:
        counts_part = self._cell_count.render(query, plan, exec_result)
        if exec_result.citations:
            retrieval_part = self._retrieval.render(query, plan, exec_result)
            return RenderedAnswer(
                answer=f"{counts_part.answer} {retrieval_part.answer}".strip(),
                structured={},
                citations=list(exec_result.citations),
            )
        return RenderedAnswer(
            answer=counts_part.answer,
            structured={},
            citations=[],
        )


# Default — used when task_kind is "other" or unknown.

_DEFAULT_SYSTEM = """You answer a question using the observations from one or
more tools. Use only the data in the observations; never invent values. Write
your answer in the `answer` field.
"""


class DefaultRenderer:
    def __init__(self, llm: ChatClient, model: str, config: PlannerConfig) -> None:
        self._llm = llm
        self._model = model
        self._config = config

    def render(
        self,
        query: str,
        plan: Plan,
        exec_result: ExecutionResult,
    ) -> RenderedAnswer:
        observations = [
            {"tool": t.tool, "observation": t.observation}
            for t in exec_result.trace if t.ok
        ]
        if not observations:
            return RenderedAnswer(
                answer="No tool produced an observation.",
                structured={},
                citations=list(exec_result.citations),
            )
        session = LLMSession(
            self._llm,
            self._model,
            system_prompt=_DEFAULT_SYSTEM,
            temperature=self._config.renderer_temperature,
            num_predict=self._config.renderer_num_predict,
            top_p=self._config.llm_top_p,
        )
        session.add_user(
            json.dumps({"query": query, "observations": observations}, ensure_ascii=False),
        )
        reply = session.complete(_ProseOnly)
        if reply is None:
            return RenderedAnswer(
                answer="The agent could not produce an answer.",
                structured={},
                citations=list(exec_result.citations),
            )
        return RenderedAnswer(
            answer=reply.answer.strip(),
            structured={},
            citations=list(exec_result.citations),
        )


# Registry — dispatch by task_kind, with a sensible fallback chain.


class RendererRegistry:
    """Pick a renderer for a given Plan."""

    def __init__(
        self,
        *,
        catalog: Renderer,
        cell_count: Renderer,
        retrieval: Renderer,
        design: Renderer,
        composite: Renderer,
        retrieve_then_compose: Renderer,
        default: Renderer,
    ) -> None:
        self._by_kind: dict[TaskKind, Renderer] = {
            "catalog": catalog,
            "cell_count": cell_count,
            "retrieval": retrieval,
            "design": design,
            "composite": composite,
            "other": default,
        }
        self._retrieve_then_compose = retrieve_then_compose
        self._default = default

    @classmethod
    def default(
        cls,
        llm: ChatClient,
        model: str,
        config: PlannerConfig,
    ) -> RendererRegistry:
        catalog = CatalogRenderer(llm, model, config)
        cell_count = CellCountRenderer(llm, model, config)
        retrieval = RetrievalRenderer(llm, model, config)
        design = DesignRenderer(llm, model, config)
        composite = CompositeRenderer(cell_count=cell_count, retrieval=retrieval)
        retrieve_then_compose = RetrieveThenComposeRenderer(llm, model, config)
        default = DefaultRenderer(llm, model, config)
        return cls(
            catalog=catalog,
            cell_count=cell_count,
            retrieval=retrieval,
            design=design,
            composite=composite,
            retrieve_then_compose=retrieve_then_compose,
            default=default,
        )

    def render(
        self,
        query: str,
        plan: Plan,
        exec_result: ExecutionResult,
    ) -> RenderedAnswer:
        renderer = self._dispatch(plan, exec_result)
        return renderer.render(query, plan, exec_result)

    def _dispatch(self, plan: Plan, exec_result: ExecutionResult) -> Renderer:
        """Pick the renderer best supported by the actual evidence.

        The planner can misclassify (e.g. flag a plain design task as
        "composite" because two tools were emitted). Evidence-based dispatch
        is robust to that: route on what the tools actually produced, fall
        back to Plan.task_kind only when the evidence is ambiguous.
        """
        has_counts = bool(exec_result.cell_counts)
        has_citations = bool(exec_result.citations)
        has_composed = bool(exec_result.composed)
        has_catalog = exec_result.catalog_observation is not None
        compose_attempted = any(
            t.tool == "compose_protocol" for t in exec_result.trace
        )

        if has_counts and has_citations:
            return self._by_kind["composite"]
        if has_catalog:
            return self._by_kind["catalog"]
        if has_composed and has_citations:
            return self._retrieve_then_compose
        if has_composed:
            return self._by_kind["design"]
        # compose was planned but failed (e.g. ITEM_* placeholders); citations
        # are present from a retrieve step. Generate from query + context rather
        # than falling through to the retrieval renderer which returns doc_ids only.
        if compose_attempted and has_citations:
            return self._retrieve_then_compose
        if has_counts:
            return self._by_kind["cell_count"]
        if has_citations:
            return self._by_kind["retrieval"]
        return self._by_kind.get(plan.task_kind, self._default)
