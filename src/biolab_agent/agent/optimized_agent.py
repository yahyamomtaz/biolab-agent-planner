"""Optimized benchmark agent.

Catalog tasks use a structured-decision schema. The LLM declares the
catalog name and each metadata field (cas, vendor, sku, concentration,
hazard, notes) as typed structured output, and the agent validates each
declared field against the ``lookup_reagent`` observation. If any field
disagrees, the agent re-prompts with the observation. ``final.answer`` is
the LLM's prose, passed through verbatim to ``AgentResult.answer``.

Routing is behavior-based: schema validation kicks in whenever the LLM
actually called ``lookup_reagent`` during the run. There is no query
pattern matching.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from biolab_agent.agent.baseline import (
    BaselineAgent,
    _extract_json,
    _serialize,
    _strip_heavy,
    _MAX_CITATIONS,
    _MAX_ITERATIONS,
)
from biolab_agent.schemas import AgentResult, ToolTrace
from biolab_agent.tools import TOOL_IMPLS


# ---------------------------------------------------------------------------
# Catalog prompt rules
# ---------------------------------------------------------------------------

_CATALOG_METADATA_FIELDS = ("cas", "vendor", "sku", "concentration", "hazard", "notes")

_CATALOG_PROMPT_RULES = """
CATALOG LOOKUP RULES

Step 1. If the question is about an item in the reagent catalog, call
        lookup_reagent ONCE with the item name from the question.
Step 2. After receiving the observation, write your final answer. Do not
        call lookup_reagent again.

lookup_reagent observation shape:
  {"requested_name": "<item>", "found": true/false,
   "name": "<exact catalog name or null>",
   "record": {"name": ..., "cas": ..., "vendor": ..., "sku": ...,
              "concentration": ..., "hazard": ..., "notes": ...} or null}

If you called lookup_reagent, your final object MUST be:
  {"final": {
     "found": true/false,
     "quoted_name":   "<observation.name or null>",
     "cas":           "<observation.record.cas or null>",
     "vendor":        "<observation.record.vendor or null>",
     "sku":           "<observation.record.sku or null>",
     "concentration": "<observation.record.concentration or null>",
     "hazard":        "<observation.record.hazard or null>",
     "notes":         "<observation.record.notes or null>",
     "answer":        "<your natural-language answer>"
   },
   "structured": {},
   "citations": []}

- Copy every field byte-for-byte from observation / observation.record.
  Use null whenever the observation has null. Do not invent values.
- When found=true, quote the exact catalog name in the prose.
- Refer to "the provided reagent catalog" so the answer is unambiguous.
- For yes/no questions, begin "answer" with "Yes." (found=true) or
  "No." (found=false). The rest of the sentence is yours.

SELF-CONSISTENCY RULE (CRITICAL):
Your prose "answer" must be consistent with the structured fields you just
declared. If you declared a field as null, the prose must NOT assert any
value for that field. Do not draw on outside knowledge; the observation is
the only source of truth. The reagent catalog in this task is intentionally
sparse — most rows have null cas/vendor/sku/concentration/hazard and that
is the correct answer.

WRONG (self-contradiction — never do this):
  declared:  {"cas": null, "vendor": null, "sku": null, ...}
  prose:     "Nuclease-Free Water with CAS 7694-01-4 from Thermo Fisher
              Scientific, SKU 2891811."
  Reason: the structured fields say there is no CAS / vendor / SKU, but
  the prose invents them from pretraining.

CORRECT (prose omits null fields entirely):
  declared:  {"cas": null, "vendor": null, "sku": null,
              "notes": "Used in protocol 0e525c (droplet digital PCR Prep)"}
  prose:     "Nuclease-Free Water is listed in the provided reagent
              catalog. The catalog does not list CAS, vendor, or SKU
              information for this entry. Notes: Used in protocol 0e525c
              (droplet digital PCR Prep)."

If the catalog has no CAS/vendor/SKU/concentration/hazard for the item,
either omit that field name from the prose entirely, or say explicitly
that the catalog does not list it. Never assert a value that does not
appear in the observation.
"""


# ---------------------------------------------------------------------------
# Schema + validator
# ---------------------------------------------------------------------------

class CatalogFinal(BaseModel):
    found: bool
    quoted_name: str | None = None
    cas: str | None = None
    vendor: str | None = None
    sku: str | None = None
    concentration: str | None = None
    hazard: str | None = None
    notes: str | None = None
    answer: str


def _catalog_consistency_error(
    final: CatalogFinal,
    observation: dict[str, Any],
) -> str | None:
    """Compare each LLM-declared field to the lookup observation.

    Returns a problem description for the first disagreement, otherwise
    ``None``. ``final.answer`` is not inspected — the prose is the model's
    responsibility.
    """
    tool_found = bool(observation.get("found"))
    tool_name = observation.get("name")
    record = observation.get("record") if isinstance(observation.get("record"), dict) else {}

    if final.found != tool_found:
        return (
            f"final.found={final.found} contradicts lookup_reagent observation "
            f"found={tool_found}."
        )

    if tool_found:
        if not isinstance(tool_name, str) or not tool_name:
            return "lookup_reagent returned found=true without an exact name."
        if final.quoted_name != tool_name:
            return (
                f"final.quoted_name must equal observation.name exactly "
                f"({tool_name!r})."
            )
        for field in _CATALOG_METADATA_FIELDS:
            expected = record.get(field)
            declared = getattr(final, field)
            if declared != expected:
                return (
                    f"final.{field}={declared!r} must equal "
                    f"observation.record.{field}={expected!r}. "
                    "Copy fields from the observation; do not invent metadata."
                )
        return None

    if final.quoted_name is not None:
        return "final.quoted_name must be null when observation.found is false."
    for field in _CATALOG_METADATA_FIELDS:
        if getattr(final, field) is not None:
            return f"final.{field} must be null when observation.found is false."
    return None


def _catalog_retry_prompt(problem: str, observation: dict[str, Any]) -> str:
    observation_text = json.dumps(observation, ensure_ascii=False)
    return (
        f"Your final object failed structured-decision validation: {problem}\n\n"
        f"lookup_reagent observation:\n{observation_text}\n\n"
        "Return exactly one JSON object with this shape:\n"
        '{"final": {"found": <bool>, "quoted_name": "<observation.name or null>", '
        '"cas": "<record.cas or null>", "vendor": "<record.vendor or null>", '
        '"sku": "<record.sku or null>", "concentration": "<record.concentration or null>", '
        '"hazard": "<record.hazard or null>", "notes": "<record.notes or null>", '
        '"answer": "<your prose>"}, "structured": {}, "citations": []}\n'
        "Copy every field from the observation byte-for-byte. Your prose may "
        "only mention metadata you have declared with a non-null value."
    )


# ---------------------------------------------------------------------------
# Structured output helpers
# ---------------------------------------------------------------------------

def _catalog_structured_output(item: str, lookup: ToolTrace) -> dict[str, Any]:
    observation = lookup.observation
    base: dict[str, Any] = {
        "requested_item": item,
        "lookup_name": str(lookup.args.get("name") or item),
        "found": False,
        "quoted_name": None,
        "record": None,
    }
    if isinstance(observation, dict):
        record = observation.get("record")
        if observation.get("found") and isinstance(record, dict):
            base.update({
                "found": True,
                "quoted_name": record.get("name") or observation.get("name"),
                "record": {
                    key: record.get(key)
                    for key in ("name", *_CATALOG_METADATA_FIELDS)
                },
            })
    return {"catalog_lookup": base}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class OptimizedAgent(BaselineAgent):
    """Baseline agent with a structured-decision schema on catalog answers."""

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        super().__init__(config)
        self._system_prompt += "\n\n" + _CATALOG_PROMPT_RULES

        # Swap in the optimized lookup tool (abbrev_map + normalized index)
        import biolab_agent.tools as _tool_registry
        from biolab_agent.tools.lookup_reagents import lookup_reagent as _opt_lookup

        _tool_registry.TOOL_IMPLS["lookup_reagent"] = _opt_lookup

    def run(
        self,
        query: str,
        image_ids: list[str] | None = None,
    ) -> AgentResult:
        start = time.perf_counter()
        trace: list[ToolTrace] = []
        confluency: dict[str, float] = {}
        cell_counts: dict[str, int] = {}
        citations: list[tuple[str, str]] = []
        structured: dict[str, Any] | None = None
        final_answer = ""

        catalog_item: str | None = None
        catalog_observation: dict | None = None
        catalog_lookup_trace: ToolTrace | None = None
        catalog_tool_called: bool = False

        user_content = query
        if image_ids:
            user_content += f"\n\nAvailable image_ids: {list(image_ids)}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        max_iterations = _MAX_ITERATIONS + 3
        for step in range(max_iterations):
            resp = self._llm.chat(
                model=self.config.llm_model,
                messages=messages,
                options={"temperature": 0.1, "num_predict": 600, "top_p": 0.9},
            )
            assistant_raw = resp["message"]["content"]
            messages.append({"role": "assistant", "content": assistant_raw})
            parsed = _extract_json(assistant_raw)
            if parsed is None:
                final_answer = assistant_raw.strip() or "Agent produced no parseable output."
                break

            if "final" in parsed:
                raw_final = parsed.get("final", "")

                # Schema validation gated on actual tool behavior, not on
                # query pattern matching.
                if catalog_tool_called and isinstance(catalog_observation, dict):
                    catalog_final: CatalogFinal | None
                    try:
                        catalog_final = CatalogFinal.model_validate(raw_final)
                    except ValidationError as exc:
                        catalog_final = None
                        problem: str | None = (
                            f"final must match CatalogFinal schema: {exc.errors()}"
                        )
                    else:
                        problem = _catalog_consistency_error(
                            catalog_final, catalog_observation
                        )

                    if problem and step < max_iterations - 1:
                        messages.append({
                            "role": "user",
                            "content": _catalog_retry_prompt(problem, catalog_observation),
                        })
                        continue

                    if catalog_final is not None:
                        # LLM's prose goes through unchanged.
                        final_answer = catalog_final.answer.strip()
                        structured = dict(structured or {})
                        structured["catalog_lookup"] = {
                            "found": catalog_final.found,
                            "quoted_name": catalog_final.quoted_name,
                            **{
                                field: getattr(catalog_final, field)
                                for field in _CATALOG_METADATA_FIELDS
                            },
                        }
                    else:
                        final_answer = str(raw_final).strip()
                else:
                    final_answer = str(raw_final).strip()

                if isinstance(parsed.get("structured"), dict):
                    structured = {**dict(parsed["structured"]), **(structured or {})}
                raw_cites = parsed.get("citations") or []
                for c in raw_cites:
                    if isinstance(c, list | tuple) and len(c) >= 2:
                        citations.append((str(c[0]), str(c[1])))
                break

            tool = parsed.get("tool")
            args = parsed.get("arguments") or parsed.get("args") or {}
            if tool == "lookup_reagent" and catalog_tool_called:
                messages.append({
                    "role": "user",
                    "content": (
                        "lookup_reagent has already been called. Emit the final "
                        "JSON answer from the existing observation."
                    ),
                })
                continue
            if not tool or tool not in TOOL_IMPLS:
                trace.append(ToolTrace(
                    step=step, tool=str(tool or "<unknown>"), args=args,
                    ok=False, error=f"Unknown tool {tool!r}",
                ))
                messages.append({"role": "tool", "content": json.dumps(
                    {"error": f"Unknown tool {tool!r}. Valid tools: {list(TOOL_IMPLS)}."}
                )})
                continue

            t0 = time.perf_counter()
            try:
                result = TOOL_IMPLS[tool](**args)
                observation = _serialize(result)
                trace.append(ToolTrace(
                    step=step, tool=tool, args=args, ok=True,
                    observation=observation,
                    elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                ))
            except Exception as exc:
                trace.append(ToolTrace(
                    step=step, tool=tool, args=args, ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                ))
                messages.append({"role": "tool", "content": json.dumps(
                    {"tool": tool, "error": str(exc)}
                )})
                continue

            # --- Tool aggregation (mirrors baseline segment_wells pattern) ---

            if tool == "segment_wells":
                for mask in observation.get("masks", []):
                    wid = args.get("image_id") or mask.get("well_id")
                    if wid:
                        confluency[str(wid)] = float(mask.get("confluency_pct", 0.0))
                        cc = mask.get("cell_count")
                        if cc is not None:
                            cell_counts[str(wid)] = int(cc)

            elif tool == "retrieve_protocol":
                for hit in observation[:_MAX_CITATIONS]:
                    citations.append((str(hit.get("doc_id", "")), str(hit.get("chunk_id", ""))))

            elif tool == "compose_protocol":
                structured = {k: v for k, v in observation.items() if v is not None}

            elif tool == "lookup_reagent":
                catalog_item = str(args.get("name") or "")
                catalog_observation = observation if isinstance(observation, dict) else None
                catalog_tool_called = True
                catalog_lookup_trace = trace[-1]

            light_observation = _strip_heavy(observation)
            messages.append({"role": "tool", "content": json.dumps(
                {"tool": tool, "observation": light_observation}
            )[:6000]})

        else:
            final_answer = (
                final_answer or "Max iterations reached before the agent emitted a final answer."
            )

        # Attach the lookup trace into structured for the harness/trace viewer.
        if catalog_tool_called and catalog_lookup_trace:
            structured = dict(structured or {})
            structured.setdefault(
                "catalog_lookup_trace",
                _catalog_structured_output(catalog_item or "", catalog_lookup_trace)[
                    "catalog_lookup"
                ],
            )

        # Baseline cell_count/confluency override (unchanged)
        if confluency or cell_counts:
            structured = dict(structured or {})
            if confluency:
                structured["confluency"] = {**structured.get("confluency", {}), **confluency}
            if cell_counts:
                structured["cell_count"] = {**structured.get("cell_count", {}), **cell_counts}

        if self.config.lora_adapter and self._is_protocol_design(query):
            polished = self._polish_with_adapter(query)
            if polished:
                structured = {**(structured or {}), **polished}

        seen: set[tuple[str, str]] = set()
        unique_cites: list[tuple[str, str]] = []
        for c in citations:
            if c not in seen:
                seen.add(c)
                unique_cites.append(c)

        return AgentResult(
            query=query,
            answer=final_answer or "(empty)",
            structured=structured,
            trace=trace,
            model=self.config.llm_model,
            adapter=self.config.lora_adapter,
            elapsed_ms=round((time.perf_counter() - start) * 1000.0, 2),
            citations=unique_cites[:_MAX_CITATIONS],
        )
