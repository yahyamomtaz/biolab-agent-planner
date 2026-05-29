"""Planner/Executor agent.

Four-component decomposition. Each component is its own module under
``biolab_agent.agent.compiler``:

    Planner      — produces a typed Plan (task_kind + ordered steps with
                   depends_on) via one schema-constrained LLM call.
    TaskFetcher  — substitutes ${steps[N].observation.path} variables,
                   expands foreach_image_id fan-outs, and emits ReadyTasks
                   to the Executor.
    Executor     — runs ReadyTasks in parallel (capped by
                   ``max_parallel_calls``) and absorbs observations into
                   a typed ExecutionResult bag.
    Renderers    — turn the ExecutionResult into a user-facing answer.
                   Doubles as a Joiner-style finalizer for the design and
                   retrieval task kinds (LLM step that decides the final
                   form rather than just templating).

"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from biolab_agent.agent.base import AgentConfig, BaseAgent
from biolab_agent.agent.compiler import (
    ExecutionResult,
    Executor,
    Joiner,
    Planner,
    PlannerConfig,
    RendererRegistry,
    ToolRegistry,
)
from biolab_agent.llm import get_client
from biolab_agent.llm.base import ChatClient
from biolab_agent.logging import get_logger
from biolab_agent.schemas import AgentResult

log = get_logger(__name__)


class PlannerAgent(BaseAgent):
    """Wires Planner -> Executor -> Renderer.

    Components are constructed once at agent startup and reused across queries.
    Each component is independently testable; this class exists only to
    sequence them and apply the tool-truth merge to ``AgentResult.structured``.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        llm: ChatClient | None = None,
        tools: ToolRegistry | None = None,
        planner_config: PlannerConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._llm = llm or get_client()
        self._tools = tools or ToolRegistry.default()
        self._planner_config = planner_config or PlannerConfig()

        self._planner = Planner(
            self._llm, config.llm_model, self._tools, self._planner_config,
        )
        self._executor = Executor(self._tools, self._planner_config)
        self._renderers = RendererRegistry.default(
            self._llm, config.llm_model, self._planner_config,
        )
        self._joiner = Joiner()

    #  entry point

    def run(self, query: str, image_ids: list[str] | None = None) -> AgentResult:
        start = time.perf_counter()

        plan = self._planner.plan(query, image_ids)
        if plan is None:
            log.warning("planner.plan_failed", query=query[:80])
            return self._empty_result(query, start, "Planner failed to produce a valid plan.")

        # LLMCompiler plan / execute / joiner loop. The Joiner decides
        # after each Executor pass whether the evidence is enough to
        # finalize or whether the Planner should be re-invoked with a
        # failure summary. Capped by ``max_replans`` so we don't
        # double-cost the happy path.
        exec_result: ExecutionResult
        for replan_attempt in range(self._planner_config.max_replans + 1):
            with self._evict_planner_llm():
                exec_result = self._executor.run(plan, image_ids)
            decision = self._joiner.decide(plan, exec_result)
            if decision.action == "finalize":
                break
            if replan_attempt >= self._planner_config.max_replans:
                log.warning(
                    "planner.replan_budget_exhausted",
                    feedback=decision.feedback,
                    task_kind=plan.task_kind,
                )
                break
            log.info(
                "planner.replan",
                attempt=replan_attempt + 1,
                feedback=decision.feedback,
                task_kind=plan.task_kind,
            )
            new_plan = self._planner.plan(
                query, image_ids, feedback=decision.feedback,
            )
            if new_plan is None:
                log.warning("planner.replan_plan_failed")
                break
            plan = new_plan

        rendered = self._renderers.render(query, plan, exec_result)

        structured = self._merge_structured(rendered.structured, exec_result)
        citations = self._merge_citations(rendered.citations, exec_result)

        return AgentResult(
            query=query,
            answer=rendered.answer or "(empty)",
            structured=structured or None,
            trace=exec_result.trace,
            model=self.config.llm_model,
            adapter=self.config.lora_adapter,
            elapsed_ms=round((time.perf_counter() - start) * 1000.0, 2),
            citations=citations[: self._planner_config.max_citations],
        )

    # post-render merging

    @staticmethod
    def _merge_structured(
        rendered: dict[str, Any],
        exec_result: ExecutionResult,
    ) -> dict[str, Any]:
        """Tool aggregates always win over renderer values.

        Cell counts and confluency come from segment_wells; compose_protocol
        output is the source of truth for design tasks. The renderer can add
        narrative context but cannot overwrite measured numbers.
        """
        structured = dict(rendered or {})
        if exec_result.cell_counts:
            structured["cell_count"] = {
                **structured.get("cell_count", {}),
                **exec_result.cell_counts,
            }
        if exec_result.composed:
            has_renderer_protocol = any(
                k in structured for k in ("title", "labware", "pipettes", "reagents")
            )
            if not has_renderer_protocol:
                structured.update(
                    {k: v for k, v in exec_result.composed.items() if v is not None},
                )
        return structured

    @staticmethod
    def _merge_citations(
        rendered_citations: list[tuple[str, str]],
        exec_result: ExecutionResult,
    ) -> list[tuple[str, str]]:
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for pair in list(exec_result.citations) + list(rendered_citations):
            if pair in seen:
                continue
            seen.add(pair)
            out.append(pair)
        return out

    # VRAM management

    @contextmanager
    def _evict_planner_llm(self) -> Iterator[None]:
        """Free the planner LLM's VRAM before the executor calls tools.

        Tools like ``segment_wells`` load their own GPU tensors (SAM ~734 MiB
        in fp32) and OOM if MedGemma-4B (~5 GB) is still pinned by Ollama.
        We hit the eviction from two angles because either alone is unreliable:

          * Ollama HTTP ``keep_alive=0`` — triggers Ollama's own unload path.
          * ``ollama.generate(keep_alive=0)`` — same effect via the Python
            client, in case the HTTP host is misconfigured.
          * ``torch.cuda.empty_cache()`` + ``ipc_collect()`` — reclaim
            fragmentation inside our process.

        After the executor returns we ask Ollama to re-warm the model so the
        renderer's first call doesn't pay the cold-load latency.
        """
        self._unload_llm()
        try:
            yield
        finally:
            self._reload_llm()

    def _unload_llm(self) -> None:
        try:
            import httpx
            try:
                httpx.post(
                    f"{self.config.ollama_host}/api/generate",
                    json={"model": self.config.llm_model, "keep_alive": 0},
                    timeout=3.0,
                )
            except httpx.RequestError as exc:
                log.debug("planner.ollama_http_evict_failed", error=str(exc))
        except ImportError:
            pass

        try:
            import ollama
            ollama.generate(model=self.config.llm_model, prompt="", keep_alive=0)
        except (ImportError, ConnectionError, RuntimeError) as exc:
            log.debug("planner.ollama_evict_failed", error=str(exc))

        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
        except ImportError:
            pass
        except RuntimeError as exc:
            log.debug("planner.torch_evict_failed", error=str(exc))

    def _reload_llm(self) -> None:
        """Best-effort warm of the Ollama model so the renderer doesn't pay
        the full cold-load latency on its first call."""
        try:
            import ollama
            ollama.generate(model=self.config.llm_model, prompt="", keep_alive="5m")
        except (ImportError, ConnectionError, RuntimeError) as exc:
            log.debug("planner.ollama_warm_failed", error=str(exc))

    # helpers

    def _empty_result(self, query: str, start: float, answer: str) -> AgentResult:
        return AgentResult(
            query=query,
            answer=answer,
            structured=None,
            trace=[],
            model=self.config.llm_model,
            adapter=self.config.lora_adapter,
            elapsed_ms=round((time.perf_counter() - start) * 1000.0, 2),
            citations=[],
        )
