"""Executor — LLMCompiler's parallel tool runner.

The Executor receives ready-to-dispatch tasks from
the Task Fetching Unit and runs them.

Concretely: a single ``PlanStep`` with ``foreach_image_id=True`` fans
out (in the TaskFetcher) into N ReadyTasks, one per ``image_id``. The
Executor runs them through a bounded ``ThreadPoolExecutor``. Independent
steps emitted by future Planner versions would dispatch the same way.

"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from biolab_agent.agent.compiler.config import PlannerConfig
from biolab_agent.agent.compiler.fetcher import ReadyTask, TaskFetcher
from biolab_agent.agent.compiler.tool_registry import ToolRegistry
from biolab_agent.agent.compiler.types import Plan, PlanStep
from biolab_agent.logging import get_logger
from biolab_agent.schemas import ProtocolHit, ToolTrace, WellMasks

log = get_logger(__name__)


def _serialize(obj: Any) -> Any:
    """Tool outputs may be Pydantic models, dicts, lists, or primitives.

    Always emit the full ``model_dump()``. The trace observation is the
    source of truth — both for re-validation by the absorbers
    (``WellMasks``, ``ProtocolHit``) and for the renderer's LLM input.
    Heavy-field stripping that ``for_llm()`` provides is for the **LLM
    consumer**, not for the internal aggregator; calling it here once
    broke segmentation aggregation because the stripped dict couldn't
    round-trip through the strict Pydantic schema.
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


@dataclass(slots=True)
class ExecutionResult:
    """Typed bag the Executor returns to the Renderer.

    Each field has one owner (one tool that writes it) so the contract is
    obvious. Adding a new tool means extending this dataclass — a
    deliberate trade-off: explicit > magic.
    """

    trace: list[ToolTrace] = field(default_factory=list)
    confluency: dict[str, float] = field(default_factory=dict)
    cell_counts: dict[str, int] = field(default_factory=dict)
    citations: list[tuple[str, str]] = field(default_factory=list)
    composed: dict[str, Any] | None = None
    catalog_observation: dict[str, Any] | None = None
    observations: dict[int, Any] = field(default_factory=dict)


class Executor:
    """Parallel-capable tool dispatcher.

    Public entry: ``run(plan, image_ids) -> ExecutionResult``. Internally
    delegates variable substitution and DAG bookkeeping to a
    ``TaskFetcher`` and dispatches each step's ready tasks through a
    thread pool whose width is capped by
    ``PlannerConfig.max_parallel_calls``.

    Default cap is intentionally low: ``segment_wells`` is GPU-bound on
    an 8 GB budget, and concurrent SAM calls can multiply VRAM pressure.
    Raise the cap when the workload is I/O-bound (retrieval, REST tools).
    """

    def __init__(self, tools: ToolRegistry, config: PlannerConfig) -> None:
        self._tools = tools
        self._config = config

    # run loop

    def run(self, plan: Plan, image_ids: list[str] | None) -> ExecutionResult:
        result = ExecutionResult()
        fetcher = TaskFetcher()

        for step in plan.steps[: self._config.max_plan_steps]:
            tasks = fetcher.tasks_for(step, image_ids)
            traces = self._dispatch(tasks, len(result.trace))

            for trace in traces:
                result.trace.append(trace)
                if trace.ok and trace.observation is not None:
                    self._absorb(result, step, trace.observation)

            # Record the representative observation for downstream
            # placeholder resolution. For foreach steps we pick the last
            # successful run; for single-call steps it's just the one.
            last_ok = next(
                (
                    t.observation
                    for t in reversed(traces)
                    if t.ok and t.observation is not None
                ),
                None,
            )
            fetcher.record(step.id, last_ok)

        result.observations = fetcher.observations()
        return result

    # dispatch

    def _dispatch(
        self,
        tasks: list[ReadyTask],
        trace_offset: int,
    ) -> list[ToolTrace]:
        """Run all ready tasks, in parallel when allowed by the config.

        With a single task or ``max_parallel_calls == 1`` we fall through
        to a plain sequential loop — no thread overhead, behavior
        identical to the pre-split Executor.
        """
        n = len(tasks)
        workers = min(n, max(1, self._config.max_parallel_calls))
        if workers == 1:
            return [
                self._call_tool(task, trace_offset + i)
                for i, task in enumerate(tasks)
            ]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            indexed = list(enumerate(tasks))
            return list(
                pool.map(
                    lambda it: self._call_tool(it[1], trace_offset + it[0]),
                    indexed,
                )
            )

    def _call_tool(self, task: ReadyTask, step_index: int) -> ToolTrace:
        tool_name = task.step.tool
        t0 = time.perf_counter()
        try:
            impl = self._tools.get(tool_name)
        except KeyError as exc:
            return ToolTrace(
                step=step_index, tool=tool_name, args=task.args,
                ok=False, error=str(exc),
            )
        try:
            result = impl(**task.args)
        except (TypeError, ValueError, RuntimeError, FileNotFoundError) as exc:
            log.warning(
                "executor.tool_failed",
                tool=tool_name, step=step_index, error=str(exc),
            )
            return ToolTrace(
                step=step_index, tool=tool_name, args=task.args, ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
        return ToolTrace(
            step=step_index, tool=tool_name, args=task.args, ok=True,
            observation=_serialize(result),
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

    # absorption

    def _absorb_segment(self, result: ExecutionResult, obs: Any) -> None:
        try:
            wm = WellMasks.model_validate(obs)
        except ValidationError as exc:
            log.warning("executor.segment_validate_failed", error=str(exc))
            return
        for mask in wm.masks:
            result.confluency[mask.well_id] = mask.confluency_pct
            if mask.cell_count is not None:
                result.cell_counts[mask.well_id] = mask.cell_count

    def _absorb_retrieve(self, result: ExecutionResult, obs: Any) -> None:
        if not isinstance(obs, list):
            return
        for raw_hit in obs[: self._config.max_citations]:
            try:
                hit = ProtocolHit.model_validate(raw_hit)
            except ValidationError as exc:
                log.warning("executor.retrieve_validate_failed", error=str(exc))
                continue
            result.citations.append((hit.doc_id, hit.chunk_id))

    def _absorb(self, result: ExecutionResult, step: PlanStep, obs: Any) -> None:
        if step.tool == "segment_wells":
            self._absorb_segment(result, obs)
        elif step.tool == "retrieve_protocol":
            self._absorb_retrieve(result, obs)
        elif step.tool == "compose_protocol" and isinstance(obs, dict):
            result.composed = obs
        elif step.tool == "lookup_reagent" and isinstance(obs, dict):
            result.catalog_observation = obs
