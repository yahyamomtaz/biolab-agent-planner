"""Planner / TaskFetcher / Executor / Renderer agent components.

Layout follows the four-component decomposition:

    Planner       -->  TaskFetcher  -->  Executor  -->  Renderer
       |                    |               |              |
       |                    |               |              +-- per task-kind
       |                    |               |                  (deterministic for
       |                    |               |                  catalog / cell_count;
       |                    |               |                  LLM-backed for
       |                    |               |                  retrieval / design;
       |                    |               |                  Joiner-style finalizer)
       |                    |               |
       |                    |               +-- parallel tool dispatch over
       |                    |                   ready tasks (ThreadPoolExecutor,
       |                    |                   width = max_parallel_calls)
       |                    |
       |                    +-- placeholder DSL substitution + DAG bookkeeping
       |                        (${steps[N].observation.path} -> resolved value)
       |
       +-- produces a typed Plan (task_kind + ordered steps with depends_on)
           via a single schema-constrained LLM call

Wiring lives in `biolab_agent.agent.planner.PlannerAgent`. Each module
in this package is independently importable and unit-testable.
"""

from biolab_agent.agent.compiler.config import PlannerConfig
from biolab_agent.agent.compiler.executor import ExecutionResult, Executor
from biolab_agent.agent.compiler.fetcher import ReadyTask, TaskFetcher
from biolab_agent.agent.compiler.joiner import Joiner, JoinerDecision
from biolab_agent.agent.compiler.llm_session import LLMSession
from biolab_agent.agent.compiler.plan import Planner
from biolab_agent.agent.compiler.renderers import RendererRegistry
from biolab_agent.agent.compiler.tool_registry import ToolRegistry
from biolab_agent.agent.compiler.types import Plan, PlanStep, RenderedAnswer

__all__ = [
    "ExecutionResult",
    "Executor",
    "Joiner",
    "JoinerDecision",
    "LLMSession",
    "Plan",
    "PlanStep",
    "Planner",
    "PlannerConfig",
    "ReadyTask",
    "RenderedAnswer",
    "RendererRegistry",
    "TaskFetcher",
    "ToolRegistry",
]
