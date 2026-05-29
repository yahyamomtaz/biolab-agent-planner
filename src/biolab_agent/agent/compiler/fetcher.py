"""TaskFetcher

  1. Receive tasks planned by the Planner.
  2. Substitute ``$N`` variable references once their producing tasks
    have resolved.
  3. Hand the Executor a set of tasks that are ready to dispatch.

No LLM is invoked here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from biolab_agent.agent.compiler.types import PlanStep


_PLACEHOLDER_RE = re.compile(r"\$\{steps\[(\d+)\]\.observation\.?([^\}]*)\}")
_INDEX_RE = re.compile(r"^\[(\d+)\]$")


def _resolve_path(val: Any, path: str) -> Any:
    if not path:
        return val
    for part in path.split("."):
        if not part:
            continue
        if (m := _INDEX_RE.match(part)) and isinstance(val, list):
            val = val[int(m.group(1))]
        elif isinstance(val, dict):
            val = val.get(part)
        else:
            return None
        if val is None:
            return None
    return val


@dataclass(slots=True, frozen=True)
class ReadyTask:
    """One concrete tool call with all variables substituted.

    A single ``PlanStep`` may produce multiple ReadyTasks (when
    ``foreach_image_id`` fans out over image_ids); those tasks are
    independent and may be executed in parallel.
    """

    step: PlanStep
    args: dict[str, Any]


class TaskFetcher:
    """Resolves Plan steps into ready-to-execute tool calls.

    Stateful across a single Plan run: cross-step variable substitution
    requires remembering what each completed step observed. The Executor
    feeds observations back in via ``record`` after each step finishes.
    """

    def __init__(self) -> None:
        self._observations: dict[int, Any] = {}

    def record(self, step_id: int, observation: Any) -> None:
        """Record an observation so downstream steps can reference it."""
        self._observations[step_id] = observation

    def observations(self) -> dict[int, Any]:
        """Snapshot of observations recorded so far (for ExecutionResult)."""
        return dict(self._observations)

    def tasks_for(
        self,
        step: PlanStep,
        image_ids: list[str] | None,
    ) -> list[ReadyTask]:
        """Yield all parallel-executable tasks for ``step``.

        With ``foreach_image_id=True`` and non-empty ``image_ids``,
        returns one ReadyTask per image — independent calls suitable for
        parallel dispatch. Otherwise returns a single ReadyTask.
        """
        resolved = self._resolve_args(step)
        if step.foreach_image_id and image_ids:
            return [
                ReadyTask(step=step, args={**resolved, "image_id": img})
                for img in image_ids
            ]
        return [ReadyTask(step=step, args=resolved)]

    def _resolve_args(self, step: PlanStep) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for key, val in step.args.items():
            if not (isinstance(val, str) and "${" in val):
                resolved[key] = val
                continue

            def _sub(m: re.Match[str]) -> str:
                obs = self._observations.get(int(m.group(1)) + 1)
                if obs is None:
                    return ""
                got = _resolve_path(obs, m.group(2))
                return "" if got is None else str(got)

            resolved[key] = _PLACEHOLDER_RE.sub(_sub, val)
        return resolved
