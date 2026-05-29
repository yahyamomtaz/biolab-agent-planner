"""Benchmark runner  -  runs the configured agent against a task set.

Dispatches on ``task.kind`` to the scoring functions in :mod:`eval.metrics`.

Usage::

    biolab-bench --queries data/queries_public.yaml --report artifacts/bench_report.json

Or programmatically::

    from eval.harness import run_benchmark
    report = run_benchmark(Path("data/queries_public.yaml"))
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from biolab_agent.agent import load_agent
from biolab_agent.agent.base import BaseAgent
from biolab_agent.schemas import AgentResult
from eval import metrics as m

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Task:
    id: str
    kind: str
    query: str
    image_ids: list[str]
    scoring: dict[str, Any]
    pass_threshold: float = 0.7
    weight: float = 1.0


def load_tasks(path: Path) -> list[Task]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tasks = []
    for entry in raw.get("queries", []):
        tasks.append(
            Task(
                id=entry["id"],
                kind=entry["kind"],
                query=entry["query"],
                image_ids=entry.get("image_ids", []),
                scoring=entry.get("scoring", {}),
                pass_threshold=float(entry.get("pass_threshold", 0.7)),
                weight=float(entry.get("weight", 1.0)),
            )
        )
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    return tasks


def grade(task: Task, result: AgentResult) -> tuple[float, dict[str, Any]]:
    kind = task.kind
    s = task.scoring

    if kind == "confluency":
        return m.confluency_score(
            result,
            target={k: float(v) for k, v in s["target_confluency_pct"].items()},
            tolerance_pct=float(s.get("tolerance_pct", 5.0)),
        )
    if kind == "cell_count":
        return m.cell_count_score(
            result,
            target={k: int(v) for k, v in s["target_cell_count"].items()},
            rel_tolerance=float(s.get("rel_tolerance", 0.30)),
            abs_tolerance=int(s.get("abs_tolerance", 10)),
        )
    if kind == "rag_retrieval":
        return m.retrieval_score(
            result,
            expected_doc_ids=list(s.get("expected_doc_ids", [])),
            require_all=bool(s.get("require_all", False)),
        )
    if kind == "structured_protocol":
        return m.structured_protocol_score(
            result,
            min_labware=int(s.get("min_labware", 1)),
            min_pipettes=int(s.get("min_pipettes", 1)),
            min_reagents=int(s.get("min_reagents", 0)),
        )
    if kind == "answer_contains":
        return m.answer_contains_score(
            result,
            expected_substrings=list(s.get("expected_substrings", [])),
            case_sensitive=bool(s.get("case_sensitive", False)),
        )
    if kind == "tool_order":
        return m.tool_order_score(
            result,
            required_tools=list(s.get("required_tools", [])),
            ordered=bool(s.get("ordered", False)),
        )
    if kind == "composite":
        parts = s.get("parts", [])
        if not parts:
            return 0.0, {"error": "composite task has no parts"}
        scores: list[float] = []
        details_parts: list[dict[str, Any]] = []
        for part in parts:
            part_task = Task(
                id=f"{task.id}:{part.get('name', part['kind'])}",
                kind=part["kind"],
                query=task.query,
                image_ids=task.image_ids,
                scoring=part,
                pass_threshold=task.pass_threshold,
                weight=float(part.get("weight", 1.0)),
            )
            ps, pd = grade(part_task, result)
            scores.append(ps * part_task.weight)
            details_parts.append(
                {"name": part.get("name", part["kind"]), "score": ps, "details": pd}
            )
        total_weight = sum(float(p.get("weight", 1.0)) for p in parts)
        final = sum(scores) / max(1e-9, total_weight)
        return final, {"parts": details_parts, "combined_weight": total_weight}

    return 0.0, {"error": f"unknown task kind: {kind}"}


def _safe_run(agent: BaseAgent, task: Task) -> tuple[AgentResult | None, str | None]:
    try:
        return agent.run(task.query, image_ids=task.image_ids or None), None
    except Exception as exc:
        log.exception("Task %s raised: %s", task.id, exc)
        return None, f"{type(exc).__name__}: {exc}"


def run_benchmark(queries_path: Path, task_ids: list[str] | None = None) -> dict[str, Any]:
    tasks = load_tasks(queries_path)
    if task_ids:
        tasks = [t for t in tasks if t.id in task_ids]
        if not tasks:
            raise ValueError(f"No tasks matched: {task_ids}")
    log.info("Loaded %d tasks from %s", len(tasks), queries_path)

    agent = load_agent()

    per_task: list[dict[str, Any]] = []
    total_weight = 0.0
    weighted_score = 0.0
    passed = 0

    for task in tasks:
        start = time.perf_counter()
        result, err = _safe_run(agent, task)
        wall_ms = (time.perf_counter() - start) * 1000.0

        if result is None:
            score = 0.0
            details: dict[str, Any] = {"error": err}
            raw_result: dict[str, Any] = {}
        else:
            score, details = grade(task, result)
            raw_result = json.loads(result.model_dump_json())

        is_pass = score >= task.pass_threshold
        passed += int(is_pass)
        total_weight += task.weight
        weighted_score += score * task.weight

        per_task.append(
            {
                "task_id": task.id,
                "kind": task.kind,
                "score": round(score, 4),
                "passed": is_pass,
                "weight": task.weight,
                "pass_threshold": task.pass_threshold,
                "details": details,
                "wall_ms": round(wall_ms, 1),
                "result": raw_result,
            }
        )

    overall = weighted_score / max(1e-9, total_weight)
    return {
        "queries_file": str(queries_path),
        "total": len(tasks),
        "passed": passed,
        "overall_score": round(overall, 4),
        "per_task": per_task,
    }
