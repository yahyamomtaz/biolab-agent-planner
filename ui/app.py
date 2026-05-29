"""Gradio comparison UI for the biolab agent.

Launches on ``0.0.0.0:7860``. Expects Ollama + Qdrant to be reachable via the
same env vars used by the FastAPI service.

Run::

    python ui/app.py
    # or inside the container:
    docker compose exec app python ui/app.py
"""

from __future__ import annotations

import concurrent.futures
import html
import json
import logging
import multiprocessing as mp
import os
import subprocess
import tempfile
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import gradio as gr
import yaml
from eval.harness import Task, grade

from biolab_agent.config import load_settings
from biolab_agent.logging import configure_logging
from biolab_agent.schemas import AgentResult
from biolab_agent.segmentation.visualize import render_segmentation_overlay

configure_logging()
log = logging.getLogger("biolab.ui.compare")

QUERIES_PATH = Path("data/queries_public.yaml")
NO_TASK = "(none - free-form query)"
BASELINE_AGENT_TARGET = "biolab_agent.agent.baseline:BaselineAgent"
PLANNER_AGENT_TARGET = "biolab_agent.agent.planner:PlannerAgent"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
GPU_SAMPLE_INTERVAL_S = 0.5
BRANCH_OUTPUT_COUNT = 7

VARIANTS: tuple[dict[str, str], ...] = (
    {
        "key": "baseline",
        "title": "Baseline Agent",
    },
    {
        "key": "planner",
        "title": "Planner Agent",
    },
)


def _variant_by_key(key: str) -> dict[str, str]:
    for variant in VARIANTS:
        if variant["key"] == key:
            return variant
    raise ValueError(f"Unknown variant key: {key}")


@contextmanager
def _temporary_env(updates: dict[str, str | None]) -> Iterator[None]:
    """Temporarily update environment variables inside a worker process."""
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _to_int(value: str) -> int | None:
    value = value.strip()
    if not value or value.upper() == "[N/A]":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _read_gpu_snapshot(worker_pid: int) -> dict[str, Any]:
    """Read a best-effort GPU snapshot from nvidia-smi."""
    try:
        gpu_proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found"}
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "error": f"nvidia-smi failed: {exc}"}

    devices: list[dict[str, Any]] = []
    uuid_to_index: dict[str, str] = {}
    for line in gpu_proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        index, uuid, name, util, mem_used, mem_total = parts[:6]
        uuid_to_index[uuid] = index
        devices.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "util_pct": _to_int(util),
                "memory_used_mb": _to_int(mem_used),
                "memory_total_mb": _to_int(mem_total),
                "worker_memory_mb": 0,
            }
        )

    try:
        proc_proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (subprocess.SubprocessError, OSError):
        proc_proc = None

    worker_memory_by_index: dict[str, int] = {}
    if proc_proc is not None:
        for line in proc_proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            gpu_uuid, pid_raw, used_memory = parts[:3]
            pid = _to_int(pid_raw)
            memory_mb = _to_int(used_memory)
            if pid != worker_pid or memory_mb is None:
                continue
            index = uuid_to_index.get(gpu_uuid)
            if index is not None:
                worker_memory_by_index[index] = worker_memory_by_index.get(index, 0) + memory_mb

    for device in devices:
        device["worker_memory_mb"] = worker_memory_by_index.get(device["index"], 0)
    return {"available": True, "devices": devices}


def _sample_gpu_until_stopped(
    stop_event: threading.Event,
    samples: list[dict[str, Any]],
    worker_pid: int,
) -> None:
    while not stop_event.is_set():
        samples.append(_read_gpu_snapshot(worker_pid))
        stop_event.wait(GPU_SAMPLE_INTERVAL_S)
    samples.append(_read_gpu_snapshot(worker_pid))


def _summarize_gpu_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    available_samples = [sample for sample in samples if sample.get("available")]
    if not available_samples:
        error = next((sample.get("error") for sample in samples if sample.get("error")), None)
        return {"available": False, "error": error or "GPU telemetry unavailable"}

    devices_by_index: dict[str, dict[str, Any]] = {}
    for sample in available_samples:
        for device in sample.get("devices", []):
            index = str(device.get("index", "?"))
            summary = devices_by_index.setdefault(
                index,
                {
                    "index": index,
                    "name": device.get("name") or f"GPU {index}",
                    "memory_total_mb": device.get("memory_total_mb"),
                    "peak_util_pct": 0,
                    "peak_memory_used_mb": 0,
                    "peak_worker_memory_mb": 0,
                },
            )
            summary["memory_total_mb"] = summary["memory_total_mb"] or device.get(
                "memory_total_mb"
            )
            summary["peak_util_pct"] = max(
                summary["peak_util_pct"],
                int(device.get("util_pct") or 0),
            )
            summary["peak_memory_used_mb"] = max(
                summary["peak_memory_used_mb"],
                int(device.get("memory_used_mb") or 0),
            )
            summary["peak_worker_memory_mb"] = max(
                summary["peak_worker_memory_mb"],
                int(device.get("worker_memory_mb") or 0),
            )

    if not devices_by_index:
        return {"available": False, "error": "No GPU devices reported by nvidia-smi"}

    def sort_key(item: dict[str, Any]) -> int:
        try:
            return int(item["index"])
        except (TypeError, ValueError):
            return 999

    return {
        "available": True,
        "sample_count": len(available_samples),
        "devices": sorted(devices_by_index.values(), key=sort_key),
    }


def _format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{max(0.0, min(value, 100.0)):.0f}%"


def _gpu_bar(label: str, pct: float | None, detail: str, color: str) -> str:
    if pct is None:
        fill_width = "0%"
        value = "-"
        fill_color = "#bbb"
    else:
        fill_width = _format_pct(pct)
        value = _format_pct(pct)
        fill_color = color
    return (
        "<div style='margin:0.35rem 0'>"
        "<div style='display:flex;justify-content:space-between;gap:0.75rem;"
        "font-size:0.9em;font-weight:600'>"
        f"<span>{html.escape(label)}</span><span>{html.escape(value)}</span>"
        "</div>"
        "<div style='height:0.65rem;background:#eceff1;border-radius:999px;overflow:hidden'>"
        f"<div style='width:{fill_width};height:100%;background:{fill_color};"
        "border-radius:999px'></div>"
        "</div>"
        f"<div style='color:#666;font-size:0.82em'>{html.escape(detail)}</div>"
        "</div>"
    )


def _format_gpu_usage(gpu: dict[str, Any] | None) -> str:
    if not gpu or not gpu.get("available"):
        error = html.escape(str((gpu or {}).get("error") or "GPU telemetry unavailable"))
        return (
            "<div style='border:1px solid #e0e0e0;border-radius:8px;padding:0.65rem;"
            "margin:0.5rem 0;color:#777'>"
            f"<div style='font-weight:700;color:#555'>GPU usage</div><div>{error}</div>"
            "</div>"
        )

    sections: list[str] = []
    for device in gpu.get("devices", []):
        total = int(device.get("memory_total_mb") or 0)
        peak_used = int(device.get("peak_memory_used_mb") or 0)
        worker_used = int(device.get("peak_worker_memory_mb") or 0)
        mem_pct = (peak_used / total * 100.0) if total else None
        worker_pct = (worker_used / total * 100.0) if total and worker_used else None
        util_pct = float(device.get("peak_util_pct") or 0)
        title = f"GPU {device.get('index', '?')} - {device.get('name', 'unknown')}"
        worker_detail = (
            f"branch process peak {worker_used} / {total} MB"
            if worker_used and total
            else "branch process VRAM not reported by nvidia-smi"
        )
        sections.append(
            f"<div style='font-weight:700;margin-top:0.15rem'>{html.escape(title)}</div>"
            + _gpu_bar(
                "GPU-wide peak utilization",
                util_pct,
                f"{gpu.get('sample_count', 0)} samples",
                "#2f80ed",
            )
            + _gpu_bar("GPU-wide peak VRAM", mem_pct, f"{peak_used} / {total} MB", "#207a4a")
            + _gpu_bar("Branch process VRAM", worker_pct, worker_detail, "#9b6a00")
        )

    return (
        "<div style='border:1px solid #e0e0e0;border-radius:8px;padding:0.65rem;"
        "margin:0.5rem 0'>"
        f"{''.join(sections)}"
        "</div>"
    )


def _run_agent_worker(
    variant_key: str,
    query: str,
    image_ids: list[str],
    reranker_model: str,
    rerank_candidates: int,
) -> dict[str, Any]:
    """Run one isolated agent branch.

    This is executed in a child process. Keeping each branch in its own process
    avoids races because the existing RAG tools read their mode from env vars.
    """
    if variant_key == "baseline":
        env_updates = {
            "BIOLAB_AGENT_CLASS": BASELINE_AGENT_TARGET,
            "BIOLAB_HYBRID_RETRIEVAL": "",
            "BIOLAB_RERANKER_MODEL": "",
            "BIOLAB_RERANK_CANDIDATES": str(rerank_candidates),
        }
    else:
        env_updates = {
            "BIOLAB_AGENT_CLASS": PLANNER_AGENT_TARGET,
            "BIOLAB_HYBRID_RETRIEVAL": "1",
            "BIOLAB_RERANK_CANDIDATES": str(rerank_candidates),
            "BIOLAB_RERANKER_MODEL": reranker_model.strip() or DEFAULT_RERANKER_MODEL,
        }

    started = time.perf_counter()
    gpu_samples: list[dict[str, Any]] = []
    stop_gpu_sampling = threading.Event()
    gpu_sampler = threading.Thread(
        target=_sample_gpu_until_stopped,
        args=(stop_gpu_sampling, gpu_samples, os.getpid()),
        daemon=True,
    )
    gpu_sampler.start()
    payload: dict[str, Any]
    try:
        with _temporary_env(env_updates):
            if variant_key == "baseline":
                import biolab_agent.tools as tool_registry
                from biolab_agent.tools.rag import retrieve_protocol, retrieve_protocol_spec
                from biolab_agent.tools.reagents import reagents, reagents_spec

                tool_registry.TOOL_IMPLS["retrieve_protocol"] = retrieve_protocol
                tool_registry.TOOL_IMPLS["reagents"] = tool_registry.TOOL_IMPLS.pop("lookup_reagent")
                tool_registry.TOOL_SPECS = [
                    retrieve_protocol_spec if spec["function"]["name"] == "retrieve_protocol"
                    else reagents_spec if spec["function"]["name"] == "lookup_reagent"
                    else spec
                    for spec in tool_registry.TOOL_SPECS
                ]

            from biolab_agent.agent import load_agent

            agent = load_agent()
            result = agent.run(query=query, image_ids=image_ids or None)
        payload = {
            "ok": True,
            "variant_key": variant_key,
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "result": result.model_dump(mode="json"),
        }
    except Exception as exc:  # pragma: no cover - displayed in the UI
        payload = {
            "ok": False,
            "variant_key": variant_key,
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
    finally:
        stop_gpu_sampling.set()
        gpu_sampler.join(timeout=3)
    payload["gpu"] = _summarize_gpu_samples(gpu_samples)
    return payload


def _list_images() -> list[str]:
    settings = load_settings()
    img_dir = Path(settings.biolab_data_dir) / "images"
    if not img_dir.exists():
        return []
    return sorted(p.stem for p in img_dir.glob("*.png"))


def _resolve_image_paths(image_ids: list[str]) -> list[str]:
    settings = load_settings()
    img_dir = Path(settings.biolab_data_dir) / "images"
    return [str(img_dir / f"{i}.png") for i in image_ids if (img_dir / f"{i}.png").exists()]


def _load_tasks() -> dict[str, dict[str, Any]]:
    """Map task_id to raw YAML entry. Empty dict if the file is missing."""
    if not QUERIES_PATH.exists():
        return {}
    doc = yaml.safe_load(QUERIES_PATH.read_text(encoding="utf-8")) or {}
    return {q["id"]: q for q in doc.get("queries", [])}


def _task_to_obj(entry: dict[str, Any]) -> Task:
    return Task(
        id=entry["id"],
        kind=entry["kind"],
        query=entry["query"],
        image_ids=entry.get("image_ids", []),
        scoring=entry.get("scoring", {}),
        pass_threshold=float(entry.get("pass_threshold", 0.7)),
        weight=float(entry.get("weight", 1.0)),
    )


def _empty_result(query: str) -> AgentResult:
    """Placeholder AgentResult-shaped value for scoring after a crash."""
    return AgentResult(query=query, answer="", model="", elapsed_ms=0.0)


def _build_segmentation_overlays(
    result: AgentResult,
    settings,
    variant_key: str,
    only_image_ids: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Render segmentation overlays from an agent trace."""
    img_dir = Path(settings.biolab_data_dir) / "images"
    out_dir = Path(tempfile.gettempdir()) / "biolab_overlays" / variant_key
    out_dir.mkdir(parents=True, exist_ok=True)
    allow = set(only_image_ids) if only_image_ids else None
    items: list[tuple[str, str]] = []
    for step in result.trace:
        if step.tool != "segment_wells" or not step.ok or not isinstance(step.observation, dict):
            continue
        image_id = step.args.get("image_id") or step.observation.get("image_id")
        if not image_id:
            continue
        if allow is not None and image_id not in allow:
            continue
        img_path = img_dir / f"{image_id}.png"
        if not img_path.exists():
            continue
        for mask in step.observation.get("masks", []):
            rle = mask.get("rle") or ""
            try:
                overlay = render_segmentation_overlay(
                    img_path, rle, int(mask["height"]), int(mask["width"])
                )
            except Exception:
                log.exception("overlay_render_failed", extra={"image_id": image_id})
                continue
            cells = mask.get("cell_count", "?")
            confl = mask.get("confluency_pct", "?")
            out_path = out_dir / f"{image_id}_step{step.step}.png"
            overlay.save(out_path)
            label = f"{image_id} - cells={cells}, confluency={confl}%"
            items.append((str(out_path), label))
    return items


def _status_banner(title: str, result: AgentResult, wall_ms: float, has_error: bool) -> str:
    """Compact summary header above one branch's result panels."""
    n_calls = len(result.trace)
    n_ok = sum(1 for t in result.trace if t.ok)
    n_err = n_calls - n_ok
    n_cites = len(result.citations)
    has_struct = bool(result.structured)
    elapsed_s = result.elapsed_ms / 1000.0
    wall_s = wall_ms / 1000.0

    if has_error or (n_calls == 0 and not result.answer):
        color = "#a33"
        state = "ERROR"
    elif n_err == 0:
        color = "#207a4a"
        state = "OK"
    else:
        color = "#9b6a00"
        state = "PARTIAL"
    parts = [
        f"{state}",
        f"agent {elapsed_s:.1f}s",
        f"wall {wall_s:.1f}s",
        f"{n_calls} tool call{'s' if n_calls != 1 else ''} ({n_ok} ok / {n_err} err)",
        f"{n_cites} citation{'s' if n_cites != 1 else ''}",
        ("structured" if has_struct else "no structured"),
    ]
    return (
        f"<div style='color:{color};font-weight:700'>{html.escape(title)}</div>"
        f"<div style='color:{color};font-weight:600'>{' | '.join(parts)}</div>"
    )


def _score_result(task_id: str, result: AgentResult) -> tuple[float | None, bool | None, str]:
    if task_id == NO_TASK or not task_id:
        return None, None, "<div style='color:#777'>No benchmark task selected.</div>"
    tasks = _load_tasks()
    entry = tasks.get(task_id)
    if entry is None:
        return None, None, f"<div style='color:#a33'>Task {task_id!r} not found.</div>"
    task = _task_to_obj(entry)
    score, details = grade(task, result)
    passed = score >= task.pass_threshold
    badge = "PASS" if passed else "FAIL"
    color = "#207a4a" if passed else "#a33"
    summary = (
        f"<div style='color:{color};font-weight:700'>"
        f"{badge} - score {score:.2f} (threshold {task.pass_threshold:.2f}, "
        f"weight {task.weight}, kind={task.kind})</div>"
    )
    detail_block = (
        "<details><summary>scoring details</summary>"
        f"<pre>{html.escape(json.dumps(details, indent=2)[:3000])}</pre>"
        "</details>"
    )
    return score, passed, summary + detail_block


def _format_trace(result: AgentResult) -> str:
    trace_rows = [
        f"{i + 1}. {t.tool}({json.dumps(t.args)[:120]})  "
        f"-> {'OK' if t.ok else 'ERR: ' + (t.error or '')}  "
        f"({t.elapsed_ms:.0f} ms)"
        for i, t in enumerate(result.trace)
    ]
    trace_str = "\n".join(trace_rows) or "(no tool calls)"
    cites_str = "\n".join(f"- {doc}:{chunk}" for doc, chunk in result.citations) or "(no citations)"
    return f"{trace_str}\n\nCitations:\n{cites_str}"


def _format_score(score: Any) -> str:
    try:
        return f"{float(score):.10f}"
    except (TypeError, ValueError):
        return "-"


def _format_retrieval_rankings(result: AgentResult) -> str:
    """Render retrieve_protocol observations and final citation order."""
    retrieval_calls = [
        step
        for step in result.trace
        if step.tool == "retrieve_protocol" and step.ok and isinstance(step.observation, list)
    ]
    if not retrieval_calls and not result.citations:
        return "<div style='color:#777'>No retrieval rankings available.</div>"

    sections: list[str] = []
    for call_idx, step in enumerate(retrieval_calls, start=1):
        query = html.escape(str(step.args.get("query", "")))
        k = html.escape(str(step.args.get("k") or len(step.observation)))
        rows: list[str] = []
        for rank, hit in enumerate(step.observation, start=1):
            if not isinstance(hit, dict):
                continue
            doc_id = html.escape(str(hit.get("doc_id", "")))
            chunk_id = html.escape(str(hit.get("chunk_id", "")))
            title = html.escape(str(hit.get("title", "")))
            score = html.escape(_format_score(hit.get("score")))
            rows.append(
                "<tr>"
                f"<td>{rank}</td>"
                f"<td><code>{doc_id}</code></td>"
                f"<td><code>{chunk_id}</code></td>"
                f"<td>{score}</td>"
                f"<td>{title}</td>"
                "</tr>"
            )
        if not rows:
            rows.append("<tr><td colspan='5' style='color:#777'>No hits returned.</td></tr>")
        sections.append(
            f"<div style='font-weight:700;margin:0.35rem 0'>Retrieval call {call_idx}: "
            f"<code>{query}</code> k={k}</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.92em'>"
            "<thead><tr><th align='left'>Rank</th><th align='left'>doc_id</th>"
            "<th align='left'>chunk_id</th><th align='left'>score</th>"
            "<th align='left'>title</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    if result.citations:
        citation_rows = [
            "<tr>"
            f"<td>{rank}</td>"
            f"<td><code>{html.escape(str(doc))}</code></td>"
            f"<td><code>{html.escape(str(chunk))}</code></td>"
            "</tr>"
            for rank, (doc, chunk) in enumerate(result.citations, start=1)
        ]
        sections.append(
            "<div style='font-weight:700;margin:0.75rem 0 0.35rem'>Final citation ranking</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.92em'>"
            "<thead><tr><th align='left'>Rank</th><th align='left'>doc_id</th>"
            "<th align='left'>chunk_id</th></tr></thead>"
            f"<tbody>{''.join(citation_rows)}</tbody></table>"
        )

    return "".join(sections)


def _format_payload(
    payload: dict[str, Any],
    variant: dict[str, str],
    query: str,
    image_ids: list[str],
    show_all_overlays: bool,
    task_id: str,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    list[tuple[str, str]],
    str,
    AgentResult | None,
    float | None,
]:
    if not payload.get("ok"):
        error = payload.get("error", "unknown error")
        tb = payload.get("traceback", "")
        empty = _empty_result(query)
        _, _, score_html = _score_result(task_id, empty)
        status = (
            f"<div style='color:#a33;font-weight:700'>{html.escape(variant['title'])}</div>"
            f"<div style='color:#a33;font-weight:600'>ERROR after "
            f"{float(payload.get('wall_ms', 0.0)) / 1000.0:.1f}s: {html.escape(error)}</div>"
        )
        return (
            status,
            f"(crashed: {error})",
            "",
            tb,
            "",
            [],
            score_html,
            None,
            None,
        )

    result = AgentResult.model_validate(payload["result"])
    settings = load_settings()
    structured_str = json.dumps(result.structured, indent=2) if result.structured else "(none)"
    overlays = _build_segmentation_overlays(
        result,
        settings,
        variant_key=variant["key"],
        only_image_ids=None if show_all_overlays else (image_ids or None),
    )
    score, _, score_html = _score_result(task_id, result)
    return (
        _status_banner(
            variant["title"],
            result,
            wall_ms=float(payload.get("wall_ms", result.elapsed_ms)),
            has_error=False,
        ),
        result.answer,
        structured_str,
        _format_trace(result),
        _format_retrieval_rankings(result),
        overlays,
        score_html,
        result,
        score,
    )


def _comparison_summary(
    task_id: str,
    payloads: dict[str, dict[str, Any]],
    formatted: dict[str, tuple[AgentResult | None, float | None]],
) -> str:
    rows = []
    for variant in VARIANTS:
        key = variant["key"]
        payload = payloads.get(key, {})
        result, score = formatted.get(key, (None, None))
        if not payload.get("ok") or result is None:
            rows.append(
                "<tr>"
                f"<td>{html.escape(variant['title'])}</td>"
                "<td>crashed</td>"
                f"<td>{float(payload.get('wall_ms', 0.0)) / 1000.0:.1f}s</td>"
                "<td>-</td><td>-</td><td>-</td>"
                "</tr>"
            )
            continue
        score_text = "-" if score is None else f"{score:.2f}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(variant['title'])}</td>"
            "<td>done</td>"
            f"<td>{result.elapsed_ms / 1000.0:.1f}s</td>"
            f"<td>{len(result.trace)}</td>"
            f"<td>{len(result.citations)}</td>"
            f"<td>{score_text}</td>"
            "</tr>"
        )
    task_label = "free-form query" if task_id == NO_TASK or not task_id else html.escape(task_id)
    return (
        f"<div style='font-weight:700;margin-bottom:0.4rem'>Parallel run summary - {task_label}</div>"
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr><th align='left'>Branch</th><th align='left'>State</th>"
        "<th align='left'>Agent time</th><th align='left'>Tools</th>"
        "<th align='left'>Citations</th><th align='left'>Score</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _single_run_summary(
    task_id: str,
    variant: dict[str, str],
    payload: dict[str, Any],
    result: AgentResult | None,
    score: float | None,
) -> str:
    task_label = "free-form query" if task_id == NO_TASK or not task_id else html.escape(task_id)
    score_text = "-" if score is None else f"{score:.2f}"
    if not payload.get("ok") or result is None:
        state = "crashed"
        agent_time = f"{float(payload.get('wall_ms', 0.0)) / 1000.0:.1f}s"
        tools = "-"
        citations = "-"
    else:
        state = "done"
        agent_time = f"{result.elapsed_ms / 1000.0:.1f}s"
        tools = str(len(result.trace))
        citations = str(len(result.citations))
    return (
        f"<div style='font-weight:700;margin-bottom:0.4rem'>Single run summary - {task_label}</div>"
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr><th align='left'>Branch</th><th align='left'>State</th>"
        "<th align='left'>Agent time</th><th align='left'>Tools</th>"
        "<th align='left'>Citations</th><th align='left'>Score</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{html.escape(variant['title'])}</td>"
        f"<td>{state}</td>"
        f"<td>{agent_time}</td>"
        f"<td>{tools}</td>"
        f"<td>{citations}</td>"
        f"<td>{score_text}</td>"
        "</tr></tbody></table>"
    )


def _on_task_change(task_id: str):  # type: ignore[no-untyped-def]
    """When the user picks a benchmark task, autofill query + image_ids."""
    if task_id == NO_TASK or not task_id:
        return gr.update(), gr.update()
    tasks = _load_tasks()
    entry = tasks.get(task_id)
    if entry is None:
        return gr.update(), gr.update()
    return gr.update(value=entry["query"]), gr.update(value=entry.get("image_ids", []))


def _empty_branch_outputs(message: str) -> tuple[str, str, str, str, str, list[Any], str]:
    return (
        f"<div style='color:#a33'>{html.escape(message)}</div>",
        message,
        "",
        "",
        "",
        [],
        "",
    )


def _run_one_payload(
    variant: dict[str, str],
    query: str,
    image_ids: list[str],
    reranker_model: str,
    candidates: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        future = pool.submit(
            _run_agent_worker,
            variant["key"],
            query,
            image_ids,
            reranker_model,
            candidates,
        )
        try:
            return future.result()
        except Exception as exc:  # pragma: no cover - displayed in the UI
            log.exception("single_branch_failed", extra={"variant": variant["key"]})
            return {
                "ok": False,
                "variant_key": variant["key"],
                "wall_ms": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }


def _run(
    query: str,
    image_ids: list[str],
    show_all_overlays: bool,
    task_id: str,
    reranker_model: str,
    rerank_candidates: int,
):  # type: ignore[no-untyped-def]
    if not query.strip():
        empty_outputs = (
            "<div style='color:#a33'>Enter a query or pick a benchmark task.</div>",
            [],
        )
        branch_outputs = _empty_branch_outputs("Enter a query.")
        return (*empty_outputs, *branch_outputs, *branch_outputs)

    image_ids = image_ids or []
    candidates = max(1, min(int(rerank_candidates or 20), 25))
    payloads: dict[str, dict[str, Any]] = {}

    # Use process isolation so both branches can run at the same time without
    # sharing the RAG mode env vars.
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(VARIANTS),
        mp_context=ctx,
    ) as pool:
        futures = {
            pool.submit(
                _run_agent_worker,
                variant["key"],
                query,
                image_ids,
                reranker_model,
                candidates,
            ): variant
            for variant in VARIANTS
        }
        for future in concurrent.futures.as_completed(futures):
            variant = futures[future]
            try:
                payloads[variant["key"]] = future.result()
            except Exception as exc:  # pragma: no cover - displayed in the UI
                log.exception("parallel_branch_failed", extra={"variant": variant["key"]})
                payloads[variant["key"]] = {
                    "ok": False,
                    "variant_key": variant["key"],
                    "wall_ms": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                }

    outputs_by_key: dict[
        str,
        tuple[str, str, str, str, str, str, list[tuple[str, str]], str],
    ] = {}
    formatted: dict[str, tuple[AgentResult | None, float | None]] = {}
    for variant in VARIANTS:
        formatted_payload = _format_payload(
            payloads[variant["key"]],
            variant,
            query,
            image_ids,
            show_all_overlays,
            task_id,
        )
        outputs_by_key[variant["key"]] = formatted_payload[:7]
        formatted[variant["key"]] = (formatted_payload[7], formatted_payload[8])

    summary = _comparison_summary(task_id, payloads, formatted)
    return (
        summary,
        _resolve_image_paths(image_ids),
        *outputs_by_key["baseline"],
        *outputs_by_key["planner"],
    )


def _run_one(
    variant_key: str,
    query: str,
    image_ids: list[str],
    show_all_overlays: bool,
    task_id: str,
    reranker_model: str,
    rerank_candidates: int,
):  # type: ignore[no-untyped-def]
    if not query.strip():
        empty_outputs = (
            "<div style='color:#a33'>Enter a query or pick a benchmark task.</div>",
            [],
        )
        branch_outputs = _empty_branch_outputs("Enter a query.")
        return (*empty_outputs, *branch_outputs)

    image_ids = image_ids or []
    candidates = max(1, min(int(rerank_candidates or 20), 25))
    variant = _variant_by_key(variant_key)
    payload = _run_one_payload(variant, query, image_ids, reranker_model, candidates)
    formatted_payload = _format_payload(
        payload,
        variant,
        query,
        image_ids,
        show_all_overlays,
        task_id,
    )
    branch_outputs = formatted_payload[:BRANCH_OUTPUT_COUNT]
    summary = _single_run_summary(task_id, variant, payload, formatted_payload[7], formatted_payload[8])
    return (
        summary,
        _resolve_image_paths(image_ids),
        *branch_outputs,
    )


def _build_interface() -> gr.Blocks:
    image_choices = _list_images()
    task_choices = [NO_TASK, *sorted(_load_tasks().keys())]
    configured_reranker = os.getenv("BIOLAB_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    configured_candidates = int(os.getenv("BIOLAB_RERANK_CANDIDATES", "25") or "25")

    css = (
        "#btn-planner { background: #e8630a; border-color: #e8630a; color: white; }"
        "#summary-panel { margin-top: 1.2rem; margin-bottom: 0.6rem; }"
    )
    with gr.Blocks(title="Biolab Agent", css=css) as ui:
        gr.Markdown("# Biolab Agent")

        with gr.Row():
            task_picker = gr.Dropdown(
                choices=task_choices,
                value=NO_TASK,
                label="Benchmark task",
            )

        query = gr.Textbox(
            label="Query",
            lines=4,
            placeholder="e.g. Find the custom PCR preparation protocol and return its id.",
        )

        with gr.Row():
            images = gr.Dropdown(
                choices=image_choices,
                multiselect=True,
                label="Image IDs",
                value=[],
            )
            show_all = gr.Checkbox(
                label="Show all segmentation overlays",
                value=False,
            )

        with gr.Accordion("Reranker settings", open=False):
            with gr.Row():
                reranker_model = gr.Textbox(
                    label="Reranker model",
                    value=configured_reranker,
                    placeholder=DEFAULT_RERANKER_MODEL,
                )
                rerank_candidates = gr.Slider(
                    label="First-stage candidates",
                    minimum=1,
                    maximum=25,
                    value=max(1, min(configured_candidates, 25)),
                    step=1,
                )

        with gr.Row():
            run_baseline = gr.Button("Run baseline agent", variant="primary")
            run_planner = gr.Button("Run planner agent", elem_id="btn-planner")

        summary = gr.HTML(value="", elem_id="summary-panel")
        gallery = gr.Gallery(label="Selected images", columns=5, rows=1, height=220)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown(f"## {VARIANTS[0]['title']}")
                baseline_status = gr.HTML(value="")
                baseline_score = gr.HTML(value="")
                baseline_answer = gr.Markdown(label="Answer", value="")
                baseline_structured = gr.Code(label="Structured output", language="json", value="")
                baseline_trace = gr.Textbox(label="Trace + citations", lines=14, interactive=False)
                baseline_rankings = gr.HTML(value="")
                baseline_overlays = gr.Gallery(
                    label="Segmentation overlays",
                    columns=3,
                    rows=1,
                    height=240,
                )

            with gr.Column(scale=1):
                gr.Markdown(f"## {VARIANTS[1]['title']}")
                rerank_status = gr.HTML(value="")
                rerank_score = gr.HTML(value="")
                rerank_answer = gr.Markdown(label="Answer", value="")
                rerank_structured = gr.Code(label="Structured output", language="json", value="")
                rerank_trace = gr.Textbox(label="Trace + citations", lines=14, interactive=False)
                rerank_rankings = gr.HTML(value="")
                rerank_overlays = gr.Gallery(
                    label="Segmentation overlays",
                    columns=3,
                    rows=1,
                    height=240,
                )

        task_picker.change(_on_task_change, inputs=[task_picker], outputs=[query, images])

        run_baseline.click(
            lambda *args: _run_one("baseline", *args),
            inputs=[
                query,
                images,
                show_all,
                task_picker,
                reranker_model,
                rerank_candidates,
            ],
            outputs=[
                summary,
                gallery,
                baseline_status,
                baseline_answer,
                baseline_structured,
                baseline_trace,
                baseline_rankings,
                baseline_overlays,
                baseline_score,
            ],
        )
        run_planner.click(
            lambda *args: _run_one("planner", *args),
            inputs=[
                query,
                images,
                show_all,
                task_picker,
                reranker_model,
                rerank_candidates,
            ],
            outputs=[
                summary,
                gallery,
                rerank_status,
                rerank_answer,
                rerank_structured,
                rerank_trace,
                rerank_rankings,
                rerank_overlays,
                rerank_score,
            ],
        )
    return ui


def main() -> None:
    host = os.getenv("UI_HOST", "0.0.0.0")
    port = int(os.getenv("UI_PORT", "7860"))
    ui = _build_interface()
    ui.launch(
        server_name=host,
        server_port=port,
        show_error=True,
        allowed_paths=["/data/images", "/tmp"],
    )


if __name__ == "__main__":
    main()
