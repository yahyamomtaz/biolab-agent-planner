"""Typer-based CLI: ``biolab-bench`` and ``biolab-index`` entrypoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from biolab_agent.config import load_settings
from biolab_agent.logging import configure_logging

app = typer.Typer(help="Biolab Agent utilities", no_args_is_help=True)
console = Console()


@app.command()
def bench(
    queries: Annotated[
        Path,
        typer.Option(help="YAML file with benchmark queries", exists=True, readable=True),
    ] = Path("data/queries_public.yaml"),
    report: Annotated[
        Path,
        typer.Option(help="Where to write the JSON report"),
    ] = Path("artifacts/bench_report.json"),
    agent_class: Annotated[
        str | None,
        typer.Option(help="Override BIOLAB_AGENT_CLASS"),
    ] = None,
    tasks: Annotated[
        str | None,
        typer.Option(help="Comma-separated task IDs to run (e.g. T4_reagent_lookup,T10_reagent_absence)"),
    ] = None,
) -> None:
    """Run the benchmark harness against the configured agent."""
    from eval.harness import run_benchmark  # lazy import: harness pulls in heavy deps

    configure_logging()
    import os

    if agent_class:
        os.environ["BIOLAB_AGENT_CLASS"] = agent_class

    settings = load_settings()
    console.print(f"[bold]Running benchmark[/bold] → {queries}")
    console.print(f"  agent_class = {os.getenv('BIOLAB_AGENT_CLASS', '<stub>')}")
    console.print(f"  model       = {settings.biolab_llm_model}")
    console.print(f"  adapter     = {settings.biolab_lora_adapter or '<none>'}")

    task_ids = [t.strip() for t in tasks.split(",")] if tasks else None
    report_data = run_benchmark(queries_path=queries, task_ids=task_ids)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(report_data, indent=2, default=str))

    table = Table(title="Benchmark results")
    table.add_column("Task")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right")
    for row in report_data["per_task"]:
        status = "[green]PASS[/green]" if row["passed"] else "[red]FAIL[/red]"
        table.add_row(row["task_id"], status, f"{row['score']:.2f}")
    console.print(table)
    console.print(
        f"\n[bold]Overall:[/bold] {report_data['overall_score']:.2f}  "
        f"passed {report_data['passed']}/{report_data['total']}"
    )
    console.print(f"Full report: {report}")
    if report_data["passed"] < report_data["total"]:
        sys.exit(1)


@app.command(name="index")
def index_corpus(
    corpus_dir: Annotated[
        Path,
        typer.Option(help="Directory with protocol JSONL files", exists=True),
    ] = Path("data/protocols"),
    collection: Annotated[
        str,
        typer.Option(help="Qdrant collection name"),
    ] = "protocols",
    dry_run: Annotated[
        bool,
        typer.Option(help="Parse and chunk only; skip embedding/indexing"),
    ] = False,
) -> None:
    """Helper to index the provided corpus into Qdrant.

    Intentionally left as a helper (not a full solution). Candidates are
    expected to supply their own production indexing pipeline in
    ``src/biolab_agent/rag/``.
    """
    from biolab_agent.rag.ingest import ingest_corpus

    configure_logging()
    stats = ingest_corpus(corpus_dir=corpus_dir, collection=collection, dry_run=dry_run)
    console.print_json(json.dumps(stats))


if __name__ == "__main__":
    app()
