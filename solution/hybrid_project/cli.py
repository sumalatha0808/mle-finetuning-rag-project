"""Command-line workflow for data inspection, training, querying, and evaluation."""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import click

from .config import Settings
from .datasets import validate_data
from .embeddings import HashEmbedder
from .evaluation import write_report
from .generation import ExtractiveSmokeGenerator, LocalModelGenerator
from .models import SystemName
from .pipeline import ComparisonPipeline
from .retrieval import Retriever
from .training import train_adapter


def _settings(hash_smoke: bool) -> Settings:
    """Load settings and label smoke-only retrieval explicitly."""

    settings = Settings.from_env()
    if hash_smoke:
        return replace(
            settings,
            embedding_model="hash-smoke-test-only",
            min_score=0.20,
        )
    return settings


def _retriever(settings: Settings, hash_smoke: bool) -> Retriever:
    """Build dense retrieval or the deterministic smoke substitute."""

    if hash_smoke:
        return Retriever(settings, embedder=HashEmbedder())
    return Retriever(settings)


def _pipeline(
    settings: Settings, hash_smoke: bool, system: SystemName | None = None
) -> ComparisonPipeline:
    """Assemble the comparison path with real or smoke generators."""

    retriever = None if system == "base-only" else _retriever(settings, hash_smoke)
    if hash_smoke:
        return ComparisonPipeline(
            settings,
            retriever,
            ExtractiveSmokeGenerator("baseline"),
            ExtractiveSmokeGenerator("hybrid"),
        )
    return ComparisonPipeline(
        settings,
        retriever,
        LocalModelGenerator(settings, use_adapter=False)
        if system != "rag-lora"
        else None,
        LocalModelGenerator(settings, use_adapter=True)
        if system in (None, "rag-lora")
        else None,
    )


@click.group()
@click.option(
    "--hash-smoke",
    is_flag=True,
    help="Use deterministic test substitutes; never treat them as model results.",
)
@click.pass_context
def main(context: click.Context, hash_smoke: bool) -> None:
    """Run the Northstar fine-tuning and RAG reference workflow."""

    context.ensure_object(dict)
    context.obj["hash_smoke"] = hash_smoke


@main.command("inspect-data")
def inspect_data() -> None:
    """Validate source, training, and evaluation contracts."""

    settings = Settings.from_env()
    summary = validate_data(
        (
            settings.knowledge_base_dir,
            settings.training_file,
            settings.evaluation_file,
        )
    )
    click.echo(json.dumps(summary.model_dump(mode="json"), indent=2))


@main.command()
@click.pass_context
def index(context: click.Context) -> None:
    """Build the shared vector index and print its configuration."""

    hash_smoke = bool(context.obj["hash_smoke"])
    settings = _settings(hash_smoke)
    retriever = _retriever(settings, hash_smoke)
    click.echo(
        f"Indexed {len(retriever.chunks)} chunks with "
        f"{settings.embedding_model} in Qdrant local mode."
    )


@main.command()
@click.pass_context
def train(context: click.Context) -> None:
    """Run real LoRA optimization and save the adapter."""

    if context.obj["hash_smoke"]:
        raise click.UsageError(
            "--hash-smoke cannot be used with train; training is real."
        )
    summary = train_adapter(Settings.from_env())
    click.echo(json.dumps(summary.model_dump(mode="json"), indent=2))


@main.command()
@click.argument("question")
@click.option(
    "--system",
    "system_name",
    type=click.Choice(["base-only", "rag-base", "rag-lora"]),
    default="rag-lora",
    show_default=True,
)
@click.pass_context
def ask(
    context: click.Context,
    question: str,
    system_name: str,
) -> None:
    """Run one question through a selected comparison system."""

    hash_smoke = bool(context.obj["hash_smoke"])
    settings = _settings(hash_smoke)
    system = cast(SystemName, system_name)
    pipeline = _pipeline(settings, hash_smoke, system)
    result = pipeline.answer(question, system)
    click.echo(json.dumps(result.model_dump(mode="json"), indent=2))


@main.command()
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Report path; defaults to reference_evaluation.md (smoke_evaluation.md in smoke mode) under solution/reports.",
)
@click.pass_context
def evaluate(context: click.Context, output: Path | None) -> None:
    """Run the complete shared benchmark and write its decision report."""

    hash_smoke = bool(context.obj["hash_smoke"])
    settings = _settings(hash_smoke)
    reports = settings.project_root / "solution" / "reports"
    reference = reports / "reference_evaluation.md"
    output = output or (reports / "smoke_evaluation.md" if hash_smoke else reference)
    if hash_smoke and output.with_suffix(".md").resolve() == reference.resolve():
        raise click.UsageError(
            "Smoke results must not overwrite the reference report or its trace."
        )
    summary = write_report(_pipeline(settings, hash_smoke), output)
    rag_base = next(system for system in summary.systems if system.system == "rag-base")
    rag_lora = next(system for system in summary.systems if system.system == "rag-lora")
    click.echo(f"Wrote evaluation report to {output}")
    click.echo(
        f"retrieval_recall={summary.retrieval_recall_at_k:.3f} "
        f"rag_base_relevance={rag_base.answer_relevance:.3f} "
        f"rag_lora_relevance={rag_lora.answer_relevance:.3f}"
    )


if __name__ == "__main__":
    main()
