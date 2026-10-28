"""Tests for project data contracts and prompt separation."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from solution.hybrid_project.config import Settings
from solution.hybrid_project.datasets import validate_data
from solution.hybrid_project.documents import chunk_documents, load_documents
from solution.hybrid_project.generation import validate_adapter
from solution.hybrid_project.models import TrainingSummary


def test_shared_data_is_valid_and_auditable() -> None:
    """All labels should resolve to supplied policy sources."""

    root = Path(__file__).parents[2]
    settings = Settings.from_env(project_root=root)
    summary = validate_data(
        (
            settings.knowledge_base_dir,
            settings.training_file,
            settings.evaluation_file,
        )
    )

    assert summary.source_count == 6
    assert summary.training_example_count == 16
    assert summary.evaluation_case_count == 14


def test_target_citations_must_be_provided(tmp_path: Path) -> None:
    """A valid corpus source is not enough: it must be present in this example."""

    settings = Settings.from_env()
    records = [
        json.loads(line) for line in settings.training_file.read_text().splitlines()
    ]
    records[-1]["reference_ids"] = ["response-style"]
    records[-1]["context"] = "[response-style] Cite policy claims."
    training = tmp_path / "training.jsonl"
    training.write_text("\n".join(json.dumps(record) for record in records))
    with pytest.raises(ValueError, match="answer cites a source absent"):
        validate_data((settings.knowledge_base_dir, training, settings.evaluation_file))


def test_chunks_retain_source_identity_and_position() -> None:
    """Citation metadata must survive policy chunking."""

    root = Path(__file__).parents[2]
    chunks = chunk_documents(load_documents(root / "data" / "knowledge_base"))

    assert chunks
    assert all(chunk.document_id in chunk.chunk_id for chunk in chunks)
    assert all(chunk.position >= 0 for chunk in chunks)


@pytest.mark.parametrize(
    "change", ["none", "data", "revision", "model", "legacy", "missing"]
)
def test_adapter_provenance_is_checked_before_reuse(
    tmp_path: Path, change: str
) -> None:
    """Keep the artifact contract independent from model downloads and training."""

    settings = replace(
        Settings.from_env(),
        training_file=tmp_path / "training.jsonl",
        adapter_path=tmp_path / "adapter",
    )
    settings.training_file.write_bytes(b"original training data")
    settings.adapter_path.mkdir()
    summary = TrainingSummary(
        base_model=settings.base_model,
        base_model_revision=settings.base_model_revision,
        training_sha256=hashlib.sha256(settings.training_file.read_bytes()).hexdigest(),
        adapter_path=settings.adapter_path,
        epochs=1,
        learning_rate=0.0002,
        max_sequence_length=256,
        lora_target_modules=["q_proj", "v_proj"],
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        random_seed=42,
        trainable_parameters=1,
        total_parameters=100,
        trainable_percentage=1.0,
        epoch_losses=[1.0],
    ).model_dump(mode="json")
    if change == "legacy":
        summary.pop("training_sha256")
    summary_path = settings.adapter_path / "training_summary.json"
    if change != "missing":
        summary_path.write_text(json.dumps(summary))
    if change == "data":
        settings.training_file.write_bytes(b"changed training data")
    elif change == "revision":
        settings = replace(settings, base_model_revision="different-revision")
    elif change == "model":
        settings = replace(settings, base_model="different-model")
    if change == "none":
        validate_adapter(settings)
    else:
        with pytest.raises((ValueError, FileNotFoundError), match="training command"):
            validate_adapter(settings)
