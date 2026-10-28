"""Load and validate supervised and held-out project data."""

import json
from pathlib import Path

from .documents import load_documents
from .generation import citations_in
from .models import DataSummary, EvaluationCase, TrainingExample


def load_training_examples(
    path: Path, *, content: bytes | None = None
) -> list[TrainingExample]:
    """Parse JSON Lines examples with line-specific validation errors."""

    examples: list[TrainingExample] = []
    for line_number, line in enumerate(
        (path.read_bytes() if content is None else content).splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            examples.append(TrainingExample.model_validate_json(line))
        except ValueError as error:
            raise ValueError(
                f"Invalid training record at {path}:{line_number}"
            ) from error
    if not examples:
        raise ValueError(f"No training examples found in {path}")
    return examples


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Parse the shared held-out benchmark."""

    records = json.loads(path.read_text(encoding="utf-8"))
    cases = [EvaluationCase.model_validate(record) for record in records]
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def _unique_ids(values: list[str], label: str) -> None:
    """Reject duplicate IDs because they make traces and metrics ambiguous."""

    if len(values) != len(set(values)):
        raise ValueError(f"{label} IDs must be unique")


def validate_data(settings_path: tuple[Path, Path, Path]) -> DataSummary:
    """Verify sources, training examples, and evaluation cases agree."""

    source_dir, training_file, evaluation_file = settings_path
    documents = load_documents(source_dir)
    examples = load_training_examples(training_file)
    cases = load_evaluation_cases(evaluation_file)
    source_ids = {document.document_id for document in documents}

    _unique_ids([example.id for example in examples], "Training example")
    _unique_ids([case.id for case in cases], "Evaluation case")

    for example in examples:
        missing = set(example.reference_ids) - source_ids
        if missing:
            raise ValueError(
                f"Training example {example.id} references missing sources: "
                f"{sorted(missing)}"
            )
        if not all(
            f"[{source_id}]" in example.context for source_id in example.reference_ids
        ):
            raise ValueError(
                f"Training example {example.id} context must label every source ID"
            )
        if not set(citations_in(example.answer)).issubset(example.reference_ids):
            raise ValueError(
                f"Training example {example.id} answer cites a source absent from its context"
            )

    for case in cases:
        missing = set(case.expected_sources) - source_ids
        if missing:
            raise ValueError(
                f"Evaluation case {case.id} references missing sources: "
                f"{sorted(missing)}"
            )

    training_questions = {example.question.casefold() for example in examples}
    leaked_cases = [
        case.id for case in cases if case.question.casefold() in training_questions
    ]
    if leaked_cases:
        raise ValueError(f"Evaluation questions leaked into training: {leaked_cases}")

    return DataSummary(
        source_count=len(documents),
        training_example_count=len(examples),
        evaluation_case_count=len(cases),
        source_ids=sorted(source_ids),
    )
