"""Validate the knowledge base, training examples, and evaluation cases.

Run before training so broken citations or leaked evaluation questions are
caught early. Errors block training; warnings are reported for review.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
TRAINING_PATH = Path("data/training/instruction_examples.jsonl")
EVALUATION_PATH = Path("data/evaluation/cases.json")

TRAINING_FIELDS = ("id", "context", "question", "answer", "reference_ids")
EVALUATION_FIELDS = ("id", "question", "expected_sources", "should_abstain")
CITATION_PATTERN = re.compile(r"\[([a-z0-9-]+)\]")


@dataclass
class ValidationReport:
    """Collected problems; ``ok`` is False when any error was found."""

    source_ids: list[str]
    training_count: int
    evaluation_count: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_training(path: Path = TRAINING_PATH) -> list[dict]:
    """Read one JSON object per non-empty line."""

    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_evaluation(path: Path = EVALUATION_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of word sets, used to flag near-duplicate questions."""

    left, right = set(_normalize(a).split()), set(_normalize(b).split())
    return len(left & right) / len(left | right) if left | right else 0.0


def validate(
    knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR,
    training: list[dict] | None = None,
    evaluation: list[dict] | None = None,
    near_duplicate_threshold: float = 0.6,
) -> ValidationReport:
    """Check all three data layers and return every problem found."""

    source_ids = sorted(path.stem for path in knowledge_base_dir.glob("*.md"))
    training = training if training is not None else load_training()
    evaluation = evaluation if evaluation is not None else load_evaluation()
    report = ValidationReport(source_ids, len(training), len(evaluation))
    known = set(source_ids)

    # Training examples: required fields, known sources, valid citations.
    seen_training_ids: set[str] = set()
    for index, example in enumerate(training):
        example_id = example.get("id") or f"<line {index + 1}>"
        missing = [name for name in TRAINING_FIELDS if not example.get(name)]
        if missing:
            report.errors.append(f"{example_id}: missing or empty {missing}")
            continue
        if example_id in seen_training_ids:
            report.errors.append(f"{example_id}: duplicate training ID")
        seen_training_ids.add(example_id)

        for source_id in example["reference_ids"]:
            if source_id not in known:
                report.errors.append(f"{example_id}: unknown reference_id '{source_id}'")
        for source_id in CITATION_PATTERN.findall(example["answer"]):
            if source_id not in known:
                report.errors.append(f"{example_id}: answer cites unknown source '{source_id}'")
            elif source_id not in example["reference_ids"]:
                report.errors.append(
                    f"{example_id}: answer cites '{source_id}' not listed in reference_ids"
                )
        if not CITATION_PATTERN.search(example["answer"]):
            report.warnings.append(f"{example_id}: answer has no [source] citation")

    # Evaluation cases: required fields, unique IDs, known sources.
    seen_evaluation_ids: set[str] = set()
    for case in evaluation:
        case_id = case.get("id", "<no id>")
        missing = [name for name in EVALUATION_FIELDS if name not in case]
        if missing:
            report.errors.append(f"{case_id}: missing {missing}")
            continue
        if case_id in seen_evaluation_ids:
            report.errors.append(f"{case_id}: duplicate evaluation ID")
        seen_evaluation_ids.add(case_id)
        for source_id in case["expected_sources"]:
            if source_id not in known:
                report.errors.append(f"{case_id}: unknown expected source '{source_id}'")
        if case["should_abstain"] and case["expected_sources"]:
            report.warnings.append(f"{case_id}: abstain case lists expected sources")

    # Separation: no shared IDs or questions between training and evaluation.
    for shared_id in seen_training_ids & seen_evaluation_ids:
        report.errors.append(f"{shared_id}: ID used in both training and evaluation")
    training_questions = {_normalize(e["question"]): e["id"] for e in training if e.get("question")}
    for case in evaluation:
        normalized = _normalize(case.get("question", ""))
        if normalized in training_questions:
            report.errors.append(
                f"{case['id']}: question duplicates training example "
                f"{training_questions[normalized]}"
            )
            continue
        for example in training:
            overlap = _token_overlap(case["question"], example.get("question", ""))
            if overlap >= near_duplicate_threshold:
                report.warnings.append(
                    f"{case['id']}: {overlap:.0%} word overlap with {example['id']}"
                )

    return report
