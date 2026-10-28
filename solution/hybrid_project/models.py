"""Typed records shared across data loading, training, retrieval, and evaluation."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SystemName = Literal["base-only", "rag-base", "rag-lora"]
AnswerStatus = Literal["answered", "abstained", "invalid-citations"]


class SourceDocument(BaseModel):
    """One approved Markdown policy source."""

    document_id: str
    title: str
    text: str
    path: Path


class Chunk(BaseModel):
    """One retrievable policy passage with stable citation metadata."""

    chunk_id: str
    document_id: str
    title: str
    text: str
    position: int


class SearchHit(Chunk):
    """A retrieved chunk paired with its cosine-similarity score."""

    score: float


class TrainingExample(BaseModel):
    """One supervised context-question-answer example for LoRA training."""

    id: str
    reference_ids: list[str]
    context: str
    question: str
    answer: str


class EvaluationCase(BaseModel):
    """One held-out case used by every compared system."""

    id: str
    question: str
    expected_sources: list[str]
    required_terms: list[str]
    forbidden_terms: list[str]
    should_abstain: bool


class DataSummary(BaseModel):
    """Audit summary proving that project data contracts are consistent."""

    source_count: int
    training_example_count: int
    evaluation_case_count: int
    source_ids: list[str]


class TrainingSummary(BaseModel):
    """Reproducible record of one completed LoRA optimization run."""

    base_model: str
    base_model_revision: str | None
    training_sha256: str
    adapter_path: Path
    epochs: int
    learning_rate: float
    max_sequence_length: int
    lora_target_modules: list[str]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    random_seed: int
    trainable_parameters: int
    total_parameters: int
    trainable_percentage: float
    epoch_losses: list[float]
    training_seconds: float | None = None
    training_device: str | None = None


class Answer(BaseModel):
    """Final answer contract shared by all benchmark systems."""

    system: SystemName
    question: str
    answer: str
    status: AnswerStatus
    raw_answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    top_score: float | None = None


class RetrievalRow(BaseModel):
    """Per-question retrieval evidence used for recall and MRR."""

    id: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    first_relevant_rank: int | None
    top_score: float | None
    should_abstain: bool


class SystemEvaluationRow(BaseModel):
    """Per-question answer evidence for one compared system."""

    id: str
    system: SystemName
    status: AnswerStatus
    answer: str
    citations: list[str]
    raw_answer: str | None = None
    raw_relevance: float | None = None
    relevance: float
    faithful: bool
    style_compliance: float
    decision_correct: bool
    latency_seconds: float


class SystemSummary(BaseModel):
    """Aggregate answer metrics for one system."""

    system: SystemName
    answer_relevance: float
    faithfulness: float
    style_compliance: float
    decision_accuracy: float
    mean_latency_seconds: float
    rows: list[SystemEvaluationRow]


class BenchmarkSummary(BaseModel):
    """Complete retrieval and answer comparison for one benchmark run."""

    retrieval_recall_at_k: float
    mean_reciprocal_rank: float
    retrieval_rows: list[RetrievalRow]
    systems: list[SystemSummary]
