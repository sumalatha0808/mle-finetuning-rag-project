"""Environment-backed configuration for training and evaluation."""

import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_BASE_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@dataclass(frozen=True)
class Settings:
    """All paths and controls needed by the local project workflow."""

    project_root: Path
    knowledge_base_dir: Path
    training_file: Path
    evaluation_file: Path
    adapter_path: Path
    storage_path: Path | None
    base_model: str
    embedding_model: str
    top_k: int
    min_score: float
    max_sequence_length: int
    max_new_tokens: int
    train_epochs: int
    learning_rate: float
    random_seed: int
    base_model_revision: str | None = None
    embedding_revision: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid controls before loading models or writing artifacts."""

        for name, value in (
            ("TOP_K", self.top_k),
            ("MAX_SEQUENCE_LENGTH", self.max_sequence_length),
            ("MAX_NEW_TOKENS", self.max_new_tokens),
            ("TRAIN_EPOCHS", self.train_epochs),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("LEARNING_RATE must be finite and positive")
        if not -1 <= self.min_score <= 1:
            raise ValueError("MIN_SCORE must be a finite cosine score between -1 and 1")
        if not 0 <= self.random_seed < 2**32:
            raise ValueError("RANDOM_SEED must be between 0 and 2**32 - 1")

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        """Load repository defaults and allow explicit environment overrides."""

        root = project_root or Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")
        adapter_value = os.getenv("ADAPTER_PATH", "artifacts/support-lora-adapter")
        storage_value = os.getenv("VECTOR_STORAGE_PATH", "")
        base_model = os.getenv("BASE_MODEL", DEFAULT_BASE_MODEL)
        embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        return cls(
            project_root=root,
            knowledge_base_dir=root / "data" / "knowledge_base",
            training_file=root / "data" / "training" / "instruction_examples.jsonl",
            evaluation_file=root / "data" / "evaluation" / "cases.json",
            adapter_path=root / adapter_value,
            storage_path=root / storage_value if storage_value else None,
            base_model=base_model,
            embedding_model=embedding_model,
            top_k=int(os.getenv("TOP_K", "3")),
            min_score=float(os.getenv("MIN_SCORE", "0.30")),
            max_sequence_length=int(os.getenv("MAX_SEQUENCE_LENGTH", "256")),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "80")),
            train_epochs=int(os.getenv("TRAIN_EPOCHS", "8")),
            learning_rate=float(os.getenv("LEARNING_RATE", "0.0002")),
            random_seed=int(os.getenv("RANDOM_SEED", "42")),
            base_model_revision=os.getenv("BASE_MODEL_REVISION")
            or (DEFAULT_BASE_REVISION if base_model == DEFAULT_BASE_MODEL else None),
            embedding_revision=os.getenv("EMBEDDING_REVISION")
            or (
                DEFAULT_EMBEDDING_REVISION
                if embedding_model == DEFAULT_EMBEDDING_MODEL
                else None
            ),
        )
