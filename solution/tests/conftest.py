"""Shared deterministic fixtures for fast behavioral tests."""

from dataclasses import replace
from pathlib import Path

import pytest

from solution.hybrid_project.config import Settings
from solution.hybrid_project.embeddings import HashEmbedder
from solution.hybrid_project.generation import ExtractiveSmokeGenerator
from solution.hybrid_project.pipeline import ComparisonPipeline
from solution.hybrid_project.retrieval import Retriever


@pytest.fixture
def smoke_pipeline() -> ComparisonPipeline:
    """Build the full orchestration path without downloading models."""

    root = Path(__file__).parents[2]
    settings = replace(
        Settings.from_env(project_root=root),
        embedding_model="hash-smoke-test-only",
        min_score=0.20,
    )
    retriever = Retriever(settings, embedder=HashEmbedder())
    return ComparisonPipeline(
        settings,
        retriever,
        ExtractiveSmokeGenerator("baseline"),
        ExtractiveSmokeGenerator("hybrid"),
    )
