"""Tests for the benchmark and required report sections."""

import pytest

from solution.hybrid_project.datasets import load_evaluation_cases
from solution.hybrid_project.evaluation import (
    _contains_term,
    _style_compliance,
    _term_coverage,
    evaluate,
    evaluate_system,
    render_report,
)
from solution.hybrid_project.pipeline import ComparisonPipeline


def test_benchmark_compares_all_systems(
    smoke_pipeline: ComparisonPipeline,
) -> None:
    """One run should retain base-only, baseline RAG, and hybrid rows."""

    summary = evaluate(smoke_pipeline)

    assert [system.system for system in summary.systems] == [
        "base-only",
        "rag-base",
        "rag-lora",
    ]
    assert all(len(system.rows) == 14 for system in summary.systems)


def test_report_contains_decision_evidence(
    smoke_pipeline: ComparisonPipeline,
) -> None:
    """The generated report must support a baseline-versus-hybrid decision."""

    summary = evaluate(smoke_pipeline)
    report = render_report(
        smoke_pipeline.settings,
        smoke_pipeline,
        summary,
    )

    assert "## Shared Retrieval Results" in report
    assert "## Hybrid Delta Against Baseline RAG" in report
    assert "## Failure And Fragility Analysis" in report
    assert "## Threshold Decision" in report
    assert "## Recommendation" in report
    assert "### Content Versus Citation Format" in report
    assert "## Cost And Limitations" in report
    assert "Prompt:" in report
    assert "Inference prompt limit:" in report


def test_canonical_abstentions_receive_credit_on_unsupported_cases(
    smoke_pipeline: ComparisonPipeline,
) -> None:
    """The smoke baseline returns a real abstention when it has no evidence."""

    summary = evaluate(smoke_pipeline)
    unsupported_ids = {row.id for row in summary.retrieval_rows if row.should_abstain}
    assert unsupported_ids
    base_only = next(
        system for system in summary.systems if system.system == "base-only"
    )
    for row in base_only.rows:
        if row.id in unsupported_ids:
            assert row.status == "abstained"
            assert row.relevance == 1.0


def test_terms_are_not_arbitrary_substrings() -> None:
    assert _term_coverage("A note about the 30-day policy.", ["not", "30-day"]) == 0.5
    assert _contains_term("It is NOT allowed.", "not")
    assert not _contains_term("Invoice number: 130 days", "30 days")
    assert _contains_term("Wait 30 days.", "30 days")


def test_rejected_content_remains_diagnostic_only(smoke_pipeline) -> None:
    class UncitedGenerator:
        mode = "uncited-test"

        def generate(self, question, hits):
            return "Standard shipping takes three to five business days."

    pipeline = ComparisonPipeline(
        smoke_pipeline.settings,
        smoke_pipeline.retriever,
        UncitedGenerator(),
        smoke_pipeline.lora_generator,
    )
    case = next(
        c
        for c in load_evaluation_cases(pipeline.settings.evaluation_file)
        if c.id == "eval-standard-shipping"
    )
    row = evaluate_system(pipeline, [case], "rag-base").rows[0]
    assert row.status == "invalid-citations"
    assert row.relevance == 0
    assert row.raw_relevance == 1
    assert row.raw_answer == "Standard shipping takes three to five business days."
    assert row.answer != row.raw_answer
    unsupported = next(
        c
        for c in load_evaluation_cases(pipeline.settings.evaluation_file)
        if c.should_abstain
    )
    row = evaluate_system(pipeline, [unsupported], "base-only").rows[0]
    assert row.status == "answered"
    assert row.relevance == 0
    assert not row.decision_correct


def test_style_gain_cannot_hide_content_regression(smoke_pipeline) -> None:
    summary = evaluate(smoke_pipeline)
    baseline, hybrid = summary.systems[1:]
    baseline.answer_relevance, hybrid.answer_relevance = 0.8, 0.7
    baseline.style_compliance, hybrid.style_compliance = 0.5, 1.0
    assert "Do not promote" in render_report(
        smoke_pipeline.settings, smoke_pipeline, summary
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [("answered", 1.0), ("abstained", 1.0), ("invalid-citations", 0.0)],
)
def test_style_score_does_not_reward_a_rejection_template(
    status: str, expected: float
) -> None:
    """A short validation message is not a successful model response."""

    answer = (
        "The generated answer failed citation validation. Escalate for manual review."
    )
    assert _style_compliance(answer, status) == expected
