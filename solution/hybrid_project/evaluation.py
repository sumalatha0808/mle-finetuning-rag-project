"""Shared benchmark metrics and decision-report generation."""

import json
import re
from pathlib import Path
from time import perf_counter

from .config import Settings
from .datasets import load_evaluation_cases
from .models import (
    BenchmarkSummary,
    EvaluationCase,
    RetrievalRow,
    SystemEvaluationRow,
    SystemName,
    SystemSummary,
    TrainingSummary,
)
from .pipeline import ComparisonPipeline
from .retrieval import Retriever

SYSTEMS: tuple[SystemName, ...] = ("base-only", "rag-base", "rag-lora")


def _first_relevant_rank(retrieved: list[str], expected: list[str]) -> int | None:
    """Return the first one-based expected-source rank."""

    for rank, source_id in enumerate(retrieved, start=1):
        if source_id in expected:
            return rank
    return None


def evaluate_retrieval(
    retriever: Retriever,
    cases: list[EvaluationCase],
) -> tuple[float, float, list[RetrievalRow]]:
    """Measure retrieval once because both RAG systems share it."""

    rows: list[RetrievalRow] = []
    for case in cases:
        hits = retriever.search(case.question)
        retrieved = list(dict.fromkeys(hit.document_id for hit in hits))
        rows.append(
            RetrievalRow(
                id=case.id,
                expected_sources=case.expected_sources,
                retrieved_sources=retrieved,
                first_relevant_rank=_first_relevant_rank(
                    retrieved, case.expected_sources
                ),
                top_score=hits[0].score if hits else None,
                should_abstain=case.should_abstain,
            )
        )
    supported = [row for row in rows if not row.should_abstain]
    recall = sum(row.first_relevant_rank is not None for row in supported) / len(
        supported
    )
    mrr = sum(
        0.0 if row.first_relevant_rank is None else 1.0 / row.first_relevant_rank
        for row in supported
    ) / len(supported)
    return recall, mrr, rows


def _contains_term(answer: str, term: str) -> bool:
    """Match a labeled phrase without confusing substrings such as not/note."""

    return (
        re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", answer, re.IGNORECASE)
        is not None
    )


def _term_coverage(answer: str, required_terms: list[str]) -> float:
    """Calculate transparent required-term coverage."""

    if not required_terms:
        return 1.0
    return sum(_contains_term(answer, term) for term in required_terms) / len(
        required_terms
    )


def _style_compliance(answer: str, status: str) -> float:
    """Score accepted responses, never the template replacing a rejected answer."""

    if status == "abstained":
        return 1.0
    if status != "answered":
        return 0.0
    words = answer.split()
    first_sentence = answer.split(".", maxsplit=1)[0].split()
    checks = (
        len(words) <= 80,
        len(first_sentence) <= 30,
        not any(hedge in answer.lower() for hedge in ("maybe", "probably", "perhaps")),
    )
    return sum(checks) / len(checks)


def _faithful(
    case: EvaluationCase,
    status: str,
    answer: str,
    citations: list[str],
) -> bool:
    """Check abstention, expected citations, and forbidden claims."""

    if case.should_abstain:
        return status == "abstained"
    if status != "answered" or not citations:
        return False
    forbidden_present = any(
        _contains_term(answer, term) for term in case.forbidden_terms
    )
    return set(citations).issubset(set(case.expected_sources)) and not forbidden_present


def evaluate_system(
    pipeline: ComparisonPipeline,
    cases: list[EvaluationCase],
    system: SystemName,
) -> SystemSummary:
    """Run one system over every held-out case with identical rubrics."""

    rows: list[SystemEvaluationRow] = []
    for case in cases:
        started = perf_counter()
        result = pipeline.answer(case.question, system)
        latency = perf_counter() - started
        decision_correct = (
            result.status == "abstained"
            if case.should_abstain
            else result.status == "answered"
        )
        relevance = (
            float(result.status == "abstained")
            if case.should_abstain
            else _term_coverage(result.answer, case.required_terms)
            if result.status == "answered"
            else 0.0
        )
        rows.append(
            SystemEvaluationRow(
                id=case.id,
                system=system,
                status=result.status,
                answer=result.answer,
                citations=result.citations,
                raw_answer=result.raw_answer,
                raw_relevance=(
                    _term_coverage(result.raw_answer or "", case.required_terms)
                    if not case.should_abstain
                    else None
                ),
                relevance=relevance,
                faithful=_faithful(
                    case,
                    result.status,
                    result.answer,
                    result.citations,
                ),
                style_compliance=_style_compliance(result.answer, result.status),
                decision_correct=decision_correct,
                latency_seconds=latency,
            )
        )
    return SystemSummary(
        system=system,
        answer_relevance=sum(row.relevance for row in rows) / len(rows),
        faithfulness=sum(row.faithful for row in rows) / len(rows),
        style_compliance=sum(row.style_compliance for row in rows) / len(rows),
        decision_accuracy=sum(row.decision_correct for row in rows) / len(rows),
        mean_latency_seconds=sum(row.latency_seconds for row in rows) / len(rows),
        rows=rows,
    )


def evaluate(pipeline: ComparisonPipeline) -> BenchmarkSummary:
    """Run retrieval and every answer system on the shared benchmark."""

    cases = load_evaluation_cases(pipeline.settings.evaluation_file)
    if pipeline.retriever is None:
        raise ValueError("The benchmark requires a loaded retriever")
    recall, mrr, retrieval_rows = evaluate_retrieval(pipeline.retriever, cases)
    systems = [evaluate_system(pipeline, cases, system) for system in SYSTEMS]
    return BenchmarkSummary(
        retrieval_recall_at_k=recall,
        mean_reciprocal_rank=mrr,
        retrieval_rows=retrieval_rows,
        systems=systems,
    )


def _system(summary: BenchmarkSummary, name: SystemName) -> SystemSummary:
    """Return one named system summary from the benchmark."""

    return next(item for item in summary.systems if item.system == name)


def render_report(
    settings: Settings,
    pipeline: ComparisonPipeline,
    summary: BenchmarkSummary,
) -> str:
    """Create a reviewable Markdown decision report."""

    if pipeline.base_generator is None or pipeline.lora_generator is None:
        raise ValueError("The comparison report requires both generators")
    rag_base = _system(summary, "rag-base")
    rag_lora = _system(summary, "rag-lora")
    relevance_delta = rag_lora.answer_relevance - rag_base.answer_relevance
    faithfulness_delta = rag_lora.faithfulness - rag_base.faithfulness
    style_delta = rag_lora.style_compliance - rag_base.style_compliance
    decision_delta = rag_lora.decision_accuracy - rag_base.decision_accuracy
    supported_scores = [
        row.top_score
        for row in summary.retrieval_rows
        if not row.should_abstain and row.top_score is not None
    ]
    unsupported_scores = [
        row.top_score
        for row in summary.retrieval_rows
        if row.should_abstain and row.top_score is not None
    ]
    weakest_supported = min(supported_scores) if supported_scores else 0.0
    strongest_unsupported = max(unsupported_scores) if unsupported_scores else 0.0
    training_summary_path = settings.adapter_path / "training_summary.json"
    training_summary = (
        TrainingSummary.model_validate_json(
            training_summary_path.read_text(encoding="utf-8")
        )
        if training_summary_path.exists()
        and settings.embedding_model != "hash-smoke-test-only"
        else None
    )
    try:
        adapter_display = settings.adapter_path.relative_to(settings.project_root)
    except ValueError:
        adapter_display = settings.adapter_path

    lines = [
        "# Reference Hybrid Evaluation",
        "",
        "## Configuration",
        "",
        f"- Base model: `{settings.base_model}`",
        f"- Base revision: `{settings.base_model_revision or 'not pinned'}`",
        f"- Adapter path: `{adapter_display}`",
        f"- Embedding model: `{settings.embedding_model}`",
        f"- Embedding revision: `{settings.embedding_revision or 'not pinned'}`",
        "- Vector database: `Qdrant` local mode with cosine similarity",
        f"- Retrieval: `top_k={settings.top_k}`, `min_score={settings.min_score:.2f}`",
        f"- Baseline generator: `{pipeline.base_generator.mode}`",
        f"- Hybrid generator: `{pipeline.lora_generator.mode}`",
        "- Prompt: shared system instruction, `[source-id] text` context, question, assistant answer; native chat template when available, plain-text fallback otherwise",
        f"- Inference prompt limit: `{settings.max_sequence_length}` tokens; generation limit: `{settings.max_new_tokens}` new tokens; greedy decoding",
        "- Overlength prompts and training examples fail explicitly; they are not silently truncated",
    ]
    if training_summary is not None:
        lines.extend(
            [
                f"- Training data SHA-256: `{training_summary.training_sha256}`",
                f"- LoRA targets: `{', '.join(training_summary.lora_target_modules)}`",
                (
                    f"- LoRA rank / alpha / dropout: `{training_summary.lora_rank}` / "
                    f"`{training_summary.lora_alpha}` / "
                    f"`{training_summary.lora_dropout:.2f}`"
                ),
                (
                    f"- Training: `{training_summary.epochs}` epochs, learning rate "
                    f"`{training_summary.learning_rate}`, seed "
                    f"`{training_summary.random_seed}`"
                ),
                f"- Training sequence limit: `{training_summary.max_sequence_length}` tokens (prompt plus answer)",
                "- Optimizer: `AdamW`, batch size `1`, default betas `(0.9, 0.999)`, weight decay `0.01`",
                f"- Training device: `{training_summary.training_device or 'not recorded'}`",
                (
                    f"- Training and save duration: `{training_summary.training_seconds:.1f}` seconds"
                    if training_summary.training_seconds is not None
                    else "- Training duration: not recorded in this adapter"
                ),
                (
                    f"- Trainable parameters: `{training_summary.trainable_parameters}` "
                    f"of `{training_summary.total_parameters}` "
                    f"(`{training_summary.trainable_percentage:.3f}%`)"
                ),
                (
                    f"- Training loss: `{training_summary.epoch_losses[0]:.3f}` to "
                    f"`{training_summary.epoch_losses[-1]:.3f}`"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Metric Rubric",
            "",
            "- `recall@k`: supported cases whose expected source appears in the shared retrieval results.",
            "- `MRR`: reciprocal rank of the first expected source for supported cases.",
            "- `answer relevance`: whole-term coverage in accepted answers; blocked answers score 0 and unsupported cases score 1 only on abstention.",
            "- `faithfulness`: a citation/forbidden-term proxy, not a semantic judge; unsupported cases must abstain. Base-only has no evidence to cite, so this citation-based score is not directly comparable to RAG.",
            "- `style compliance`: concise first sentence, concise total response, and no vague hedge; rejected answers score 0, retrieval abstentions score 1.",
            "- `decision accuracy`: accept answers on supported cases and abstain on unsupported cases, not factual correctness. An always-answer system benefits from a majority of supported cases.",
            "- Lexical labels miss synonyms and some negations. Inspect raw answers before drawing quality conclusions.",
            "- Training and evaluation share policies and some scenarios. This is a small within-domain check, not evidence of generalization to new policies.",
            "",
            "## Shared Retrieval Results",
            "",
            "| Metric | Score |",
            "| --- | ---: |",
            f"| Retrieval recall@{settings.top_k} | {summary.retrieval_recall_at_k:.3f} |",
            f"| Mean reciprocal rank | {summary.mean_reciprocal_rank:.3f} |",
            "",
            "## System Comparison",
            "",
            "| System | Relevance | Faithfulness proxy | Style | Decision accuracy | Mean latency (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for system in summary.systems:
        lines.append(
            f"| {system.system} | {system.answer_relevance:.3f} | "
            f"{system.faithfulness:.3f} | {system.style_compliance:.3f} | "
            f"{system.decision_accuracy:.3f} | "
            f"{system.mean_latency_seconds:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Content Versus Citation Format",
            "",
            "Raw coverage is measured before citation rejection on supported cases (no generation scores 0). It is a lexical diagnostic, not permission to serve a rejected answer. Raw answers remain in the JSON trace.",
            "",
            "| System | Raw coverage (supported) | Citation rejections |",
            "| --- | ---: | ---: |",
        ]
    )
    for system in summary.systems:
        raw_scores = [
            row.raw_relevance for row in system.rows if row.raw_relevance is not None
        ]
        raw_mean = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
        rejected = sum(row.status == "invalid-citations" for row in system.rows)
        lines.append(
            f"| {system.system} | {raw_mean:.3f} | {rejected}/{len(system.rows)} |"
        )
    lines.extend(
        [
            "",
            "## Hybrid Delta Against Baseline RAG",
            "",
            f"- Answer relevance delta: `{relevance_delta:+.3f}`",
            f"- Faithfulness delta: `{faithfulness_delta:+.3f}`",
            f"- Style-compliance delta: `{style_delta:+.3f}`",
            "",
            "## Per-Case Comparison",
            "",
            "| ID | System | Status | Relevance | Citation/term check | Style | Citations |",
            "| --- | --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    for system in summary.systems:
        for row in system.rows:
            citations = ", ".join(row.citations) if row.citations else "-"
            lines.append(
                f"| {row.id} | {row.system} | {row.status} | "
                f"{row.relevance:.2f} | {'yes' if row.faithful else 'no'} | "
                f"{row.style_compliance:.2f} | {citations} |"
            )
    lines.extend(["", "## Failure And Fragility Analysis", ""])
    cases = {case.id: case for case in load_evaluation_cases(settings.evaluation_file)}
    failure_count = 0
    for system in summary.systems:
        system_failures = [
            row
            for row in system.rows
            if (not row.decision_correct or not row.faithful or row.relevance < 1.0)
        ]
        # Surface unsupported answers before easier formatting failures.
        system_failures.sort(key=lambda row: not cases[row.id].should_abstain)
        for row in system_failures[:3]:
            failure_count += 1
            case = cases[row.id]
            if row.status == "invalid-citations":
                reason = "Citation formatting or source IDs failed validation; inspect the raw trace before attributing this to content quality."
            elif row.status == "abstained":
                reason = "A supported question was left unanswered; inspect retrieval scores and the raw answer to distinguish threshold rejection from model abstention."
            elif case.should_abstain:
                reason = "The system answered a question not covered by the policies; similarity alone does not establish answerability."
            else:
                missing = [
                    term
                    for term in case.required_terms
                    if not _contains_term(row.answer, term)
                ]
                reason = f"Missing labeled terms: {', '.join(missing) or 'none'}. Check the answer against the policy; term coverage alone cannot establish correctness."
                if not row.faithful and row.system != "base-only":
                    reason += " Expected-source citations are missing/wrong or a forbidden phrase is present."
            lines.append(
                f"- `{row.system}` on `{row.id}`: status `{row.status}`, "
                f"relevance `{row.relevance:.2f}`, citation/term check "
                f"`{'yes' if row.faithful else 'no'}`. {reason}"
            )
    if failure_count == 0:
        lines.append(
            "- No supplied case fails the deterministic rubric. The benchmark "
            "is small and does not establish production reliability."
        )
    lines.extend(
        [
            (
                f"- Shared retrieval recall is "
                f"`{summary.retrieval_recall_at_k:.3f}`; failures after successful "
                "retrieval are generation, citation, or answer-coverage failures."
            ),
            (
                f"- The adapter changes faithfulness by "
                f"`{faithfulness_delta:+.3f}` and decision accuracy from "
                f"`{rag_base.decision_accuracy:.3f}` to "
                f"`{rag_lora.decision_accuracy:.3f}`. Remaining invalid citations "
                "still require rejection rather than automatic repair."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Threshold Decision",
            "",
            (
                f"The shared threshold is `{settings.min_score:.2f}`. The weakest "
                f"supported top score is `{weakest_supported:.3f}` and the strongest "
                f"unsupported top score is `{strongest_unsupported:.3f}`. The same "
                "threshold is applied to baseline RAG and hybrid RAG so adapter "
                "quality cannot hide retrieval uncertainty."
            ),
            "These are benchmark diagnostics, not a separate calibration set. Do not tune the threshold on these cases and call the resulting score held-out performance.",
            (
                "Supported and unsupported scores overlap: no single similarity threshold separates these cases. Inspect answerability, not just nearest-neighbor similarity."
                if supported_scores
                and unsupported_scores
                and strongest_unsupported >= weakest_supported
                else "The observed scores do not establish a reliable threshold for unseen questions."
            ),
            "",
            "## Cost And Limitations",
            "",
            "Training adds an optimization run and an adapter to version, validate, and retrain when data changes. The frozen base model still occupies memory and participates in forward/backward computation; a small trainable percentage is not the same percentage of total memory or compute.",
            "Inference latencies above exclude model loading and index construction. Both model instances are loaded together for this comparison, so it is not a deployment memory estimate. Peak memory and energy were not measured; local execution does not mean zero compute cost.",
            "Citation-format gains can dominate the accepted-answer metrics. Compare raw coverage and rejected traces, then consider a shared citation example and counterfactual policy values as separate experiments before claiming improved understanding.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if (
        relevance_delta >= 0
        and faithfulness_delta >= 0
        and decision_delta >= 0
        and (relevance_delta > 0 or faithfulness_delta > 0)
    ):
        lines.append(
            "Keep the hybrid path as an experiment, not as an automatic "
            "replacement. Content/citation proxies improve without a regression "
            "in these metrics or answer/abstain decisions, but the "
            "failure rows and added adapter lifecycle still require review."
        )
    else:
        lines.append(
            "Do not promote the hybrid path yet. The adapter does not improve "
            "content/citation proxies without regressions. Style alone does not "
            "justify training and maintenance."
        )
    lines.extend(
        [
            "",
            "## Prioritized Next Experiments",
            "",
            "1. Expand held-out paraphrases and policy-change cases before tuning the adapter again.",
            "2. Compare a stronger small instruction model using the same retrieval and metric contracts.",
            "3. Review invalid-citation cases and adjust training data only after identifying a repeatable failure pattern.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    pipeline: ComparisonPipeline,
    output_path: Path,
) -> BenchmarkSummary:
    """Run the benchmark and write Markdown plus a regenerated JSON trace."""

    summary = evaluate(pipeline)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_report(pipeline.settings, pipeline, summary),
        encoding="utf-8",
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return summary
