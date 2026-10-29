"""Benchmark baseline RAG against hybrid RAG + LoRA on the held-out cases.

Retrieval is evaluated once and its results are shared: both systems answer
every case from the same evidence and are scored with the same deterministic
rubric. Outputs:

- ``reports/evaluation_results.json``: full trace (evidence, raw and served answers, scores),
- ``reports/evaluation_report.md``: generated tables plus the hand-written
  ``reports/analysis.md`` (failure analysis and recommendation).

Run the benchmark, or rebuild the report from the saved trace after editing the analysis:

    uv run python -m src.evaluation
    uv run python -m src.evaluation --report-only
"""

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from src.datasets import load_evaluation
from src.pipeline import Answer, answer_question
from src.prompts import SYSTEM_INSTRUCTION
from src.retrieval import Retriever, build_chunks, evaluate_retrieval
from src.training import SUMMARY_FILENAME, TrainingConfig

REPORTS_DIR = Path("reports")
RESULTS_PATH = REPORTS_DIR / "evaluation_results.json"
REPORT_PATH = REPORTS_DIR / "evaluation_report.md"
ANALYSIS_PATH = REPORTS_DIR / "analysis.md"
CHUNK_MAX_CHARS = 600
THRESHOLD_SWEEP = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# Style rubric, from data/knowledge_base/response-style.md.
MAX_WORDS = 60
MAX_FIRST_SENTENCE_WORDS = 25
FILLER_OPENINGS = ("thank you", "thanks", "dear", "hello", "hi ", "i'm sorry", "i am sorry", "sure", "as an ai")
HEDGES = ("i think", "maybe", "perhaps", "probably", "might be", "not sure", "it depends")


# ---------------------------------------------------------------------------
# Deterministic rubric
# ---------------------------------------------------------------------------


def contains_term(text: str, term: str) -> bool:
    """Case-insensitive whole-phrase match, so "not" does not match "note"."""

    return re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", text.lower()) is not None


def term_coverage(text: str, terms: list[str]) -> float:
    """Fraction of required terms present in the text."""

    if not terms:
        return 1.0
    return sum(contains_term(text, term) for term in terms) / len(terms)


def style_checks(text: str) -> dict[str, bool]:
    """Concise, direct first sentence, no greeting/apology opener, no hedging."""

    lowered = text.lower().strip()
    first_sentence = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return {
        "concise": len(text.split()) <= MAX_WORDS,
        "short_first_sentence": len(first_sentence.split()) <= MAX_FIRST_SENTENCE_WORDS,
        "no_filler_opening": not lowered.startswith(FILLER_OPENINGS),
        "no_hedging": not any(hedge in lowered for hedge in HEDGES),
    }


def score_case(case: dict, answer: Answer) -> dict:
    """Score one served answer.

    - relevance: required-term coverage of the served answer (0 if not answered);
      unsupported cases score 1 only when the system abstained,
    - raw_coverage: the same coverage on the raw output before citation checks
      (diagnostic only: separates content from citation format),
    - faithful: answered, cites an expected source, and no forbidden claim;
      unsupported cases are faithful only when the system abstained,
    - style: share of style checks passed; the fallback reply scores 1, a rejection 0,
    - decision_correct: answered a supported case or abstained on an unsupported one.
    """

    answered = answer.status == "answered"
    abstained = answer.status == "abstained"
    if case["should_abstain"]:
        relevance, raw_coverage, forbidden, cites_expected = float(abstained), None, [], None
        faithful = decision_correct = abstained
    else:
        relevance = term_coverage(answer.text, case["required_terms"]) if answered else 0.0
        raw_coverage = term_coverage(answer.raw, case["required_terms"]) if answer.raw else 0.0
        forbidden = [term for term in case["forbidden_terms"] if contains_term(answer.raw, term)]
        cites_expected = bool(set(answer.citations) & set(case["expected_sources"]))
        faithful = answered and cites_expected and not forbidden
        decision_correct = answered

    checks = style_checks(answer.text) if answered else {}
    if answered:
        style = sum(checks.values()) / len(checks)
    else:
        style = 1.0 if abstained else 0.0

    return {
        "relevance": relevance,
        "raw_coverage": raw_coverage,
        "faithful": faithful,
        "forbidden_found": forbidden,
        "cites_expected": cites_expected,
        "style": style,
        "style_checks": checks,
        "decision_correct": decision_correct,
    }


def aggregate(rows: list[dict]) -> dict:
    """System-level averages over all cases (raw coverage over supported cases)."""

    supported = [r for r in rows if not r["should_abstain"]]
    generated = [r for r in rows if r["answer"]["raw"]]
    statuses = [r["answer"]["status"] for r in rows]
    return {
        "answered": statuses.count("answered"),
        "abstained": statuses.count("abstained"),
        "rejected": statuses.count("rejected"),
        "relevance": mean(r["score"]["relevance"] for r in rows),
        "raw_coverage_supported": mean(r["score"]["raw_coverage"] for r in supported),
        "faithfulness": mean(float(r["score"]["faithful"]) for r in rows),
        "style": mean(r["score"]["style"] for r in rows),
        "decision_accuracy": mean(float(r["score"]["decision_correct"]) for r in rows),
        "mean_latency": mean(r["answer"]["latency"] for r in rows),
        "median_generation_latency": median(r["answer"]["latency"] for r in generated)
        if generated
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------


def run_benchmark() -> dict:
    """Retrieve once, answer with both systems, score, and return the full trace."""

    from src.generation import make_generator

    cases = load_evaluation()
    training_config = TrainingConfig()
    summary = json.loads(
        (training_config.adapter_path / SUMMARY_FILENAME).read_text(encoding="utf-8")
    )

    chunks = build_chunks(max_chars=CHUNK_MAX_CHARS)
    retriever = Retriever()
    retriever.index(chunks)
    retrieval_report = evaluate_retrieval(retriever, cases)

    # One retrieval per case, reused by both systems so the evidence is identical.
    retrievals, retrieval_latency = {}, {}
    for case in cases:
        started = perf_counter()
        retrievals[case["id"]] = retriever.retrieve(case["question"])
        retrieval_latency[case["id"]] = perf_counter() - started

    systems = {"baseline-rag": None, "hybrid-rag-lora": training_config.adapter_path}
    results: dict[str, dict] = {}
    for name, adapter_path in systems.items():
        started = perf_counter()
        generate = make_generator(adapter_path)
        load_seconds = perf_counter() - started
        rows = []
        for case in cases:
            answer = answer_question(name, retrievals[case["id"]], generate)
            rows.append(
                {
                    "id": case["id"],
                    "should_abstain": case["should_abstain"],
                    "answer": asdict(answer),
                    "score": score_case(case, answer),
                }
            )
            print(f"[{name}] {case['id']:<28} {answer.status:<9} {answer.latency:5.2f}s")
        results[name] = {
            "adapter_path": adapter_path.as_posix() if adapter_path else None,
            "load_seconds": load_seconds,
            "aggregate": aggregate(rows),
            "cases": rows,
        }

    return {
        "config": {
            "retrieval": {
                "embedding_model": retriever.embedding_model,
                "embedding_revision": retriever.embedding_revision,
                "vector_store": "Qdrant local mode, cosine distance",
                "chunking": f"paragraph-aware, merged up to {CHUNK_MAX_CHARS} characters",
                "chunks": len(chunks),
                "top_k": retriever.top_k,
                "min_score": retriever.min_score,
            },
            "generation": {
                "decoding": "greedy (do_sample=False)",
                "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "80")),
                "system_instruction": SYSTEM_INSTRUCTION,
            },
            "training": {k: v for k, v in summary.items() if k != "step_losses"},
        },
        "retrieval": {
            **retrieval_report,
            "mean_latency": mean(retrieval_latency.values()),
            "evidence": {
                case_id: [
                    {"chunk_id": h.chunk.chunk_id, "score": round(h.score, 3)} for h in r.hits
                ]
                for case_id, r in retrievals.items()
            },
        },
        "cases": cases,
        "systems": results,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def _yes_no(value) -> str:
    return "-" if value is None else ("yes" if value else "no")


def build_report(results: dict) -> str:
    """Render the Markdown decision report from a benchmark trace."""

    config, retrieval, systems = results["config"], results["retrieval"], results["systems"]
    cases = {case["id"]: case for case in results["cases"]}
    names = list(systems)
    training = config["training"]
    lines = ["# Evaluation Report: Baseline RAG vs Hybrid RAG + LoRA", ""]
    lines += [
        "Generated by `uv run python -m src.evaluation` from `reports/evaluation_results.json`. "
        "The analysis section is written by hand in `reports/analysis.md`.",
        "",
    ]

    # Configuration
    r, g = config["retrieval"], config["generation"]
    lines += ["## 1. Configuration", "", "### Retrieval (shared by both systems)", ""]
    lines += [
        f"- Embedding model: `{r['embedding_model']}` @ `{r['embedding_revision']}`",
        f"- Vector store: {r['vector_store']}; {r['chunks']} chunks ({r['chunking']})",
        f"- Chunk payload: document ID, title, chunk position, source text",
        f"- `top_k={r['top_k']}`, evidence threshold `min_score={r['min_score']}` "
        "(below it the system abstains without generating)",
        "",
        "### Base model and generation (shared by both systems)",
        "",
        f"- Base model: `{training['base_model']}` @ `{training['base_model_revision']}`",
        f"- Prompt: {training['prompt_format']}",
        f"- Evidence format: one `[document-id] chunk text` line per retrieved chunk",
        f"- Decoding: {g['decoding']}, `max_new_tokens={g['max_new_tokens']}`",
        f"- System instruction: \"{g['system_instruction']}\"",
        "",
        "### Adapter (hybrid system only)",
        "",
        f"- Library: Hugging Face PEFT LoRA; output path `{training['adapter_path']}`",
        f"- Target modules `{training['target_modules']}`, rank `{training['lora_rank']}`, "
        f"alpha `{training['lora_alpha']}`, dropout `{training['lora_dropout']}`",
        f"- Max sequence length `{training['max_sequence_length']}`, epochs `{training['epochs']}`, "
        f"learning rate `{training['learning_rate']}`, seed `{training['seed']}`, "
        "AdamW, batch size 1, prompt tokens masked from the loss",
        f"- Trainable parameters `{training['trainable_parameters']:,}` of "
        f"`{training['total_parameters']:,}` ({training['trainable_percentage']:.3f}%)",
        f"- {training['training_examples']} training examples, SHA-256 `{training['training_sha256']}`",
        f"- Training loss {training['epoch_losses'][0]:.3f} → {training['epoch_losses'][-1]:.3f} "
        f"(per epoch: {', '.join(f'{x:.3f}' for x in training['epoch_losses'])})",
        "",
    ]

    # Rubric
    lines += [
        "## 2. Metric rubric",
        "",
        "All metrics are deterministic and computed by `src/evaluation.py`; no LLM judge or API key is used.",
        "",
        "- **Status**: `answered` (all citations were retrieved), `abstained` (no evidence above "
        "`min_score`, or the model gave the fallback reply), `rejected` (no citation, or a citation "
        "that was not retrieved). Rejected and abstained results serve the fallback reply.",
        "- **Recall@k / MRR**: supported cases only, on the ranked document IDs before the threshold.",
        "- **Relevance**: required-term coverage of the served answer (whole-phrase, case-insensitive). "
        "Rejected/abstained supported cases score 0; unsupported cases score 1 only if abstained.",
        "- **Raw coverage**: the same coverage on the raw model output before citation validation "
        "(supported cases). A diagnostic that separates content from citation format.",
        "- **Faithfulness**: answered, cites at least one expected source, and contains no forbidden "
        "claim. Unsupported cases are faithful only if the system abstained. This is a citation "
        "proxy, not a semantic check.",
        f"- **Style**: share of 4 checks — ≤{MAX_WORDS} words, first sentence "
        f"≤{MAX_FIRST_SENTENCE_WORDS} words, no greeting/apology opener, no hedging. "
        "The fallback reply scores 1, a rejection 0.",
        "- **Decision accuracy**: answered a supported case, or abstained on an unsupported one. "
        "A rejection counts as wrong in both cases.",
        "- **Latency**: wall-clock seconds per answer for generation + validation (retrieval is shared "
        "and reported once; model loading is reported under cost).",
        "",
    ]

    # Retrieval
    lines += ["## 3. Retrieval (evaluated once, shared)", ""]
    lines += _table(
        ["Metric", "Value"],
        [
            [f"Recall@{retrieval['top_k']}", f"{retrieval['recall_at_k']:.3f}"],
            ["MRR", f"{retrieval['mrr']:.3f}"],
            ["Retrieval-only decision accuracy", f"{retrieval['decision_accuracy']:.3f}"],
            ["Mean retrieval latency (s)", f"{retrieval['mean_latency']:.3f}"],
        ],
    )
    lines += [""]
    lines += _table(
        ["Case", "Should abstain", "Top score", "Retrieved (ranked)", "Expected", "Abstained"],
        [
            [
                row["id"], _yes_no(row["should_abstain"]), f"{row['top_score']:.3f}",
                ", ".join(row["retrieved"]), ", ".join(row["expected"]) or "-",
                _yes_no(row["abstained"]),
            ]
            for row in retrieval["per_case"]
        ],
    )
    lines += [""]

    # Threshold evidence
    scores = {row["id"]: row["top_score"] for row in retrieval["per_case"]}
    supported = sorted(scores[c] for c in cases if not cases[c]["should_abstain"])
    unsupported = sorted(scores[c] for c in cases if cases[c]["should_abstain"])
    lines += [
        "## 4. Retrieval threshold evidence",
        "",
        f"- Supported top scores: {', '.join(f'{s:.3f}' for s in supported)}",
        f"- Unsupported top scores: {', '.join(f'{s:.3f}' for s in unsupported)}",
        f"- Lowest supported score {min(supported):.3f}; highest unsupported score {max(unsupported):.3f}.",
        "",
    ]
    sweep_rows = []
    for threshold in THRESHOLD_SWEEP:
        missed = sum(s < threshold for s in supported)
        wrong = sum(s >= threshold for s in unsupported)
        accuracy = (len(scores) - missed - wrong) / len(scores)
        marker = " ← chosen" if abs(threshold - retrieval["min_score"]) < 1e-9 else ""
        sweep_rows.append([f"{threshold:.2f}{marker}", f"{accuracy:.3f}", missed, wrong])
    lines += _table(
        ["min_score", "Decision accuracy", "Supported refused", "Unsupported answered"], sweep_rows
    )
    lines += [""]

    # Aggregate comparison
    lines += ["## 5. Aggregate comparison", ""]
    agg = {name: systems[name]["aggregate"] for name in names}
    metric_rows = [
        ("Answered / abstained / rejected", lambda a: f"{a['answered']} / {a['abstained']} / {a['rejected']}"),
        ("Relevance (required-term coverage)", lambda a: f"{a['relevance']:.3f}"),
        ("Raw coverage, supported (diagnostic)", lambda a: f"{a['raw_coverage_supported']:.3f}"),
        ("Faithfulness (citation proxy)", lambda a: f"{a['faithfulness']:.3f}"),
        ("Style compliance", lambda a: f"{a['style']:.3f}"),
        ("Decision accuracy", lambda a: f"{a['decision_accuracy']:.3f}"),
        ("Mean latency per answer (s)", lambda a: f"{a['mean_latency']:.2f}"),
        ("Median generation latency (s)", lambda a: f"{a['median_generation_latency']:.2f}"),
    ]
    rows = [[label] + [fmt(agg[name]) for name in names] for label, fmt in metric_rows]
    if len(names) == 2:
        base, hybrid = agg[names[0]], agg[names[1]]
        deltas = ["", "relevance", "raw_coverage_supported", "faithfulness", "style", "decision_accuracy",
                  "mean_latency", "median_generation_latency"]
        for row, key in zip(rows, deltas, strict=True):
            row.append(f"{hybrid[key] - base[key]:+.3f}" if key else "")
    lines += _table(["Metric", *names, "Δ hybrid − baseline"], rows)
    lines += [""]

    # Per-case
    lines += ["## 6. Per-case evidence", ""]
    per_case_rows = []
    for case_id in cases:
        for name in names:
            row = next(r for r in systems[name]["cases"] if r["id"] == case_id)
            s, a = row["score"], row["answer"]
            per_case_rows.append(
                [
                    case_id, name, a["status"], f"{s['relevance']:.2f}", _yes_no(s["faithful"]),
                    f"{s['style']:.2f}", _yes_no(s["decision_correct"]),
                    ", ".join(a["citations"]) or "-", f"{a['latency']:.2f}",
                ]
            )
    lines += _table(
        ["Case", "System", "Status", "Relevance", "Faithful", "Style", "Decision ok", "Citations", "Latency (s)"],
        per_case_rows,
    )
    lines += ["", "### Raw answers", ""]
    for case_id, case in cases.items():
        evidence = ", ".join(f"{e['chunk_id']} ({e['score']})" for e in retrieval["evidence"][case_id])
        lines += [
            f"**{case_id}** — {case['question']}",
            "",
            f"- Expected: {', '.join(case['expected_sources']) or 'abstain'}; "
            f"required terms: {case['required_terms'] or '-'}; evidence: {evidence or 'none above threshold'}",
        ]
        for name in names:
            row = next(r for r in systems[name]["cases"] if r["id"] == case_id)
            raw = row["answer"]["raw"].replace("\n", " ") or "(not generated)"
            lines.append(f"- `{name}` — **{row['answer']['status']}** ({row['answer']['reason']}): {raw}")
        lines.append("")

    # Cost
    lines += ["## 7. Training and inference cost", ""]
    adapter_dir = Path(training["adapter_path"])
    adapter_mb = (
        sum(p.stat().st_size for p in adapter_dir.iterdir() if p.is_file()) / 1024**2
        if adapter_dir.exists()
        else float("nan")
    )
    lines += _table(
        ["Item", *names],
        [
            ["Training time", "-", f"{training['training_seconds']} s on {training['device']}"],
            ["Extra artifact to store and version", "-", f"{adapter_mb:.1f} MB adapter folder"],
            ["Model load time (s)", *[f"{systems[n]['load_seconds']:.1f}" for n in names]],
            ["Mean latency per answer (s)", *[f"{agg[n]['mean_latency']:.2f}" for n in names]],
        ],
    )
    lines += [""]

    # Hand-written analysis
    lines += ["## 8. Analysis and recommendation", ""]
    if ANALYSIS_PATH.exists():
        lines += [ANALYSIS_PATH.read_text(encoding="utf-8").strip(), ""]
    else:
        lines += ["_Write the analysis in `reports/analysis.md`, then run `python -m src.evaluation --report-only`._", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    REPORTS_DIR.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        results = run_benchmark()
        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(build_report(results), encoding="utf-8")
    for name, system in results["systems"].items():
        a = system["aggregate"]
        print(
            f"{name:<16} relevance={a['relevance']:.3f} faithfulness={a['faithfulness']:.3f} "
            f"style={a['style']:.3f} decision={a['decision_accuracy']:.3f} latency={a['mean_latency']:.2f}s"
        )
    print(f"Wrote {RESULTS_PATH} and {REPORT_PATH}")
