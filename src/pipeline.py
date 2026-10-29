"""Answer questions with baseline RAG or hybrid RAG + LoRA and validate citations.

Both systems receive the same ``RetrievalResult``, so evidence, prompt, and
threshold are identical; the only difference is the ``generate`` function
(base model or base model + adapter). Every answer ends in one of three states:

- ``abstained``: no evidence passed ``min_score``, or the model gave the fallback reply,
- ``rejected``: the model gave no citation, or cited a source that was not retrieved,
- ``answered``: every citation refers to retrieved evidence.

Compare both systems on one question:

    uv run python -m src.pipeline --question "Can support change a delivery address after packing?"
"""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from src.datasets import CITATION_PATTERN
from src.prompts import FALLBACK_ANSWER, evidence_text
from src.retrieval import RetrievalResult

ABSTAIN_MARKER = "approved material is insufficient"


@dataclass
class Answer:
    """One system's outcome for one question; ``raw`` keeps the unvalidated output."""

    system: str
    question: str
    status: str
    text: str
    raw: str
    citations: list[str]
    retrieved_ids: list[str]
    reason: str
    latency: float


def validate_answer(raw: str, retrieved_ids: list[str]) -> tuple[str, str, list[str]]:
    """Return (status, reason, citations) for a generated answer."""

    citations = list(dict.fromkeys(CITATION_PATTERN.findall(raw)))
    if ABSTAIN_MARKER in raw.lower():
        return "abstained", "model replied that the evidence is insufficient", citations
    if not citations:
        return "rejected", "no [source] citation", citations
    unretrieved = [c for c in citations if c not in retrieved_ids]
    if unretrieved:
        return "rejected", f"cited sources that were not retrieved: {unretrieved}", citations
    return "answered", "all citations were retrieved", citations


def answer_question(
    system: str, retrieval: RetrievalResult, generate: Callable[[str, str], str]
) -> Answer:
    """Generate from the retrieved evidence, then serve the answer only if it validates.

    Rejected and abstained results serve the fallback reply, never the raw output.
    """

    started = perf_counter()
    if retrieval.abstain:
        return Answer(
            system, retrieval.question, "abstained", FALLBACK_ANSWER, "", [],
            [], "no evidence above min_score", perf_counter() - started,
        )

    raw = generate(retrieval.question, evidence_text(retrieval.hits))
    status, reason, citations = validate_answer(raw, retrieval.source_ids)
    return Answer(
        system=system,
        question=retrieval.question,
        status=status,
        text=raw if status == "answered" else FALLBACK_ANSWER,
        raw=raw,
        citations=citations,
        retrieved_ids=retrieval.source_ids,
        reason=reason,
        latency=perf_counter() - started,
    )


if __name__ == "__main__":
    import argparse

    from src.generation import make_generator
    from src.retrieval import Retriever, build_chunks
    from src.training import TrainingConfig

    parser = argparse.ArgumentParser(description="Compare baseline RAG and hybrid RAG + LoRA")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    retriever = Retriever()
    retriever.index(build_chunks())
    retrieval = retriever.retrieve(args.question)
    print("Question: ", args.question)
    print("Retrieved:", retrieval.source_ids or "(none above min_score)")
    for hit in retrieval.hits:
        print(f"  {hit.score:.3f} [{hit.chunk.chunk_id}] {hit.chunk.text[:80]}...")

    systems = {
        "baseline-rag": None,
        "hybrid-rag-lora": TrainingConfig().adapter_path,
    }
    for name, adapter_path in systems.items():
        answer = answer_question(name, retrieval, make_generator(adapter_path))
        print(f"\n[{name}] {answer.status.upper()} ({answer.reason}) in {answer.latency:.2f}s")
        print("  raw:   ", answer.raw or "(not generated)")
        print("  served:", answer.text)
