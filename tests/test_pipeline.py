"""Citation validation, abstention, and rubric tests; no model is loaded."""

from src.evaluation import contains_term, score_case, style_checks, term_coverage
from src.pipeline import answer_question, validate_answer
from src.prompts import FALLBACK_ANSWER
from src.retrieval import Chunk, RetrievalResult, SearchHit

SUPPORTED_CASE = {
    "id": "eval-x",
    "question": "Can support change the address after packing?",
    "expected_sources": ["shipping"],
    "required_terms": ["cannot", "carrier"],
    "forbidden_terms": ["support can change the address after packing"],
    "should_abstain": False,
}
UNSUPPORTED_CASE = {**SUPPORTED_CASE, "expected_sources": [], "required_terms": [],
                    "forbidden_terms": [], "should_abstain": True}


def retrieval(*document_ids: str) -> RetrievalResult:
    hits = [
        SearchHit(Chunk(f"{doc}#0", doc, doc.title(), 0, f"{doc} policy text"), 0.6)
        for doc in document_ids
    ]
    return RetrievalResult(SUPPORTED_CASE["question"], hits, abstain=not hits)


def test_valid_citation_is_answered():
    status, _, citations = validate_answer("Support cannot change it. [shipping]", ["shipping"])
    assert (status, citations) == ("answered", ["shipping"])


def test_unretrieved_citation_is_rejected():
    status, reason, _ = validate_answer("Refund it. [refunds]", ["shipping"])
    assert status == "rejected" and "refunds" in reason


def test_missing_citation_is_rejected():
    assert validate_answer("Support cannot change it.", ["shipping"])[0] == "rejected"


def test_fallback_reply_is_abstention():
    assert validate_answer(FALLBACK_ANSWER, ["shipping"])[0] == "abstained"


def test_below_threshold_abstains_without_generating():
    def generate(question, context):
        raise AssertionError("must not generate without evidence")

    answer = answer_question("baseline-rag", retrieval(), generate)
    assert answer.status == "abstained" and answer.text == FALLBACK_ANSWER


def test_rejected_answer_serves_fallback_and_keeps_raw():
    answer = answer_question("hybrid", retrieval("shipping"), lambda q, c: "Yes. [refunds]")
    assert answer.status == "rejected"
    assert answer.text == FALLBACK_ANSWER and answer.raw == "Yes. [refunds]"


def test_both_systems_receive_identical_evidence():
    seen = []
    shared = retrieval("shipping", "response-style")
    for name in ("baseline-rag", "hybrid-rag-lora"):
        answer_question(name, shared, lambda q, c: seen.append((q, c)) or "Ok. [shipping]")
    assert seen[0] == seen[1]
    assert "[shipping] shipping policy text" in seen[0][1]


def test_term_matching_uses_whole_phrases():
    assert contains_term("Support cannot do that.", "cannot")
    assert not contains_term("See the note.", "not")
    assert term_coverage("It cannot change; contact the carrier.", ["cannot", "carrier"]) == 1.0


def test_style_checks_flag_long_hedged_replies():
    assert all(style_checks("Support cannot change it. Contact the carrier. [shipping]").values())
    checks = style_checks("Thank you for asking, I think maybe " + "word " * 70)
    assert not checks["concise"] and not checks["no_hedging"] and not checks["no_filler_opening"]


def test_scores_supported_and_unsupported_cases():
    good = answer_question(
        "s", retrieval("shipping"), lambda q, c: "Support cannot change it; contact the carrier. [shipping]"
    )
    score = score_case(SUPPORTED_CASE, good)
    assert score["relevance"] == 1.0 and score["faithful"] and score["decision_correct"]

    answered_unsupported = score_case(UNSUPPORTED_CASE, good)
    assert not answered_unsupported["decision_correct"] and not answered_unsupported["faithful"]

    abstained = answer_question("s", retrieval(), lambda q, c: "")
    assert score_case(UNSUPPORTED_CASE, abstained)["decision_correct"]
