"""Prompt serialization shared by training and inference."""

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from .models import SearchHit, TrainingExample

ANSWER_MARKER = "### Support answer\n"
FALLBACK_ANSWER = "The approved material is insufficient. Escalate for manual review."
SYSTEM_INSTRUCTION = (
    "Answer only from approved context. Start with the direct answer, then "
    "give the next action. Cite every policy claim with source IDs in square "
    "brackets. If the context is insufficient, reply exactly: "
    f"{FALLBACK_ANSWER}"
)


def evidence_text(hits: Sequence[SearchHit]) -> str:
    """Render retrieved chunks with visible source IDs."""

    return "\n".join(f"[{hit.document_id}] {hit.text}" for hit in hits)


def build_prompt(question: str, evidence: str) -> str:
    """Build a plain-text fallback for tokenizers without chat templates."""

    context = evidence or "No approved policy evidence was supplied."
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"### Approved context\n{context}\n\n"
        f"### Customer question\n{question}\n\n"
        f"{ANSWER_MARKER}"
    )


def _chat_messages(
    question: str,
    evidence: str,
) -> list[dict[str, str]]:
    """Build role-separated messages for instruction-tuned tokenizers."""

    context = evidence or "No approved policy evidence was supplied."
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": f"Approved context:\n{context}\n\nQuestion: {question}",
        },
    ]


def inference_prompt(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    evidence: str,
) -> str:
    """Render the native chat format when the tokenizer defines one."""

    if tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            _chat_messages(question, evidence),
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise TypeError("Chat template did not return a prompt string")
        return rendered
    return build_prompt(question, evidence)


def training_text(
    tokenizer: PreTrainedTokenizerBase,
    example: TrainingExample,
) -> str:
    """Render a supervised conversation or the plain-text fallback."""

    if tokenizer.chat_template:
        messages = _chat_messages(example.question, example.context)
        messages.append({"role": "assistant", "content": example.answer})
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(rendered, str):
            raise TypeError("Chat template did not return training text")
        return rendered
    end_token = tokenizer.eos_token or ""
    return (
        f"{build_prompt(example.question, example.context)}{example.answer}{end_token}"
    )
