"""Prompt serialization shared by LoRA training and inference.

Training and generation must render the exact same prompt, otherwise the
adapter learns a format it never sees at inference time. Both paths use the
base model's chat template; the answer is the assistant turn.
"""

from transformers import PreTrainedTokenizerBase

FALLBACK_ANSWER = "The approved material is insufficient. Escalate for manual review."
SYSTEM_INSTRUCTION = (
    "Answer only from approved context. Start with the direct answer, then "
    "give the next action. Cite every policy claim with source IDs in square "
    "brackets. If the context is insufficient, reply exactly: "
    f"{FALLBACK_ANSWER}"
)


def chat_messages(question: str, context: str) -> list[dict[str, str]]:
    """System instruction plus one user turn holding the evidence and question."""

    context = context or "No approved policy evidence was supplied."
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Approved context:\n{context}\n\nQuestion: {question}"},
    ]


def inference_prompt(tokenizer: PreTrainedTokenizerBase, question: str, context: str) -> str:
    """Prompt up to and including the assistant header; generation continues it."""

    return tokenizer.apply_chat_template(
        chat_messages(question, context), tokenize=False, add_generation_prompt=True
    )


def training_text(
    tokenizer: PreTrainedTokenizerBase, question: str, context: str, answer: str
) -> str:
    """Full supervised conversation: the inference prompt followed by the answer."""

    messages = chat_messages(question, context)
    messages.append({"role": "assistant", "content": answer})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
