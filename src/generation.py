"""Load the base model, optionally with the saved LoRA adapter, and generate replies.

The base model and the adapter are loaded separately: the adapter folder holds
only LoRA weights, which PEFT attaches to the frozen base model at load time.

Reload the adapter and answer a question with retrieved evidence:

    uv run python -m src.generation --adapter artifacts/support-lora-adapter \
        --question "A customer forgot their password. What should support do?"

Leave out ``--adapter`` to answer with the unchanged base model.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import inference_prompt
from src.training import SUMMARY_FILENAME, TrainingConfig


def load_model(adapter_path: Path | str | None = None):
    """Return (model, tokenizer); the base model always comes from the pinned revision.

    With an adapter, the base model and revision recorded at training time are
    used so the LoRA weights are attached to the weights they were trained on.
    """

    config = TrainingConfig()
    base_model, revision = config.base_model, config.base_model_revision
    if adapter_path is not None:
        summary = json.loads((Path(adapter_path) / SUMMARY_FILENAME).read_text(encoding="utf-8"))
        base_model, revision = summary["base_model"], summary["base_model_revision"]

    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, revision=revision, dtype=torch.float32)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, context: str, max_new_tokens: int = 80) -> str:
    """Greedy decoding so baseline and adapter outputs are deterministic and comparable."""

    prompt = inference_prompt(tokenizer, question, context)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--question", required=True)
    parser.add_argument("--adapter", help="saved LoRA adapter folder; omit for the base model")
    parser.add_argument("--context", help="evidence text; omit to retrieve it from the policies")
    args = parser.parse_args()

    context = args.context
    if context is None:
        from src.retrieval import Retriever, build_chunks

        retriever = Retriever()
        retriever.index(build_chunks())
        result = retriever.retrieve(args.question)
        context = "\n".join(f"[{hit.chunk.document_id}] {hit.chunk.text}" for hit in result.hits)
        print("Retrieved sources:", result.source_ids or "(none, below threshold)")

    model, tokenizer = load_model(args.adapter)
    print("Model:", "base + LoRA adapter " + args.adapter if args.adapter else "base model only")
    print("Answer:", generate_answer(model, tokenizer, args.question, context))
