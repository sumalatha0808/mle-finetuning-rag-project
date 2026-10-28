"""Base and adapter-backed local answer generation."""

import hashlib
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, cast

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

if TYPE_CHECKING:
    from transformers._typing import GenerativePreTrainedModel

from .config import Settings
from .models import SearchHit, TrainingSummary
from .prompts import FALLBACK_ANSWER, evidence_text, inference_prompt

CITATION_PATTERN = re.compile(r"\[([a-z0-9-]+)\]")


def citations_in(answer: str) -> list[str]:
    """Extract ordered unique citation IDs from generated text."""

    return list(dict.fromkeys(CITATION_PATTERN.findall(answer)))


def citations_are_valid(answer: str, hits: Sequence[SearchHit]) -> bool:
    """Require citations and reject source IDs absent from retrieved evidence."""

    citations = citations_in(answer)
    retrieved = {hit.document_id for hit in hits}
    return bool(citations) and set(citations).issubset(retrieved)


def validate_adapter(settings: Settings) -> None:
    """Reject stale or untraceable adapters before loading any model weights."""

    summary_path = settings.adapter_path / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "Adapter provenance is missing; run the training command."
        )
    try:
        summary = TrainingSummary.model_validate_json(summary_path.read_bytes())
    except ValueError as error:
        raise ValueError(
            "Adapter provenance is invalid; rerun the training command."
        ) from error
    if (summary.base_model, summary.base_model_revision) != (
        settings.base_model,
        settings.base_model_revision,
    ):
        raise ValueError(
            "The base model or revision changed; rerun the training command."
        )
    if (
        summary.training_sha256
        != hashlib.sha256(settings.training_file.read_bytes()).hexdigest()
    ):
        raise ValueError("Training data changed; rerun the training command.")


class Generator(Protocol):
    """Generation contract used by every benchmark path."""

    @property
    def mode(self) -> str:
        """Return a report-friendly generator label."""

    def generate(self, question: str, hits: Sequence[SearchHit]) -> str:
        """Generate an answer from a question and optional evidence."""


class LocalModelGenerator:
    """Generate with the unchanged base model or a loaded LoRA adapter."""

    def __init__(self, settings: Settings, use_adapter: bool) -> None:
        self.settings = settings
        self.use_adapter = use_adapter
        if use_adapter:
            validate_adapter(settings)
        tokenizer = AutoTokenizer.from_pretrained(
            settings.base_model, revision=settings.base_model_revision
        )
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError("The configured model did not return a tokenizer")
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            settings.base_model, revision=settings.base_model_revision
        )
        if use_adapter:
            adapter_model = PeftModel.from_pretrained(
                base_model, str(settings.adapter_path)
            )
            model: GenerativePreTrainedModel | PeftModel = adapter_model
        else:
            # AutoModelForCausalLM supplies the generative model contract.
            model = cast("GenerativePreTrainedModel", base_model)
        self.model = model
        # Models load on CPU by default; derive the actual placement so inputs match.
        self.device = next(self.model.parameters()).device
        self.model.eval()

    @property
    def mode(self) -> str:
        """Identify whether the adapter is active."""

        suffix = "LoRA adapter" if self.use_adapter else "unchanged base model"
        return f"{self.settings.base_model} with {suffix}"

    def generate(self, question: str, hits: Sequence[SearchHit]) -> str:
        """Run deterministic local generation from the shared prompt format."""

        prompt = inference_prompt(
            self.tokenizer,
            question,
            evidence_text(hits),
        )
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.shape[1] > self.settings.max_sequence_length:
            raise ValueError(
                "Prompt exceeds MAX_SEQUENCE_LENGTH; shorten the evidence/question "
                "or increase the limit. The question must not be truncated."
            )
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0, input_ids.shape[1] :]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        if not isinstance(answer, str):
            raise TypeError("Decoding one answer must return a string")
        return answer.strip() or FALLBACK_ANSWER


class ExtractiveSmokeGenerator:
    """Deterministic evidence answerer for tests, never final model claims."""

    def __init__(self, label: str) -> None:
        self.label = label

    @property
    def mode(self) -> str:
        """Label smoke output clearly in reports."""

        return f"deterministic smoke generator: {self.label}"

    def generate(self, question: str, hits: Sequence[SearchHit]) -> str:
        """Return the strongest visible evidence sentence with its citation."""

        del question
        if not hits:
            return FALLBACK_ANSWER
        hit = hits[0]
        sentence = hit.text.split(". ")[0].rstrip(".")
        return f"{sentence}. [{hit.document_id}]"
