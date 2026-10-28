"""Token-length and loss-mask regressions using a real, local tokenizer."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from solution.hybrid_project import training
from solution.hybrid_project.config import Settings
from solution.hybrid_project.generation import LocalModelGenerator
from solution.hybrid_project.models import TrainingExample
from solution.hybrid_project.prompts import inference_prompt
from solution.hybrid_project.training import _encoded_example


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    raw = Tokenizer(
        WordLevel({"[UNK]": 0, "question": 1, "answer": 2}, unk_token="[UNK]")
    )
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")


@pytest.fixture
def example() -> TrainingExample:
    return TrainingExample(
        id="test",
        reference_ids=[],
        context="context",
        question="question",
        answer="answer",
    )


def test_only_answer_tokens_are_supervised(tokenizer, example) -> None:
    inputs, mask, labels = _encoded_example(tokenizer, example, 256)
    prompt = inference_prompt(tokenizer, example.question, example.context)
    prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    assert torch.all(labels[:, :prompt_length] == -100)
    assert torch.equal(labels[:, prompt_length:], inputs[:, prompt_length:])
    assert (labels != -100).sum() == 1
    assert mask.shape == inputs.shape


def test_partial_answer_truncation_is_rejected(tokenizer, example) -> None:
    example.answer = "answer " * 20
    inputs, _, _ = _encoded_example(tokenizer, example, 256)
    with pytest.raises(ValueError, match="exceeds MAX_SEQUENCE_LENGTH"):
        _encoded_example(tokenizer, example, inputs.shape[1] - 1)


def test_no_answer_tokens_is_rejected(tokenizer, example) -> None:
    example.answer = ""
    with pytest.raises(ValueError, match="no supervised answer tokens"):
        _encoded_example(tokenizer, example, 256)


def test_overlength_question_never_reaches_generation(tokenizer, monkeypatch) -> None:
    class Model:
        def generate(self, **kwargs):
            pytest.fail("An overlength prompt must fail before generation")

    generator = object.__new__(LocalModelGenerator)
    generator.settings = replace(Settings.from_env(), max_sequence_length=16)
    generator.tokenizer = tokenizer
    generator.device = torch.device("cpu")
    monkeypatch.setattr(generator, "model", Model(), raising=False)
    with pytest.raises(ValueError, match="question must not be truncated"):
        generator.generate("question " * 100, [])


def test_training_tokenizes_once_before_model_loading(
    tokenizer, example, tmp_path, monkeypatch
):
    settings = replace(
        Settings.from_env(),
        training_file=tmp_path / "train.jsonl",
        adapter_path=tmp_path / "adapter",
        train_epochs=3,
    )
    settings.training_file.write_text(
        example.model_dump_json() + "\n" + example.model_dump_json()
    )
    encode = Mock(wraps=_encoded_example)
    monkeypatch.setattr(training, "_encoded_example", encode)
    monkeypatch.setattr(
        training.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer
    )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(1, 1)
            self.v_proj = torch.nn.Linear(1, 1)
            self.config = SimpleNamespace(use_cache=True)
            self.forward = lambda *args, **kwargs: SimpleNamespace(
                loss=sum(p.square().sum() for p in self.parameters())
            )

        def save_pretrained(self, *args, **kwargs):
            pass

    def load(*args, **kwargs):
        assert encode.call_count == 2
        return TinyModel()

    monkeypatch.setattr(training.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(training, "get_peft_model", lambda model, config: model)
    monkeypatch.setattr(training, "_device", lambda: torch.device("cpu"))
    summary = training.train_adapter(settings)
    assert encode.call_count == 2
    assert len(summary.epoch_losses) == 3


def test_invalid_training_example_fails_before_model_loading(
    tokenizer, example, tmp_path, monkeypatch
):
    settings = replace(
        Settings.from_env(),
        training_file=tmp_path / "train.jsonl",
        max_sequence_length=1,
    )
    settings.training_file.write_text(example.model_dump_json())
    monkeypatch.setattr(
        training.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer
    )
    loader = Mock()
    monkeypatch.setattr(training.AutoModelForCausalLM, "from_pretrained", loader)
    with pytest.raises(ValueError, match="exceeds MAX_SEQUENCE_LENGTH"):
        training.train_adapter(settings)
    loader.assert_not_called()
