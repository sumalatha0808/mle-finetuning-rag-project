"""Real laptop-friendly LoRA training for the project reference solution."""

import hashlib
import json
from time import perf_counter

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    set_seed,
)

from .config import Settings
from .datasets import load_training_examples
from .models import TrainingExample, TrainingSummary
from .prompts import inference_prompt, training_text


def _device() -> torch.device:
    """Prefer Apple Metal when available and otherwise use the CPU."""

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters after attaching LoRA."""

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def _encoded_example(
    tokenizer: PreTrainedTokenizerBase,
    example: TrainingExample,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize one record and mask prompt tokens from the training loss."""

    full_text = training_text(tokenizer, example)
    prompt = inference_prompt(tokenizer, example.question, example.context)
    encoded: BatchEncoding = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_tokens: BatchEncoding = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    prompt_input_ids = prompt_tokens.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("Tokenizer did not return tensor input_ids")
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("Tokenizer did not return a tensor attention_mask")
    if not isinstance(prompt_input_ids, torch.Tensor):
        raise TypeError("Tokenizer did not return prompt input_ids")
    if input_ids.shape[1] > max_length:
        raise ValueError(
            f"Training example {example.id} exceeds MAX_SEQUENCE_LENGTH; "
            "increase the limit or shorten the example rather than truncate its answer."
        )
    if not torch.equal(input_ids[:, : prompt_input_ids.shape[1]], prompt_input_ids):
        raise ValueError(
            "Training text must start with the exact inference prompt tokens"
        )
    labels = input_ids.clone()
    prompt_length = min(prompt_input_ids.shape[1], labels.shape[1])
    labels[:, :prompt_length] = -100
    if bool(torch.all(labels == -100)):
        raise ValueError("The training example has no supervised answer tokens.")
    return input_ids, attention_mask, labels


def _lora_targets(model: torch.nn.Module) -> tuple[list[str], bool]:
    """Select adapter projections from the loaded model architecture."""

    module_names = {
        name.rsplit(".", maxsplit=1)[-1] for name, _module in model.named_modules()
    }
    if {"q_proj", "v_proj"}.issubset(module_names):
        return ["q_proj", "v_proj"], False
    if "c_attn" in module_names:
        return ["c_attn"], True
    raise ValueError(
        "No supported LoRA targets found. Expected q_proj/v_proj or c_attn."
    )


def train_adapter(settings: Settings) -> TrainingSummary:
    """Train LoRA weights and save a reproducible training summary."""

    set_seed(settings.random_seed)
    # Fingerprint the same bytes used for training, even if the file is edited later.
    training_bytes = settings.training_file.read_bytes()
    examples = load_training_examples(settings.training_file, content=training_bytes)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.base_model, revision=settings.base_model_revision
    )
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("The configured model did not return a tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Validate and tokenize this small dataset once, before loading model weights.
    encoded_examples = [
        _encoded_example(tokenizer, example, settings.max_sequence_length)
        for example in examples
    ]
    base_model = AutoModelForCausalLM.from_pretrained(
        settings.base_model, revision=settings.base_model_revision, dtype=torch.float32
    )
    # Some tiny checkpoint configs omit this value even though the model is causal.
    base_model.loss_type = "ForCausalLM"
    base_model.config.use_cache = False
    target_modules, fan_in_fan_out = _lora_targets(base_model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        fan_in_fan_out=fan_in_fan_out,
    )
    model = get_peft_model(base_model, lora_config)
    device = _device()
    model.to(str(device))
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=settings.learning_rate,
    )

    epoch_losses: list[float] = []
    started = perf_counter()
    model.train()
    for _epoch in range(settings.train_epochs):
        running_loss = 0.0
        for input_ids, attention_mask, labels in encoded_examples:
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("The causal language model did not return a loss")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += float(loss.detach().cpu())
        epoch_losses.append(running_loss / len(examples))

    settings.adapter_path.mkdir(parents=True, exist_ok=True)
    # Only attention projections were trained; vocabulary and embeddings are unchanged.
    model.save_pretrained(str(settings.adapter_path), save_embedding_layers=False)
    tokenizer.save_pretrained(str(settings.adapter_path))
    trainable, total = _parameter_counts(model)
    summary = TrainingSummary(
        base_model=settings.base_model,
        base_model_revision=settings.base_model_revision,
        training_sha256=hashlib.sha256(training_bytes).hexdigest(),
        adapter_path=settings.adapter_path,
        epochs=settings.train_epochs,
        learning_rate=settings.learning_rate,
        max_sequence_length=settings.max_sequence_length,
        lora_target_modules=target_modules,
        lora_rank=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        random_seed=settings.random_seed,
        trainable_parameters=trainable,
        total_parameters=total,
        trainable_percentage=100 * trainable / total,
        epoch_losses=epoch_losses,
        training_seconds=perf_counter() - started,
        training_device=str(device),
    )
    summary_path = settings.adapter_path / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return summary
