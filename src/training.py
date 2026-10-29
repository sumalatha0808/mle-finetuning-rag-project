"""Train a LoRA adapter on the Northstar support examples with Hugging Face PEFT.

A real optimization loop runs over the supervised examples. Prompt tokens are
masked from the loss so the adapter only learns to write the answer. Only the
adapter weights are saved; the base model stays unchanged on the Hub cache.

Run from the project root:

    uv run python -m src.training
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import torch
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from src.datasets import TRAINING_PATH, load_training, validate
from src.prompts import inference_prompt, training_text

load_dotenv()

DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_BASE_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SUMMARY_FILENAME = "training_summary.json"


@dataclass
class TrainingConfig:
    """Every setting that affects the adapter; recorded in the training summary."""

    base_model: str = field(default_factory=lambda: os.getenv("BASE_MODEL", DEFAULT_BASE_MODEL))
    base_model_revision: str | None = None
    adapter_path: Path = field(
        default_factory=lambda: Path(os.getenv("ADAPTER_PATH", "artifacts/support-lora-adapter"))
    )
    max_sequence_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_SEQUENCE_LENGTH", "256"))
    )
    epochs: int = field(default_factory=lambda: int(os.getenv("TRAIN_EPOCHS", "8")))
    learning_rate: float = field(
        default_factory=lambda: float(os.getenv("LEARNING_RATE", "0.0002"))
    )
    seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def __post_init__(self) -> None:
        # Pin the default model's revision so reruns use the same weights.
        if self.base_model_revision is None:
            self.base_model_revision = os.getenv("BASE_MODEL_REVISION") or (
                DEFAULT_BASE_REVISION if self.base_model == DEFAULT_BASE_MODEL else None
            )
        self.adapter_path = Path(self.adapter_path)


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    """Trainable and total parameters (total includes the frozen base model)."""

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def encode_example(tokenizer, example: dict, max_length: int) -> dict[str, torch.Tensor]:
    """Tokenize one example and set prompt labels to -100 so only the answer is learned.

    Overlong examples raise instead of being truncated, which would silently cut
    the answer the model is supposed to learn.
    """

    prompt = inference_prompt(tokenizer, example["question"], example["context"])
    full = training_text(tokenizer, example["question"], example["context"], example["answer"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    encoded = tokenizer(full, add_special_tokens=False, return_tensors="pt")

    if encoded.input_ids.shape[1] > max_length:
        raise ValueError(
            f"{example['id']} has {encoded.input_ids.shape[1]} tokens; "
            f"raise MAX_SEQUENCE_LENGTH above {max_length} or shorten the example"
        )
    if not torch.equal(encoded.input_ids[:, : prompt_ids.shape[1]], prompt_ids):
        raise ValueError(f"{example['id']}: training text does not start with the inference prompt")

    labels = encoded.input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return {
        "input_ids": encoded.input_ids,
        "attention_mask": encoded.attention_mask,
        "labels": labels,
    }


def train_adapter(config: TrainingConfig | None = None, training_path: Path = TRAINING_PATH) -> dict:
    """Train, save the adapter, and return the summary written next to it."""

    config = config or TrainingConfig()
    set_seed(config.seed)

    # Refuse to train on data that fails validation (bad citations, leaked eval cases).
    training_bytes = training_path.read_bytes()
    examples = load_training(training_path)
    report = validate(training=examples)
    if not report.ok:
        raise ValueError("Data validation failed:\n" + "\n".join(report.errors))

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.base_model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_example(tokenizer, e, config.max_sequence_length) for e in examples]

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model, revision=config.base_model_revision, dtype=torch.float32
    )
    base_model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.target_modules,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )
    model = get_peft_model(base_model, lora_config)
    trainable, total = parameter_counts(model)
    print(
        f"trainable params: {trainable:,} || all params: {total:,} "
        f"|| trainable%: {100 * trainable / total:.4f}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config.learning_rate
    )

    # Batch size 1: examples have different lengths and the dataset is tiny.
    step_losses: list[float] = []
    epoch_losses: list[float] = []
    started = perf_counter()
    model.train()
    for epoch in range(1, config.epochs + 1):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        order = torch.randperm(len(encoded), generator=generator).tolist()
        total_loss = 0.0
        for index in order:
            batch = {key: value.to(device) for key, value in encoded[index].items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step_losses.append(float(loss.detach().cpu()))
            total_loss += step_losses[-1]
        epoch_losses.append(total_loss / len(encoded))
        print(f"epoch {epoch}/{config.epochs}  mean loss {epoch_losses[-1]:.4f}")
    training_seconds = perf_counter() - started

    # save_pretrained on a PeftModel writes only adapter_config.json and the
    # LoRA weights, not the base model. Embeddings were not trained, so skip them.
    config.adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.adapter_path), save_embedding_layers=False)

    summary = {
        **asdict(config),
        "adapter_path": config.adapter_path.as_posix(),
        "prompt_format": "base model chat template: system instruction, user "
        "'Approved context:\\n{context}\\n\\nQuestion: {question}', assistant answer; "
        "prompt tokens masked from the loss",
        "training_examples": len(examples),
        "training_sha256": hashlib.sha256(training_bytes).hexdigest(),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percentage": 100 * trainable / total,
        "epoch_losses": epoch_losses,
        "step_losses": step_losses,
        "training_seconds": round(training_seconds, 1),
        "device": str(device),
    }
    (config.adapter_path / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    result = train_adapter()
    print(f"\nAdapter saved to {result['adapter_path']} in {result['training_seconds']}s")
    for path in sorted(Path(result["adapter_path"]).iterdir()):
        print(f"  {path.name:<28} {path.stat().st_size / 1024:>8.1f} KB")
