# Reference Solution

This folder contains a complete implementation of the Northstar Fine-Tuning +
RAG project. The solution keeps data validation, retrieval, adapter
training, generation, and evaluation as separate responsibilities so each
decision can be inspected or replaced.

The reference project path uses:

- [Sentence Transformers](https://www.sbert.net/) for dense policy embeddings,
- [Qdrant local mode](https://qdrant.tech/documentation/quickstart/) for vector
  storage with source metadata,
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
  and [PyTorch](https://pytorch.org/docs/stable/index.html) for the local causal
  language model, using
  [SmolLM2 135M Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
  by default,
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index) for a real LoRA
  optimization run,
- deterministic metric rubrics that keep retrieval, relevance, faithfulness,
  response style, decisions, and latency separate.

## Read In Three Steps

These three reading steps cover the four [project milestones](../project-description.md#suggested-milestones). Read only the files for your current step; this is one reference design, not the module structure you must reproduce.

| Reading Step | Read first | Follow the flow |
| --- | --- | --- |
| 1. Data and retrieval | [datasets.py](hybrid_project/datasets.py), [documents.py](hybrid_project/documents.py), [retrieval.py](hybrid_project/retrieval.py) | Validate records, preserve source IDs, and build the shared dense index. |
| 2. Training and cited answers | [training.py](hybrid_project/training.py), [generation.py](hybrid_project/generation.py), [pipeline.py](hybrid_project/pipeline.py) | Train LoRA, compare base and adapted answers, and validate citations. |
| 3. Evaluation and decision | [evaluation.py](hybrid_project/evaluation.py), [reference report](reports/reference_evaluation.md) | Apply the same rubric to both systems, inspect failures, and justify the added complexity. |

Dependencies and inputs remain at the project root. The [simpler suggested layout](../project-description.md#suggested-repository-layout) is sufficient for your submission.

## Run The Reference Workflow

Follow the [main README setup](../README.md#setup), then run every command below from the project root using the shared `uv` environment.

### 1. Validate The Data

```bash
uv run python -m solution.hybrid_project inspect-data
```

This checks source IDs, target citations, and exact question duplicates. Some
evaluation scenarios still overlap with training: this is a within-domain check,
not a claim of generalization to unseen policies. Two near-domain unsupported
cases check that retrieval similarity is not confused with answerability.

### 2. Check The Shared Index

```bash
uv run python -m solution.hybrid_project index
```

The command downloads the configured embedding model on first use, creates
paragraph-aware chunks, and indexes them in Qdrant local mode.
Each command rebuilds this small index, including when persistent storage is
configured; the default in-memory index disappears on exit. This step verifies
indexing, rather than creating a cache for subsequent commands.

### 3. Train The LoRA Adapter

```bash
uv run python -m solution.hybrid_project train
```

The command performs a real optimization loop and saves adapter weights plus
`training_summary.json` under `artifacts/support-lora-adapter/`. The summary
records the base-model revision and the SHA-256 of the exact training bytes.
Generation rejects adapters with missing provenance, changed training data, or
a different base model/revision: rerun `train` in those cases. Existing legacy
adapters must also be retrained. Generated weights remain ignored by Git.
Overlength training examples and inference prompts raise an explicit error;
shorten the evidence or increase `MAX_SEQUENCE_LENGTH` rather than silently
losing the question or part of the answer.

### 4. Compare One Question

```bash
uv run python -m solution.hybrid_project ask \
  --system rag-base \
  "Can support change an address after the warehouse packed an order?"

uv run python -m solution.hybrid_project ask \
  --system rag-lora \
  "Can support change an address after the warehouse packed an order?"
```

Both systems receive the same retrieved chunks. The second command changes only
the model by loading the adapter.
Each `ask` command loads only the requested model; `rag-base` works before
training an adapter, and `base-only` also skips retrieval.

### 5. Generate The Evaluation Report

```bash
uv run python -m solution.hybrid_project evaluate
```

This writes
[reference_evaluation.md](reports/reference_evaluation.md) and a regenerated
JSON trace. The report separates shared retrieval metrics from each system's
answer metrics. It also preserves raw generated answers in the JSON trace and
compares their lexical coverage with citation rejections. Rejected text is for
inspection only, never a validated answer. The rubric cannot establish semantic
correctness; manually inspect failures and policy-change cases before concluding
that LoRA improves reasoning rather than citation format.

### 6. Run Tests

```bash
uv run pytest solution/tests
```

Tests use explicitly labeled deterministic embedding and generation
substitutes. They validate contracts and behavior without downloading models;
they are not evidence that LoRA improves answer quality.

## Smoke Mode

The following command checks orchestration quickly:

```bash
uv run python -m solution.hybrid_project --hash-smoke evaluate \
  --output solution/reports/smoke_evaluation.md
```

Smoke mode is for development only. The submitted decision report must come
from the real dense retriever and real base/adapter model paths.
Without `--output`, smoke evaluation writes `smoke_evaluation.md`, never the
reference report. `--hash-smoke train` is rejected because training is real.

## Implementation Decisions

### Why share the retriever?

Using one retriever, evidence threshold, prompt structure, and benchmark keeps
the comparison focused on the adapter. Different retrieval results would
confound the fine-tuning decision.

### Why mask prompt tokens during training?

The adapter loss is calculated on the target answer rather than rewarding the
model for reproducing context and question text. This keeps the small training
signal focused on response behavior.

### Why validate citations after generation?

Prompt instructions do not guarantee compliant output. A generated answer is
accepted only when it cites at least one retrieved source and invents no source
ID.

### Why include base-only results?

Base-only results expose whether retrieval itself adds value. The required
project decision remains baseline RAG versus hybrid RAG + LoRA.
