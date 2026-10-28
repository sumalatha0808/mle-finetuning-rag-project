# Project Description

Northstar operates a subscription platform and a digital workshop marketplace.
Its support team answers questions about refunds, account security, shipping,
subscriptions, and technical incidents. Policy changes are stored in approved
Markdown sources, while historical support examples show the concise,
action-oriented reply style the team expects.

Build a hybrid assistant that combines both forms of knowledge:

- retrieval supplies current, citable policy evidence,
- a locally trained LoRA adapter shapes how the model follows context and
  writes support replies.

The final benchmark must determine whether the hybrid system improves on a
baseline RAG pipeline enough to justify the additional training and deployment
complexity.

## Main Task

Implement and evaluate two comparable systems:

1. **Baseline RAG:** retrieve policy evidence and prompt the unchanged base
   model.
2. **Hybrid RAG + LoRA:** use the same retriever and prompt structure, but
   generate with the locally trained adapter.

You may also include a base-model-only result to expose the value of retrieval,
but the required decision is the baseline RAG versus hybrid comparison.

## Supplied Data

### Policy Sources

[data/knowledge_base/](data/knowledge_base/) contains the approved information
that the answer path may cite. Treat these sources as current truth. Preserve
their filenames or stable IDs through chunking, vector storage, retrieval, and
answer validation.

### Supervised Training Examples

[data/training/instruction_examples.jsonl](data/training/instruction_examples.jsonl)
contains small, auditable examples of context-following support replies. Use
them to train a LoRA adapter; do not move held-out evaluation cases into this
file to improve results.

### Held-Out Evaluation Cases

[data/evaluation/cases.json](data/evaluation/cases.json) labels expected
sources, required answer terms, forbidden claims, and unsupported questions.
Every compared system must run on the same cases with the same metric rubric.

## Required Implementation

### 1. Data Validation

Validate the three data layers before training:

- every training example has an ID, context, question, answer, and source IDs,
- every cited source ID exists in the knowledge base,
- evaluation IDs are unique,
- evaluation cases remain separate from training examples.

Document any data-quality decision that changes the supplied examples.

### 2. Retrieval

Load and chunk the Markdown sources, create embeddings, and store chunks in a
real vector index or vector database. Retain at least:

- document ID,
- title,
- chunk position,
- source text.

Define a top-k setting and evidence threshold. Unsupported questions must
abstain rather than use the nearest weak match.

### 3. Real Local LoRA Training

Train an adapter with
[Hugging Face PEFT](https://huggingface.co/docs/peft/index) or an equivalent
parameter-efficient fine-tuning library. Record:

- base model name and revision when available,
- prompt serialization format,
- LoRA target modules, rank, alpha, and dropout,
- sequence length, epochs, learning rate, and random seed,
- trainable versus total parameter counts,
- adapter output path.

The project must run a real optimization loop. Few-shot prompting alone does
not satisfy this requirement.

### 4. Comparable Answer Paths

Use the same retrieved evidence and question format for baseline RAG and the
hybrid system. The only intended model difference should be the loaded adapter.

Both paths must:

- answer only from retrieved evidence,
- cite source IDs in square brackets,
- reject citations that were not retrieved,
- apply the same no-answer threshold,
- expose whether the result was answered, abstained, or rejected.

### 5. Evaluation

Evaluate retrieval once and answer quality separately for each system.
At minimum, report:

- retrieval recall@k and mean reciprocal rank,
- required-term coverage as a transparent relevance proxy,
- citation-based faithfulness,
- response-style compliance,
- supported-versus-unsupported decision accuracy,
- latency per answer.

Keep deterministic rubrics explicit. If you add an LLM-based judge, report it
as a separate metric and retain a no-key evaluation path.

### 6. Decision Report

Submit `reports/evaluation_report.md` containing:

- complete configuration for retrieval, base model, and adapter,
- aggregate comparison table,
- per-case evidence for both systems,
- at least two failure or fragile-success analyses,
- the evidence behind the retrieval threshold,
- training and inference cost considerations,
- a recommendation to keep, revise, or reject the hybrid design.

Do not conclude that fine-tuning is useful merely because training loss
decreased. The decision must use held-out behavior.

## Suggested Repository Layout

```text
your-hybrid-project/
├── data/
│   ├── knowledge_base/
│   ├── training/
│   └── evaluation/
├── reports/
│   └── evaluation_report.md
├── src/
│   ├── retrieval.py
│   ├── training.py
│   ├── generation.py
│   └── evaluation.py
├── tests/
├── .env.example
├── README.md
├── pyproject.toml
└── uv.lock
```

## Suggested Milestones

### Milestone 1 - Auditable Data And Retrieval

Evidence to show:

- validated training and evaluation records,
- sample chunks with source metadata,
- retrieval results for supported and unsupported questions,
- a documented threshold hypothesis.

### Milestone 2 - Adapter Training

Evidence to show:

- trainable parameter summary,
- loss history,
- adapter files saved separately from the base model,
- a command that reloads the adapter for inference.

### Milestone 3 - Comparable Systems

Evidence to show:

- one baseline RAG response,
- one hybrid response for the same question and evidence,
- valid citations or a visible rejection outcome,
- a supported abstention example.

### Milestone 4 - Benchmark And Recommendation

Evidence to show:

- shared held-out benchmark,
- aggregate and per-case comparison,
- failure analysis,
- a recommendation supported by measured results.

## Minimum Acceptance Checklist

- [ ] Policy, training, and evaluation artifacts remain separate.
- [ ] Source metadata survives chunking and vector storage.
- [ ] Retrieval uses a real vector index or vector database.
- [ ] A real local LoRA optimization loop runs and saves an adapter.
- [ ] Baseline RAG and hybrid RAG use the same retriever and cases.
- [ ] Unsupported questions abstain.
- [ ] Generated citations are validated against retrieved evidence.
- [ ] Retrieval and answer metrics are reported separately.
- [ ] The report analyzes failures and adapter trade-offs.
- [ ] Tests pass from the documented environment.

## Answer the following questions

1. Which held-out behaviors changed after loading the adapter?
2. Did the adapter improve evidence use, style, both, or neither?
3. Which retrieval errors limit both systems regardless of fine-tuning?
4. Does the measured improvement justify adapter training and maintenance?
5. What experiment would most reduce uncertainty in that decision?
