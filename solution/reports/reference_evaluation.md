# Reference Hybrid Evaluation

## Configuration

- Base model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Base revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- Adapter path: `artifacts/support-lora-adapter`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Vector database: `Qdrant` local mode with cosine similarity
- Retrieval: `top_k=3`, `min_score=0.30`
- Baseline generator: `HuggingFaceTB/SmolLM2-135M-Instruct with unchanged base model`
- Hybrid generator: `HuggingFaceTB/SmolLM2-135M-Instruct with LoRA adapter`
- Prompt: shared system instruction, `[source-id] text` context, question, assistant answer; native chat template when available, plain-text fallback otherwise
- Inference prompt limit: `256` tokens; generation limit: `80` new tokens; greedy decoding
- Overlength prompts and training examples fail explicitly; they are not silently truncated
- Training data SHA-256: `a842ca7e98e2f0e37877f0c6d8d60b62479b217e1e0338ef0187b9fd6e65de07`
- LoRA targets: `q_proj, v_proj`
- LoRA rank / alpha / dropout: `8` / `16` / `0.05`
- Training: `8` epochs, learning rate `0.0002`, seed `42`
- Training sequence limit: `256` tokens (prompt plus answer)
- Optimizer: `AdamW`, batch size `1`, default betas `(0.9, 0.999)`, weight decay `0.01`
- Training device: `mps`
- Training and save duration: `14.4` seconds
- Trainable parameters: `460800` of `134975808` (`0.341%`)
- Training loss: `1.547` to `0.506`

## Metric Rubric

- `recall@k`: supported cases whose expected source appears in the shared retrieval results.
- `MRR`: reciprocal rank of the first expected source for supported cases.
- `answer relevance`: whole-term coverage in accepted answers; blocked answers score 0 and unsupported cases score 1 only on abstention.
- `faithfulness`: a citation/forbidden-term proxy, not a semantic judge; unsupported cases must abstain. Base-only has no evidence to cite, so this citation-based score is not directly comparable to RAG.
- `style compliance`: concise first sentence, concise total response, and no vague hedge; rejected answers score 0, retrieval abstentions score 1.
- `decision accuracy`: accept answers on supported cases and abstain on unsupported cases, not factual correctness. An always-answer system benefits from a majority of supported cases.
- Lexical labels miss synonyms and some negations. Inspect raw answers before drawing quality conclusions.
- Training and evaluation share policies and some scenarios. This is a small within-domain check, not evidence of generalization to new policies.

## Shared Retrieval Results

| Metric | Score |
| --- | ---: |
| Retrieval recall@3 | 1.000 |
| Mean reciprocal rank | 1.000 |

## System Comparison

| System | Relevance | Faithfulness proxy | Style | Decision accuracy | Mean latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| base-only | 0.000 | 0.000 | 0.976 | 0.714 | 0.708 |
| rag-base | 0.143 | 0.143 | 0.143 | 0.143 | 0.697 |
| rag-lora | 0.393 | 0.643 | 0.786 | 0.643 | 0.640 |

### Content Versus Citation Format

Raw coverage is measured before citation rejection on supported cases (no generation scores 0). It is a lexical diagnostic, not permission to serve a rejected answer. Raw answers remain in the JSON trace.

| System | Raw coverage (supported) | Citation rejections |
| --- | ---: | ---: |
| base-only | 0.000 | 0/14 |
| rag-base | 0.500 | 12/14 |
| rag-lora | 0.600 | 3/14 |

## Hybrid Delta Against Baseline RAG

- Answer relevance delta: `+0.250`
- Faithfulness delta: `+0.500`
- Style-compliance delta: `+0.643`

## Per-Case Comparison

| ID | System | Status | Relevance | Citation/term check | Style | Citations |
| --- | --- | --- | ---: | --- | ---: | --- |
| eval-refund-ticket | base-only | answered | 0.00 | no | 1.00 | - |
| eval-duplicate-payment | base-only | answered | 0.00 | no | 1.00 | - |
| eval-authenticator | base-only | answered | 0.00 | no | 1.00 | - |
| eval-passport | base-only | answered | 0.00 | no | 1.00 | - |
| eval-packed-address | base-only | answered | 0.00 | no | 1.00 | - |
| eval-standard-shipping | base-only | answered | 0.00 | no | 1.00 | - |
| eval-annual-refund | base-only | answered | 0.00 | no | 1.00 | - |
| eval-pause | base-only | answered | 0.00 | no | 1.00 | - |
| eval-outage-updates | base-only | answered | 0.00 | no | 1.00 | - |
| eval-log-secrets | base-only | answered | 0.00 | no | 0.67 | - |
| eval-bank-chargeback | base-only | answered | 0.00 | no | 1.00 | - |
| eval-shipping-compensation | base-only | answered | 0.00 | no | 1.00 | - |
| eval-cafeteria | base-only | answered | 0.00 | no | 1.00 | - |
| eval-bicycle | base-only | answered | 0.00 | no | 1.00 | - |
| eval-refund-ticket | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-duplicate-payment | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-authenticator | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-passport | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-packed-address | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-standard-shipping | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-annual-refund | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-pause | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-outage-updates | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-log-secrets | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-bank-chargeback | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-shipping-compensation | rag-base | invalid-citations | 0.00 | no | 0.00 | - |
| eval-cafeteria | rag-base | abstained | 1.00 | yes | 1.00 | - |
| eval-bicycle | rag-base | abstained | 1.00 | yes | 1.00 | - |
| eval-refund-ticket | rag-lora | invalid-citations | 0.00 | no | 0.00 | - |
| eval-duplicate-payment | rag-lora | answered | 1.00 | yes | 1.00 | refunds |
| eval-authenticator | rag-lora | answered | 1.00 | yes | 1.00 | account-security |
| eval-passport | rag-lora | answered | 0.00 | yes | 1.00 | account-security |
| eval-packed-address | rag-lora | answered | 0.00 | yes | 1.00 | shipping |
| eval-standard-shipping | rag-lora | answered | 1.00 | yes | 1.00 | shipping |
| eval-annual-refund | rag-lora | answered | 0.50 | yes | 1.00 | subscriptions |
| eval-pause | rag-lora | answered | 0.00 | yes | 1.00 | subscriptions |
| eval-outage-updates | rag-lora | invalid-citations | 0.00 | no | 0.00 | - |
| eval-log-secrets | rag-lora | invalid-citations | 0.00 | no | 0.00 | - |
| eval-bank-chargeback | rag-lora | answered | 0.00 | no | 1.00 | refunds |
| eval-shipping-compensation | rag-lora | answered | 0.00 | no | 1.00 | shipping |
| eval-cafeteria | rag-lora | abstained | 1.00 | yes | 1.00 | - |
| eval-bicycle | rag-lora | abstained | 1.00 | yes | 1.00 | - |

## Failure And Fragility Analysis

- `base-only` on `eval-bank-chargeback`: status `answered`, relevance `0.00`, citation/term check `no`. The system answered a question not covered by the policies; similarity alone does not establish answerability.
- `base-only` on `eval-shipping-compensation`: status `answered`, relevance `0.00`, citation/term check `no`. The system answered a question not covered by the policies; similarity alone does not establish answerability.
- `base-only` on `eval-cafeteria`: status `answered`, relevance `0.00`, citation/term check `no`. The system answered a question not covered by the policies; similarity alone does not establish answerability.
- `rag-base` on `eval-bank-chargeback`: status `invalid-citations`, relevance `0.00`, citation/term check `no`. Citation formatting or source IDs failed validation; inspect the raw trace before attributing this to content quality.
- `rag-base` on `eval-shipping-compensation`: status `invalid-citations`, relevance `0.00`, citation/term check `no`. Citation formatting or source IDs failed validation; inspect the raw trace before attributing this to content quality.
- `rag-base` on `eval-refund-ticket`: status `invalid-citations`, relevance `0.00`, citation/term check `no`. Citation formatting or source IDs failed validation; inspect the raw trace before attributing this to content quality.
- `rag-lora` on `eval-bank-chargeback`: status `answered`, relevance `0.00`, citation/term check `no`. The system answered a question not covered by the policies; similarity alone does not establish answerability.
- `rag-lora` on `eval-shipping-compensation`: status `answered`, relevance `0.00`, citation/term check `no`. The system answered a question not covered by the policies; similarity alone does not establish answerability.
- `rag-lora` on `eval-refund-ticket`: status `invalid-citations`, relevance `0.00`, citation/term check `no`. Citation formatting or source IDs failed validation; inspect the raw trace before attributing this to content quality.
- Shared retrieval recall is `1.000`; failures after successful retrieval are generation, citation, or answer-coverage failures.
- The adapter changes faithfulness by `+0.500` and decision accuracy from `0.143` to `0.643`. Remaining invalid citations still require rejection rather than automatic repair.

## Threshold Decision

The shared threshold is `0.30`. The weakest supported top score is `0.525` and the strongest unsupported top score is `0.662`. The same threshold is applied to baseline RAG and hybrid RAG so adapter quality cannot hide retrieval uncertainty.
These are benchmark diagnostics, not a separate calibration set. Do not tune the threshold on these cases and call the resulting score held-out performance.
Supported and unsupported scores overlap: no single similarity threshold separates these cases. Inspect answerability, not just nearest-neighbor similarity.

## Cost And Limitations

Training adds an optimization run and an adapter to version, validate, and retrain when data changes. The frozen base model still occupies memory and participates in forward/backward computation; a small trainable percentage is not the same percentage of total memory or compute.
Inference latencies above exclude model loading and index construction. Both model instances are loaded together for this comparison, so it is not a deployment memory estimate. Peak memory and energy were not measured; local execution does not mean zero compute cost.
Citation-format gains can dominate the accepted-answer metrics. Compare raw coverage and rejected traces, then consider a shared citation example and counterfactual policy values as separate experiments before claiming improved understanding.

## Recommendation

Keep the hybrid path as an experiment, not as an automatic replacement. Content/citation proxies improve without a regression in these metrics or answer/abstain decisions, but the failure rows and added adapter lifecycle still require review.

## Prioritized Next Experiments

1. Expand held-out paraphrases and policy-change cases before tuning the adapter again.
2. Compare a stronger small instruction model using the same retrieval and metric contracts.
3. Review invalid-citation cases and adjust training data only after identifying a repeatable failure pattern.
