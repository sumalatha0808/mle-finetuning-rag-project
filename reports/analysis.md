*Numbers below are from the run recorded in `reports/evaluation_results.json`. Re-check this section if you retrain or rerun the benchmark.*

### 8.1 What changed when the adapter was loaded

The two systems share the retriever, the evidence, the prompt, the threshold, and the base weights. The adapter is the only difference.

- **Citation format is the main change.** Baseline RAG never wrote a `[source]` citation, so all 12 generated answers were **rejected**. The hybrid wrote valid citations in 6 of 12, and those were **answered**. This one change produces most of the hybrid's gain on every aggregate metric: decision accuracy 0.143 → 0.500, faithfulness 0.143 → 0.429, style 0.143 → 0.554, relevance 0.143 → 0.250.
- **Content did not improve.** Before citation checks, the baseline's raw answers covered **more** of the required terms than the hybrid's (0.500 vs 0.400). The baseline often had the right fact and was rejected only for the missing citation: `eval-standard-shipping`, `eval-pause`, `eval-outage-updates`, `eval-duplicate-payment`.
- **Abstention did not improve.** Neither system ever produced the fallback reply. Both off-topic cases abstained only because retrieval scored them below `min_score`.
- **Answers got longer and slower.** The hybrid often copies whole evidence paragraphs until it reaches the 80-token limit (`eval-duplicate-payment`, `eval-standard-shipping`, `eval-shipping-compensation`). Its median generation latency is about twice the baseline's.

### 8.2 Failure and fragile-success analyses

**Failure 1: the hybrid answered an unsupported question with an invented claim (`eval-bank-chargeback`).**
Retrieval passed the `refunds` chunk (score 0.592, above the 0.30 threshold) because the question shares the duplicate-payment topic. The hybrid replied *"The external bank will take the full 30-day refund window before the cancellation. [refunds]"*. The policy says nothing about bank chargeback timing. The citation is valid, since `refunds` was retrieved, so the validator **served** the answer. Baseline RAG explained correctly that the context does not cover this, but it was rejected for having no citation. This is the most serious regression: a valid citation is not the same as a supported claim.

**Failure 2: the faithfulness proxy accepts wrong answers (`eval-packed-address`, `eval-authenticator`).**
- `eval-packed-address`: the hybrid answered *"Yes, the support can change the address before the warehouse packs the order. … [shipping]"* to a question about **after** packing. The answer is evasive and suggests a change is possible.
- `eval-authenticator`: the hybrid omitted both required facts (registered email, secure upload portal) and cited `incidents` and `response-style` text that is unrelated to the question.

Both count as *faithful* because they cite an expected source and contain no listed forbidden phrase word for word. Of the hybrid's 6 served answers, a manual reading finds only `eval-duplicate-payment` correct. Even that one includes the unrelated *"final 24 hours"* refund rule and copies the style guide into the reply. Treat the 0.429 faithfulness score as an **upper bound**.

**Fragile success / missed failure: forbidden claims are only caught word for word (`eval-log-secrets`).**
Both systems said that diagnostic logs **may** contain authentication tokens. The policy says they **must not**. The forbidden phrase is *"authentication tokens may be included"*, so neither answer matched it. Both were rejected only for missing citations. If either model had added `[incidents]`, the reply would have been served and the safety error would have gone unnoticed by every metric except required-term coverage.

**Fragile success: the retrieval threshold, not the model, handles abstention.**
`eval-cafeteria` and `eval-bicycle` are correct for both systems only because their top scores (~0.11) fall below `min_score`. On the two near-domain unsupported cases, which score above the threshold, neither model abstained.

### 8.3 Retrieval errors that limit both systems

- **Near-domain unsupported questions pass the threshold.** `eval-bank-chargeback` (0.592) and `eval-shipping-compensation` (0.566) score higher than several supported cases (for example, `eval-duplicate-payment` at 0.399). As the threshold sweep in section 4 shows, no `min_score` removes them without refusing supported questions. Abstention therefore depends on the generator, and neither model provides it.
- **Each document is one chunk.** With 600-character merging, every policy file becomes a single chunk (`refunds#0`, `shipping#0`, …). Recall@3 and MRR are perfect, but the evidence holds every rule in the document. Both models copy unrelated rules into their answers. For example, the hybrid's `eval-standard-shipping` reply restates the whole shipping policy.
- **Extra low-score chunks act as distractors.** `top_k=3` adds chunks just above the threshold: `subscriptions` at 0.37 for `eval-refund-ticket` and `incidents` at 0.315 for `eval-pause`. The hybrid answered `eval-refund-ticket` from the subscriptions text and cited `[incidents]` for `eval-pause`.

### 8.4 Cost

- **Training** takes about 4–5 minutes on a laptop CPU and trains 0.34% of the parameters. It produces a 1.8 MB adapter that must be versioned with the base-model revision and the training-data hash, both recorded in `training_summary.json`.
- **Inference** loads at about the same speed (one extra small file). But the hybrid's longer outputs make it slower per answer on this CPU (mean 5.13 s vs 3.83 s).
- **Maintenance**: the adapter must be retrained and re-benchmarked whenever the base model, the prompt format, or the training examples change. Policy *content* changes need only re-indexing, because facts come from retrieval.

### 8.5 Recommendation: **revise** the hybrid design (do not keep it as is, do not reject it yet)

On held-out behaviour, the adapter makes the model **pass the citation validator** much more often. That gain is real, but it is mostly *format*. Manual reading shows that the served answers are mostly wrong or evasive, and that the hybrid served an invented answer to an unsupported question that the baseline did not serve. Baseline RAG on this 135M model serves nothing useful either, because every generated answer is rejected. So **neither system is deployable**. The falling training loss (1.547 → 0.519) does not change this conclusion.

Revisions, in priority order:

1. **Add an abstention check after generation.** Reject answers whose key claims have no supporting overlap with the cited chunk, so a valid-looking citation (as in Failure 1) no longer gets an unsupported claim served.
2. **Retrain with more abstention and contrastive examples.** The training set has only one insufficient-evidence example (`train-style-02`). Add near-domain unsupported questions (not the evaluation cases) and *after packing* vs *before packing* pairs.
3. **Use paragraph-sized chunks** (`max_chars≈250`) so evidence contains only the relevant rule.
4. **Add a semantic or manual faithfulness label** next to the lexical proxy, and match forbidden claims by meaning rather than exact string.

### 8.6 Answers to the project questions

1. **Which held-out behaviours changed after loading the adapter?** Citation formatting changed: rejections fell from 12/14 to 6/14. Answers became shorter in some cases and verbose copies of the evidence in others. One unsupported question (`eval-bank-chargeback`) went from a rejected correct refusal to a served invented answer.
2. **Did the adapter improve evidence use, style, both, or neither?** It improved **style and citation format** (style 0.143 → 0.554). It did **not** improve evidence use: raw required-term coverage fell (0.500 → 0.400), and several citations point to the wrong retrieved source.
3. **Which retrieval errors limit both systems regardless of fine-tuning?** Near-domain unsupported questions score above any usable threshold. Whole-document chunks put unrelated rules into the evidence. `top_k=3` adds distractor chunks.
4. **Does the measured improvement justify adapter training and maintenance?** **Not yet.** Training is cheap: minutes on a CPU and a 1.8 MB file. But the measured gain is mostly a format effect that lets wrong answers through the validator, and it adds one unsafe served answer. The current design does not justify deployment.
5. **What experiment would most reduce uncertainty in that decision?** Give baseline RAG a one-shot citation example in the prompt and rerun both systems, using paragraph chunks and a manual faithfulness label. If the prompted baseline matches the hybrid's citation rate, the adapter adds nothing beyond prompting. If the hybrid still leads on *correct* served answers, retraining with more abstention examples, across several seeds, is worth the maintenance cost. A larger base model (for example SmolLM2-360M) is the second experiment, because 135M parameters may limit both systems.
