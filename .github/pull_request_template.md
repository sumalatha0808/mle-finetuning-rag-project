# Checklist for the Fine-Tuning + RAG Project

<!-- If you are done with a topic mark the checkboxes with an `x` (like `[x]`) -->

- [ ] I read and understood the tasks.
- [ ] I validated the policy sources, training examples, and evaluation cases and kept them separate.
- [ ] I chunked, embedded, and indexed the policy sources in a real vector index, preserving source metadata.
- [ ] I defined a top-k setting and an evidence threshold so unsupported questions abstain.
- [ ] I ran a real local LoRA training loop and saved the adapter separately from the base model.
- [ ] I built baseline RAG and hybrid RAG + LoRA answer paths that use the same retriever and prompt format.
- [ ] Generated citations are validated against the retrieved evidence.
- [ ] I evaluated both systems on the same held-out cases and reported retrieval and answer metrics separately.
- [ ] I wrote `reports/evaluation_report.md` with failure analysis, adapter trade-offs, and a recommendation.
- [ ] I added tests and they pass from the documented environment.
- [ ] I documented how to run the project and answered the submission questions.

Bonus:

- [ ] I included a base-model-only result to show the value of retrieval.
- [ ] I added an LLM-based judge as a separate metric while keeping a no-key evaluation path.
