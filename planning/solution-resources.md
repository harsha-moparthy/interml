# Resources for the solution and appendix rewrite

Checked 2026-10-28. Priority: existing C01–C36 solution code first; the appendix
rewrite now covers all 500 questions in A01–A18. The books below are optional, not prerequisites.
Publisher tables of contents were checked; full paid books have not been read.

## Optional purchases, in priority order

1. **Designing Machine Learning Systems — Chip Huyen.** Best first addition for
   training data, feature engineering, offline evaluation, deployment, drift,
   monitoring, and retraining. Particularly useful for A02's end-to-end ML systems.
   [Publisher and contents](https://www.oreilly.com/library/view/designing-machine-learning-systems/9781098107956/).
2. **Designing Data-Intensive Applications, second edition — Martin Kleppmann and
   Chris Riccomini.** Useful for the storage, consistency, reliability, and dataflow
   decisions underneath system-design answers.
   [Publisher and contents](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781098119058/).

If purchasing one book, choose the first. A lawful local PDF/EPUB copy would allow
chapter-specific references; summaries should remain original rather than copying
book passages. No claim is made that these books cover every newer LLM technique.

## Free references already useful for the code pass

- [Dive into Deep Learning: multi-head attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html): shape transformations and the implementation; used for Q316.
- [NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html): slices, advanced indexing, and copies.
- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html): singleton-axis rules.
- [NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html): matrix and batch dimensions.
- [NumPy ufunc basics](https://numpy.org/doc/stable/user/basics.ufuncs.html): repeated-index accumulation with `at`.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html): floating-point limitations; informs comparison tolerances.
- [WHATWG server-sent events](https://html.spec.whatwg.org/dev/server-sent-events.html): stream format for Q597.
- [Python incremental decoders](https://docs.python.org/3/library/codecs.html#incrementaldecoder-objects): UTF-8 state across chunk boundaries.
- [AWS: timeouts, retries, and backoff with jitter](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/): retries, time budgets, and idempotency for Q598 and later system-design answers.
- [Python JSON serialization](https://docs.python.org/3/library/json.html): deterministic object-key ordering and rejection of non-finite numbers for Q599.

## How appendix answers should use sources

Write the answer to the actual prompt: state assumptions, explain the mechanism,
work through an example or capacity estimate, and discuss a specific failure and
tradeoff. A list of technologies or instructions to read two links is not an
answer. Use authors' engineering blogs, official docs, open textbooks, and papers
when needed; a Medium article is useful only if it directly explains the topic.
Each question should name the relevant section and explain why the link helps.
Keep illustrative capacity numbers distinct from measured production results.

The first code pass overrides resources for 23 main-workbook questions. Other
existing links have been retained; this is not a completed relevance audit of all
1,100 questions of the main workbook. All appendix questions now have five selected resources.

## Appendix sources used in the first authored pass

- [Anthropic: building effective agents](https://www.anthropic.com/research/building-effective-agents): workflow boundaries, tool calls, and explicit stopping conditions.
- [Feast point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins): training feature timestamps and leakage prevention.
- [Uber QueryGPT](https://www.uber.com/blog/query-gpt/): schema context and the workflow around generated SQL.
- [Netflix artwork personalization](https://netflixtechblog.com/artwork-personalization-c589f074ad76): the artwork decision, distinct from recommending the underlying title.
- [Stripe Radar engineering](https://stripe.dev/blog/how-we-built-it-stripe-radar): operational context for fraud scoring.
- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad): industrial visual anomaly detection; replaces unrelated quantization readings.
- [Microsoft moderation and enforcement](https://www.microsoft.com/en-us/DigitalSafety/moderation-and-enforcement): automated detection, review, actions, and appeals.
- [How to Scale Your Model: sharding](https://jax-ml.github.io/scaling-book/sharding/): memory and communication reasoning for parallel training.

Each authored appendix manifest contains its own five reading links and explains
what to read them for. Public pages were used; paid book contents were not assumed.
