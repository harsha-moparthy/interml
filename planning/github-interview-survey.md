# ML and AI interview additions

This review adds 36 original questions (1101–1136) to the existing appendix chapters. FDE material is excluded. The file-level retrieval ledger is `github-interview-survey.json`; each new question also retains a GitHub discovery URL in `app/content/interview_additions.json`. These links document related topic discovery, not verbatim question ownership or authority for the answer.

The reviewed GitHub material covers ML foundations, LLMs, prompting, fine-tuning, agents, evaluation, inference, safety, multimodal systems, and MLOps. Retrieval encountered GitHub's unauthenticated API limit; this is a bounded gap review, not an exhaustive scrape of all GitHub. Failed downloads are not counted as reviewed files. Existing appendix coverage was retained rather than duplicated wholesale.

| Appendix | New IDs | Focus |
|---|---|---|
| A03 | 1101–1111, 1114 | BERT objectives and variants, representations, CRFs, adaptation, in-context learning |
| A04 | 1113 | Weight tying |
| A05 | 1129–1131 | Fairness, decision costs, interpretation |
| A06 | 1121, 1132 | Paired evaluation and information theory |
| A07 | 1120, 1122–1128 | Evaluation integrity, serving, reproducibility, registry and data lineage |
| A08 | 1133–1136 | Bandits and reinforcement learning |
| A10 | 1112, 1115–1119 | SFT, reasoning aggregation, agents, MCP, pass-at-k |

All answers are original explanatory prose with inline citations, examples, assumptions, and failure cases. Five further readings appear beneath each answer. GitHub lists informed the topic review; textbook chapters, original papers, and official documentation support the explanations. The examples and BERT diagram are newly authored rather than copied book pages or slides.

Textbook anchors include Jurafsky and Martin's *Speech and Language Processing*, August 19, 2026 draft (chapters 5, 8, 9, and 18); *Dive into Deep Learning*'s BERT chapter; Goodfellow, Bengio, and Courville's *Deep Learning*; Murphy's *Probabilistic Machine Learning*; Sutton and Barto's *Reinforcement Learning*; and Barocas, Hardt, and Narayanan's *Fairness and Machine Learning*. A reading-list entry is a recommended resource, not a claim that the entire book was read. Some textbook download hosts were unavailable during verification; explanations also cite the relevant original papers.

Maintenance: edit prompts in `app/content/interview_additions.json`, prose in `tools/appendix/expanded/qNNNN.md`, and the five readings in `app/content/solutions/qNNNN.json`. Run `python3 tools/appendix/apply_expanded.py`, import the database, run tests and `node tools/appendix/check.mjs`, then export the static API and build. Never export concurrently with backend tests or build concurrently with export. The original audited `addl_questions.md` remains unchanged.

Validation completed on November 3, 2026: 81 backend unittest cases and 138 frontend tests passed. The content checker validated all 536 appendix answers, their five readings, citations, equations, and diagrams. Production-browser checks rendered all 536 answers, exercised all 18 chapter filters, verified fenced code and diagram loading, and checked narrow-screen layouts. The static export contains 1,136 questions and 1,136 solutions, without test fixtures. The GitHub Pages production build passed using a local temporary output directory to avoid slow network-filesystem asset writes.
