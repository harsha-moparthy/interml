# Status

Updated: 2026-10-25 (Asia/Kolkata).
Phase: Reference-first 1,100-question version complete and verified.

## Agreed scope

Include all 600 workbook questions. Preserve IDs 001–600.
Include all 500 questions from `addl_questions.md` as IDs 601–1100.
Attach two ranked technical readings to every question.
Build the index first. Then complete two deep lessons per chapter.
The first version has 72 deep lessons across 36 chapters.
Provide all prerequisite articles for those lessons.
Prioritize explanations, equations, worked examples, primary sources, and technical articles.
Do not render lesson animations.
Use local personal access, browser Python/NumPy, and a separate in-application content database.
Use the supplied PDF solutions with show/hide controls.
No further client answer is required for the planning baseline.

## Task state

| Task | State | Evidence or next action |
| --- | --- | --- |
| T01 | Complete | Compared 600 IDs and titles across catalog and Markdown documents |
| T02 | Complete | Incorporated all seven client answers into `LLM_PLAN.md` |
| T03 | Complete | Rewrote plan and tasks; selected 72 lessons in `planning/LESSON_SELECTION.md` |
| T04 | Complete | `planning/SOURCE_AUDIT.md`; 600/600 titles and code blocks match the PDF; page ranges in `planning/source-audit.json` |
| T05 | Complete | `planning/ARCHITECTURE.md`; schemas in `app/backend/schema/`; service starts on 127.0.0.1:8765 and `/api/health` returns ok; frontend scaffold deferred to T07 |
| T06 | Complete | `python -m placement.importer` builds `exercises.db` (600 exercises, 36 chapters, 79 license-tagged); `python -m placement.check_links`; 11 tests pass in `app/backend/tests` |
| T07 | Complete | `/api/stats`, `/api/chapters`, `/api/exercises` (search, chapter, lesson filters); React index in `app/frontend`; browser-checked in T08 |
| T08 | Complete | `#/question/N` pages; `/api/exercises/{id}` omits the solution, `/api/exercises/{id}/solution` serves it only after Show solution; all 600 prompts and solutions compared with `workbook.json` in tests (22 pass); 14 browser checks pass with Chromium (`app/e2e/questions.mjs`) |
| T09 | Complete | `python -m placement.articles` imports authored JSON into `content.db` with validation (`planning/CONTENT_FORMAT.md`); `/api/articles`, `/api/articles/{id}`; React article page with KaTeX equations, images, tables, callouts, prerequisite and "Used by" links; question pages list linked articles. 31 backend tests and 15 browser checks pass using demo fixtures in `app/e2e/fixtures`. No real article exists yet; the first are written in C01/C02 |
| T10 | Complete | Shared reducer, labelled timeline, play/pause/forward/back/reset, keyboard keys, reduced motion, text version, linked selection across matrices, equation and code, input editing with limits, prediction/new-case/explain-back checks with specific feedback, compare block, step help that keeps state. Reference component `broadcast-trace`. 36 backend tests, 17 Node tests (304 shape pairs checked against NumPy), 3 browser suites (questions, articles, demo: 47 checks) pass. Fixture lesson in `app/e2e/fixtures` is demo content, not a real lesson |
| T11 | Complete | Pyodide 0.27.7 + NumPy 2.0.2 self-hosted; Web Worker runner with Run, Stop (also while loading), Reset, streamed output, error tracebacks, per-playground timeout, output and code caps, crash and missing-runtime handling; `playground` block and `display-only` label for PyTorch/CUDA/Triton; 42 backend tests, 17 Node tests, 4 browser suites (112 checks) pass. First-run download documented in `planning/ARCHITECTURE.md` |
| T11a | Complete | pandas 2.2.3 added to the browser runtime. Packages load per playground before the timed run and are pre-imported, so a slow `import pandas` does not use up the time limit. 43 backend tests, 17 Node tests, 4 browser suites (123 checks) pass |
| T14 | Complete | Hybrid search (keywords + local semantic model all-MiniLM-L6-v2, typo correction, abbreviations, filters, snippets) and CodeMirror code viewer with highlighting, line numbers, copy, wrap, and solution walkthrough notes. Measured: hit@5 0.94 / MRR 0.75 on 34 tuning queries, hit@5 0.96 / MRR 0.92 on 24 held-out queries (queries written by the assistant, small sample) |
| T15 | Superseded | The former interactive lesson-player work was retired by T18. Static article content and playgrounds remain |
| T16 | Complete | Parsed all 500 numbered entries in `addl_questions.md`; imported them as IDs 601–1100 in chapters A01–A18; preserved multiline equations and source order |
| T17 | Complete | Added 2,200 ranked reading rows. Every question has exactly two links with title, publisher, access note, and reason. The catalog favors primary papers, official documentation, textbooks, and technical authors |
| T17b | Complete | 2026-10-24: replaced paper-first readings with about 180 blog, tutorial, and explainer pages. Each page was fetched and checked for its key terms (for example, the online-softmax questions now link to a post with an Online Softmax section and a derivation). 0 of 2,200 links point to arXiv or other papers; 1,100 questions keep exactly two ranked links. Known gaps with no good explainer found: Huber loss, AIC/BIC, GIoU/DIoU, epipolar geometry, catastrophic forgetting, LLM unlearning, lottery ticket, BLEU/ROUGE. `otexts.com/fpp3` blocks `curl` but opens in browsers. 61 backend tests pass |
| T17c | Complete | 2026-10-25: per-question readings. 1,043 of 1,100 questions now have two readings chosen for that question's own concept (for example GQA/MQA, RoPE, MLA, FlashAttention, DPO, GRPO, speculative decoding, ZeRO, SmoothQuant, DP-SGD, DINO). They live in `app/backend/placement/reading_groups/` (one module per area, plus `_aliases.py` for questions that share a concept). Every page was fetched, including Medium and Towards Data Science through a headless browser, and must contain the group's key terms. The other 57 questions keep the older rule-based picks; about 45 of them have no strong explainer yet (see the list in the T17c report). 849 distinct links across 2,200 rows; the most-shared link now serves 16 questions (before: 895 questions shared a link used by 15 or more). `tests/test_reading_groups.py` checks two distinct readings, no paper hosts and no duplicate ids. 65 backend tests pass |
| T18 | Complete | Removed demonstration and comparison blocks from 77 article sources and `content.db`; removed the special animated route and animation registry; retained static figures, equations, checks, code walkthroughs, and playgrounds |
| C01–C36 | Complete | Exactly 72 complete deep lessons: two selected questions in each chapter. Each has linked source code, a visual asset, an interactive trace, and at least two checks |
| T12 | Complete | Content/API audit covers all 72 lessons; browser audit opens 14 representative routes, operates their timelines, checks mobile overflow, and reports no browser errors |
| T13 | Complete | Local startup, rebuild, verification, backup, and representative routes are documented in `app/README.md` |
| T19 | Complete | 2026-10-25: interface redesign (OpenSpec change `redesign-learning-ui`, 49 tasks). One token file for light, dark and print themes with a theme switch and no flash; persistent header, skip link, `?` shortcut list; loading, empty and error states with retry; index dashboard with chapter list and progress, learner-state filters, restored scroll; question page with lesson cards, done and bookmark, arrow-key navigation; article page with linked contents, current-section mark, reading progress and Continue, reading time, callout icons, print styles; `#/progress` page with export, import and confirmed reset. Progress is browser-only. Measured: 125 Node tests pass (71 contrast checks at AA in both themes, 24 progress-store tests); 65 backend tests pass; browser suites on fixtures: questions 18, articles 15, ui 172, python 48; `ui` also passes on the real data; 30 screenshots (5 pages, 360/768/1280 px, light and dark) show no horizontal overflow. Fixed along the way: index overflowed 155 px at 768 px; article headings were unreadable in dark mode; two light-theme contrast failures; stale fixture `lesson-030`; `run_suites.sh` exit code. Not measured: browsers other than Chromium, screen readers, a physical printer (a PDF is produced), page speed, and a learner trial of the new look |
| T20 | Complete | 2026-10-25: published copy on GitHub Pages (OpenSpec change `host-exact-site-on-github-pages`). Live at https://harsha-moparthy.github.io/interml/ (static JSON export, in-browser keyword search and search by meaning with the 8-bit all-MiniLM-L6-v2 model, all served from the site itself). Fixed along the way: A01–A18 returned nothing in search; the dev server crashed when an export rewrote `public/static-api`. Measured against the local service on 27 fixed queries (`app/e2e/parity.mjs`, reference `parity_reference.json`): mean top-10 overlap 7.50 of 10 on the published site, worst query 3.0 (`handle imbalanced classes`); 2 queries below 5 (`handle imbalanced classes` 3.0, `train on multiple gpus` 4.0) and 3 at exactly 5 (`316`, `how to speed up inference of language models`, `compare two distributions`); no empty lists; first search incl. model load 4.6 s from this machine (cold browser, warm network). Throttled to 10 Mbit/s the first meaning search waited 38.9 s, then showed keywords + meaning. Tests: 138 Node, 71 backend (new export test adds about 70 s), `static.mjs` 10 and `appendix.mjs` 25 browser checks. Existing `ui` (173) and `questions` (18) suites pass on the published address; its JSON is byte-equal to the local API (54 chapters, 1,100 questions, 79 articles, 72 lessons). The repository is private but the Pages site and its solution JSON are public (79 questions carry a license note). Not measured: browsers other than Chromium, phones, mobile data, screen readers. Known gap: the older `search` browser suite still expects 600 questions and fails 6 checks (it failed the same before this change) |

## Chapter state

All selected deep lessons are authored, validated, imported into `content.db`, and linked from their question pages.

| Batch | Question IDs | State |
| --- | --- | --- |
| C01 | 016, 017 | Complete |
| C02 | 027, 030 | Complete |
| C03 | 054, 057 | Complete |
| C04 | 066, 077 | Complete |
| C05 | 092, 100 | Complete |
| C06 | 108, 113 | Complete |
| C07 | 122, 132 | Complete |
| C08 | 137, 152 | Complete |
| C09 | 162, 168 | Complete |
| C10 | 181, 182 | Complete |
| C11 | 207, 209 | Complete |
| C12 | 233, 241 | Complete |
| C13 | 248, 249 | Complete |
| C14 | 267, 287 | Complete |
| C15 | 294, 308 | Complete |
| C16 | 316, 328 | Complete |
| C17 | 331, 339 | Complete |
| C18 | 354, 361 | Complete |
| C19 | 374, 379 | Complete |
| C20 | 391, 397 | Complete |
| C21 | 409, 413 | Complete |
| C22 | 420, 429 | Complete |
| C23 | 433, 439 | Complete |
| C24 | 461, 462 | Complete |
| C25 | 469, 476 | Complete |
| C26 | 485, 499 | Complete |
| C27 | 503, 509 | Complete |
| C28 | 515, 519 | Complete |
| C29 | 525, 528 | Complete |
| C30 | 542, 545 | Complete |
| C31 | 549, 555 | Complete |
| C32 | 557, 562 | Complete |
| C33 | 567, 573 | Complete |
| C34 | 577, 581 | Complete |
| C35 | 588, 592 | Complete |
| C36 | 593, 598 | Complete |

## Verification

All 600 TeX solution blocks match structured workbook code.
All standalone Python solutions match the workbook at syntax-tree level.
The PDF text confirms its edition and the sampled cross-attention question contract.
The content import reports no schema or link errors. The backend suite passes 61 tests.
The production frontend build succeeds. All 30 frontend tests pass.
The database audit confirms 1,100 sequential IDs, 54 chapters, 500 supplemental records, and 2,200 reading links.
The curriculum audit confirms 72 unique complete lessons with static media, equations, checks, walkthroughs, and playgrounds.
Playwright verifies the 1,100-item index, original solution reveal, supplemental answer guides, two-link rendering, desktop layout, and mobile layout.
The link audit received successful responses from 76 of 80 unique destinations. Two paid publisher pages returned HTTP 403 to the automated client. Two canonical Stanford links encountered an environment-specific TLS error.

## Known gaps

- `demo.mjs` and `curriculum-lessons.mjs` are out of date (they test demonstrations that T18 removed; the second expects 600 questions). `scene/`, `lessons/`, most of `demo/`, and the CSS for them in `styles.css` are no longer loaded by the app; remove them in a cleanup task.

The first version is local-only. It has no authentication or cloud deployment.
GPU lessons explain and trace the supplied CUDA/Triton code but do not execute GPU kernels in the browser.
Dark-mode article screenshots were not reviewed. The compressed first-download size was not measured. Screen-reader behaviour was not tested with an actual screen reader.

## Teaching design update

The current design is reference-first and does not render lesson animations.
Static diagrams, equations, prose, checks, solution walkthroughs, and browser playgrounds remain.
The earlier animation pilots are historical work and are superseded by T18.
The delivered scope is 1,100 indexed questions, 2,200 ranked readings, and 72 static deep lessons.

## Reference-first update

The live index now contains 1,100 questions. Original IDs did not change.
Question pages show two ranked technical readings before linked local lessons.
Supplemental questions show an answer guide instead of an empty code panel.
The resource matcher uses narrow topic rules before broad chapter matching. Examples include the Transformer paper and D2L for scaled attention, RoFormer for RoPE, vLLM for KV cache, and the original DDPM paper plus Lilian Weng for diffusion.
All animation blocks are absent from authored JSON and the content database.

## Next task

Use the local website and collect learner feedback. Replace a reading when a more rigorous or clearer source is found.

## Update rule

Record completed work, check results, and remaining limits after each task.
Keep task IDs and question IDs consistent across all planning documents.
