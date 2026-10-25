# ML coding learning website

Status: Reference-first local version implemented and verified. See `STATUS.MD` for evidence.
Date: 2026-10-24 (Asia/Kolkata).

## Goal

Help an intermediate learner understand difficult ML concepts and the questions that use them.
Make explanations the main part of the website.
Connect each explanation to a worked example, the mathematics, and the supplied solution.

## First-version scope

- Include the original 600 questions with IDs 001–600.
- Include all 500 questions from `addl_questions.md` with IDs 601–1100.
- Give every question exactly two ranked technical readings.
- Build the complete question index first.
- Then complete two important, difficult questions from each of the 36 chapters.
- Deliver 72 deep lessons with all required prerequisite articles.
- Keep solutions hidden until the learner selects Show solution. Provide Hide solution.
- Run locally for one learner.
- Run supported Python and NumPy exercises in the browser.
- Store articles in a separate content database within the same application.
- Store images and other media as linked local files.

Use `planning/LESSON_SELECTION.md` for the 72 selected questions and their teaching approaches.
“Chapter” means one of the workbook's 36 sections. It does not mean one of its 121 subsections.
The learner has intermediate experience. Do not assume knowledge of every ML topic.
No interview deadline was supplied. Do not invent one.
The first version does not require accounts, hosting, an AI tutor, or a progress system.
These features must not delay the explanations.

## Learner experience

The index supports search by ID, title, chapter, and topic.
Each entry shows whether a deep lesson is available.
Every question page shows the original task, two technical readings, and a collapsible solution or answer guide.
The 72 selected pages add the complete learning sequence below.

1. State what the question asks in plain language.
2. Explain why the operation is useful.
3. List the required concepts and link to their articles.
4. Define inputs, outputs, symbols, shapes, and constraints.
5. Work through a small numerical or text example.
6. Derive the relevant equations and explain each term.
7. Let the learner inspect the computation with static diagrams or a browser playground when useful.
8. Explain common mistakes and boundary cases.
9. Connect the supplied solution to the explanation.
10. Link two sources that supply the strongest derivation or implementation detail.

Keep the original prompt distinct from the teaching explanation.
Preserve the original contract, including masks, axes, allowed primitives, and supplied callbacks.
Do not claim that an educational simulation implements a complete production system.

## Teaching approach

Use guided discovery first. Then allow free exploration.
Follow [the teaching guide](planning/TEACHING_GUIDE.md) when writing each deep lesson.
Start with a concrete puzzle and a small, complete computation.
Reveal detail gradually. Let the learner revisit or skip familiar steps.

Use construction, prediction, controlled comparisons, failure repair, and component removal where they explain the concept.
Let the learner trace one object across values, shapes, equations, diagrams, and code.
End with a new example that requires the learner to apply the idea independently.
Judge the explanation by what the learner can derive, implement, and explain after reading it.

Provide authored help for the current step.
Offer smaller examples, prerequisites, expanded equations, alternative views, mistakes, and code connections where relevant.
Preserve the current computation state when help opens.
This feature does not require an AI tutor.

## Explanation and reading standard

Do not use lesson animations. Use static diagrams, equations, worked examples, code walkthroughs, and browser playgrounds.
Each visual must answer one learning question. Label axes, shapes, data ownership, and invented values.
Do not imply that illustrative embeddings or attention weights are learned semantic facts.

Attach exactly two readings to each question. Rank them by relevance.
Prefer original papers, official documentation, open textbooks, and respected technical authors.
Use a paid article only when it is stronger than the open alternatives. Mark access limits.
Choose sources that contain derivations, implementation details, measurements, or failure modes.
Do not use shallow career summaries or keyword-only articles.
For theory questions, provide an answer guide that defines the required mechanism, assumptions, comparison axes, and boundary cases.
For coding questions, state the required shapes, edge cases, tests, and complexity.
For system-design questions, state measurable requirements and trace data through training, serving, monitoring, and rollback.

## Sources and solution fidelity

The source folder is `ml-interview/`.
The legacy filenames contain `500`. The current edition contains 600 exercises.
Keep IDs 001–600 unchanged. Store chapter order separately from identity.

Use `ml-coding-patterns-500-solutions.pdf` as the requested solution reference.
Use its `.tex` source and `ml-coding-workbook/workbook.json` to obtain clean code.
Do not copy executable code blindly from extracted PDF text.
PDF extraction can change minus signs, spacing, and line breaks.
All 600 TeX solution blocks currently match the structured workbook code.
Verify PDF identity, question labels, page mapping, and any suspected mismatch during the import audit.
Record source hashes and PDF page references.
Preserve source code. Put explanations and any proposed correction in separate fields.

Use `ml-coding-catalog/catalog.json` for titles, ordering, and source metadata.
Use one canonical exercise record for both question and solution views.
Check the structured workbook against the PDF source before importing it.
Retain original source links, attribution, and license text.
Source websites may state different contracts. The supplied workbook contract controls this website's exercise.

## Data design

Use a local exercise database and a separate local content database.
SQLite is the planned storage format. Keep both within the application folder.
Use a small local service for persistent reads, imports, and exports.
Select frontend libraries during implementation. No hosted service is required.
Keep authored article sources and schema migrations under version control.
Use a reproducible import to build database records.
Store media paths relative to the application. Avoid machine-specific paths.

| Record | Required information |
| --- | --- |
| Exercise | Permanent ID, chapter, title, original prompt, solution, language, source, PDF pages, source hash |
| ExerciseResource | Exercise ID, rank, title, URL, publisher, access note, selection reason |
| Article | Permanent ID, title, structured content, revision, sources, review state |
| ExerciseArticle | Exercise ID, article ID, purpose, reading order |
| ArticlePrerequisite | Article ID and required earlier article ID |
| Media | File path, description, alternative text, attribution |
| LearningCheck | Article ID, prediction or new case, inputs, checked answer, explanation |
| Playground | Exercise ID, runtime support, starter code, checks, execution limits |

Validate cross-database references during import and before delivery.
Reject duplicate question IDs, missing solutions, missing media, and broken article links.
Check prerequisite links for cycles.
Make repeated imports safe. Do not overwrite authored articles.
Display separate counts for 1,100 imported questions and 72 completed deep lessons.

## Browser execution

Use a browser Python runtime with NumPy support. Evaluate Pyodide during implementation.
Run learner code in a worker so it does not freeze the page.
Provide run, stop, reset, output, errors, and an execution timeout.
Limit example input sizes. State browser memory limits accurately.
Keep runtime loading and failure states clear.
Keep code execution separate from the local database service.
Do not expose arbitrary host commands or filesystem access to learner code.

Do not promise browser execution for PyTorch, CUDA, or Triton solutions.
Show their supplied code and use clearly labeled numerical demonstrations where suitable.
A conceptual NumPy demonstration is not a replacement for the original solution.
Document required initial downloads. Full offline installation is a later packaging decision.

## Delivery sequence

1. Audit the source and preserve solution identity.
2. Build imports and synchronization checks.
3. Deliver the searchable 1,100-question index and solution controls.
4. Build the article database and reusable explanation components.
5. Complete chapters C01–C36, with two deep lessons per chapter.
6. Add prerequisite articles whenever a lesson requires them.
7. Verify all 72 lessons and provide local startup instructions.

Use `TASKS.md` for task order and completion checks.
Update `STATUS.MD` after each task and chapter batch.
Do not mark a chapter complete while its prerequisites are missing.

## Acceptance criteria

The index contains exactly 1,100 unique IDs in order.
IDs 001–600 match the workbook. IDs 601–1100 match `addl_questions.md`.
Every question has exactly two ranked technical readings.
Every question opens its matching supplied solution.
Solutions are hidden by default and can be shown and hidden.
Exactly two selected exercises per chapter have complete deep lessons.
All prerequisite links for those 72 lessons resolve to complete articles.
Each lesson connects its prompt, example, equations, visual, and solution.
Each lesson follows the completion checks in `planning/TEACHING_GUIDE.md`.
Each lesson includes a checked prediction and a new case with explained answers.
Each lesson includes a useful failure case or controlled comparison.
Step help resolves to relevant authored content.
Record learner feedback separately from technical verification.
The selected browser runtime runs supported examples and handles interruption.
The website starts locally from documented instructions.
Synchronization, link, accessibility, and coverage checks pass.

## Writing rules

Use ASD-STE100 principles for instructions and descriptive text.
Use short sentences, active voice, and one instruction per sentence.
Use one defined technical term for each concept. Preserve mathematical precision.
Keep instructions within 20 words where practical. Keep descriptive sentences within 25 words where practical.
Define specialist terms before use. Keep symbols consistent across text, diagrams, and code.
This is an STE-style document. Formal dictionary compliance has not been certified.
Reference: https://www.asd-ste100.org/about_STE.html
