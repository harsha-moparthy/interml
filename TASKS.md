# Implementation tasks

Planning baseline: 2026-10-24.
Build the index first. Then complete the 72 selected lessons. Extend the index to 1,100 questions and attach two readings to every question.
Complete tasks in order. Record results in `STATUS.MD`.

Completion record: T01–T18, C01–C36, T12, and T13 are complete as of
2026-10-24. `STATUS.MD` contains the verification evidence and remaining limits.

| ID | Task | Depends on | Completion check |
| --- | --- | --- | --- |
| T01 | Inspect source structure and indexes | None | Record counts and compare IDs and titles |
| T02 | Resolve client requirements | T01 | Record scope, audience, runtime, storage, and teaching priorities |
| T03 | Rewrite the plan and select lessons | T02 | Use STE style; select exactly two questions per chapter with reasons |
| T04 | Audit PDF and structured source fidelity | T03 | Record hashes, PDF pages, question IDs, code comparisons, and discrepancies |
| T05 | Define data models and local application setup | T04 | Define schemas, imports, database boundaries, and startup commands |
| T06 | Build import and synchronization checks | T05 | Import 600 records; detect duplicates, missing solutions, and source drift |
| T07 | Build the complete question index | T06 | Search by ID/title/chapter/topic; preserve numbering and source links |
| T08 | Build question pages and solution controls | T07 | All 600 pages show matching prompts; show/hide controls work |
| T09 | Build article storage and lesson pages | T08 | Content persists; prerequisite links and media paths resolve |
| T10 | Build explanation components | T09 | Shared state, linked views, reversible steps, comparisons, prediction reveal, and authored step help pass checks |
| T11 | Add browser Python and NumPy | T10 | Verify execution, stop/reset, output, timeout, and unsupported-runtime labels |
| C01–C36 | Complete chapter batches listed below | T11; then prior batch | Meet the lesson completion check for both selected questions |
| T12 | Audit all lesson coverage and behavior | C36 | Verify 72 complete lessons, prerequisites, examples, links, and accessibility |
| T13 | Package the local first version | T12 | Startup instructions work; dependencies, backup, and runtime limits are documented |
| T16 | Import the 500 supplemental questions | T13 | Preserve source order as IDs 601–1100; render Markdown equations; update chapters and navigation |
| T17 | Curate external readings | T16 | Store exactly two ranked technical readings for every one of the 1,100 questions; validate URLs and coverage |
| T18 | Retire lesson animations | T17 | Remove animation blocks and registry entries; keep static figures, equations, checks, walkthroughs, and playgrounds |

## Chapter batches

Read `planning/LESSON_SELECTION.md` for exact titles, reasons, and teaching approaches.
Apply `planning/TEACHING_GUIDE.md` to every selected lesson.
Choose methods for the concept. Do not require every interaction type in every lesson.
Create required prerequisite articles within each batch.
Read later topics when necessary to keep shared explanations consistent.

| Batch | Selected IDs | Deliverable |
| --- | --- | --- |
| C01 | 016, 017 | Two deep lessons and required prerequisite articles |
| C02 | 027, 030 | Two deep lessons and required prerequisite articles |
| C03 | 054, 057 | Two deep lessons and required prerequisite articles |
| C04 | 066, 077 | Two deep lessons and required prerequisite articles |
| C05 | 092, 100 | Two deep lessons and required prerequisite articles |
| C06 | 108, 113 | Two deep lessons and required prerequisite articles |
| C07 | 122, 132 | Two deep lessons and required prerequisite articles |
| C08 | 137, 152 | Two deep lessons and required prerequisite articles |
| C09 | 162, 168 | Two deep lessons and required prerequisite articles |
| C10 | 181, 182 | Two deep lessons and required prerequisite articles |
| C11 | 207, 209 | Two deep lessons and required prerequisite articles |
| C12 | 233, 241 | Two deep lessons and required prerequisite articles |
| C13 | 248, 249 | Two deep lessons and required prerequisite articles |
| C14 | 267, 287 | Two deep lessons and required prerequisite articles |
| C15 | 294, 308 | Two deep lessons and required prerequisite articles |
| C16 | 316, 328 | Two deep lessons and required prerequisite articles |
| C17 | 331, 339 | Two deep lessons and required prerequisite articles |
| C18 | 354, 361 | Two deep lessons and required prerequisite articles |
| C19 | 374, 379 | Two deep lessons and required prerequisite articles |
| C20 | 391, 397 | Two deep lessons and required prerequisite articles |
| C21 | 409, 413 | Two deep lessons and required prerequisite articles |
| C22 | 420, 429 | Two deep lessons and required prerequisite articles |
| C23 | 433, 439 | Two deep lessons and required prerequisite articles |
| C24 | 461, 462 | Two deep lessons and required prerequisite articles |
| C25 | 469, 476 | Two deep lessons and required prerequisite articles |
| C26 | 485, 499 | Two deep lessons and required prerequisite articles |
| C27 | 503, 509 | Two deep lessons and required prerequisite articles |
| C28 | 515, 519 | Two deep lessons and required prerequisite articles |
| C29 | 525, 528 | Two deep lessons and required prerequisite articles |
| C30 | 542, 545 | Two deep lessons and required prerequisite articles |
| C31 | 549, 555 | Two deep lessons and required prerequisite articles |
| C32 | 557, 562 | Two deep lessons and required prerequisite articles |
| C33 | 567, 573 | Two deep lessons and required prerequisite articles |
| C34 | 577, 581 | Two deep lessons and required prerequisite articles |
| C35 | 588, 592 | Two deep lessons and required prerequisite articles |
| C36 | 593, 598 | Two deep lessons and required prerequisite articles |

## Lesson completion check

Explain what the question asks and why the concept is useful.
Define inputs, outputs, symbols, shapes, and constraints.
Provide a checked worked example and relevant equations.
Provide a useful static visual or playground when it materially improves understanding.
Connect the explanation to the original solution.
Start with the mechanism. Then provide worked and independent examples.
Include a prediction question and a new case with checked, explained answers.
Include a useful failure case or a controlled comparison.
Provide two rigorous external readings.
Verify equations, numerical values, source links, and edited playground inputs.
Keep learner feedback separate from implementation checks.
Verify all prerequisites, source references, and media links.
Record checks and limitations before marking the batch complete.
