# Maintained solution code

`q001`–`q652` are the editable implementations displayed by the study site.
`q653`–`q684` contain authored system-design answers without code files.
Coding manifests identify a standalone Python/CUDA file. Optional `readings`
replace the two exercise resources; optional `answer` adds an explanation.
The importer validates these files and applies them after reading the original
workbook. Re-importing therefore preserves edits, original question IDs, source
attribution, and license notes.

This pass adds problem contracts to all 600 examples, consistent formatting,
comments explaining selected numerical/state invariants, and clearer local names
in attention, tokenization, calibration, retrieval, and workflow implementations.
Q316 additionally uses batched attention, computes exponentials once, and raises
explicit errors for invalid head counts. Public call signatures remain compatible.

These remain focused interview reference implementations. Formatting and passing
regression checks do not turn every example into a deployable production service.
In particular, simulation, in-memory persistence, and reference GPU kernels retain
the limitations stated in their contracts. CUDA/Triton execution requires a GPU
and was not validated by the CPU regression run.

## Editing principles

- Keep the algorithm visible. Avoid extra classes, generic frameworks, or wrappers
  unless they model state the problem actually needs.
- Preserve public function names/arguments so lessons and tests keep working.
- Name bookkeeping variables clearly; retain familiar mathematical notation where
  it makes equations easier to follow.
- Explain shapes, mutation, masking conventions, and important edge behavior.
- Comment *why* a step exists, especially normalization, tie-breaking, numerical
  stability, and transactional boundaries. Avoid narrating obvious assignments.
- Prefer a readable implementation over a clever one-liner. Document expensive
  copies and limitations instead of describing every example as production-ready.
- Resource links must explain the actual operation. Do not use a nearby topic as
  a substitute. Pair an accessible explanation with an implementation reference.

## Verification and rebuilding

From the repository root:

```sh
app/.venv-ref/bin/python tools/verify_solutions.py
app/.venv-ref/bin/python -m unittest discover -s tools/tests -v
cd app/backend
../.venv/bin/python -m placement.importer
../.venv/bin/python -m unittest discover -s tests
../.venv/bin/python -m placement.export_static
```

The first command needs the original local workbook/test fixtures. It copies them
into temporary storage and runs their existing checks against these files, without
modifying the original archive or its reports. The bank fixtures' Windows paths
are normalized so their tests actually run on Linux.

Python formatting: Black 26.3.1, 88 columns. CUDA formatting: clang-format 21.1.8,
LLVM style, four-space indentation, 88 columns.

## Authored appendix answers

All 500 questions in A01–A18 have authored answers, inline citations, and five
ranked resources each. A01 retains its 52 standalone implementations. Answers
open beneath the original question, with rendered equations and original SVG
figures where useful. Company-named system designs are illustrative proposals.

The authoring scripts and source catalog are in `tools/appendix/`. Run the numbered
`a*.py` scripts in order, then `diagrams.py`, to regenerate the manifests. Run
`node tools/appendix/check.mjs` from the repository root to validate citation
mapping, five-resource coverage, all math expressions, and figure paths.

## Validation of this pass (2026-10-28)

- All 591 Python files compile without warnings; 584 CPU-compatible files import.
- Original suites: 69/69 grouped checks (187 exercises), 100/100 project checks,
  and 245/245 repository fixture cases. These do not prove exhaustive behavioral
  coverage of all 600 exercises.
- Four focused regression tests pass, including comparison of rewritten multi-head
  attention with PyTorch, invalid head counts, masked rows, and zero denominators.
- Backend: 76/76 tests; frontend: 138/138 tests.
- The question-page browser suite passes, including copied-code equality and source
  attribution, with no page/console errors.
- Walkthrough anchors were updated in 71 lessons; literal anchors in the lesson
  generator resolve against the maintained code. Search continues to use original
  concept terms so presentation edits do not perturb ranking.

## Appendix verification

The 52 A01 tests compare implementations with library operations or independently
computed results, including gradients, cache equivalence, speculative-sampling
frequencies, metric ties, and final partial accumulation batches. These are CPU
checks; CUDA mixed-precision execution and production-scale performance remain
unvalidated. System-design capacity numbers are illustrative assumptions.

Validated after the appendix update: 56 numerical/regression tests, 78 backend
tests, 138 frontend tests, and the question-page browser suite. Browser checks cover all 500 appendix answers, citations, five-resource lists,
figure loading, mobile layouts, and automatic answer visibility.

## Full appendix validation (2026-11-02)

- 500 authored answers, 2,500 reading entries, and 835 inline citations.
- All 920 answer math expressions parse with strict KaTeX checks.
- Six original SVG diagrams appear in nine relevant answers.
- Browser traversal renders all 500 answers with citations and five resources;
  figure loading, mobile overflow, and math inside emphasized prompts pass.
- 78 backend, 138 frontend, and 56 numerical/regression tests passed.
- Internal lesson links and whitespace checks pass. External sources were
  researched selectively; this is not a claim of an exhaustive live URL audit.
