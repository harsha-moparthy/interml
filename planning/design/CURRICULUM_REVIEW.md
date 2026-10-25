# Curriculum integration review

Date: 2026-10-24

## Result

The local website contains all 600 questions with IDs 001–600. It contains
exactly 72 complete deep lessons: two selected questions in each of the 36
chapters.

Every selected lesson has:

- a plain-language statement of the task and required background;
- an equation, invariant, or explicit algorithm contract;
- a generated SVG that summarizes the mechanism;
- an interactive timeline with at least four reversible steps;
- a connection to the supplied source solution;
- a boundary, failure, or comparison case; and
- at least two learner checks with explained answers.

Question 030 uses its dedicated projected 3D broadcasting experience. The
other lessons use the redesigned concept laboratory derived from that visual
standard. It puts a large dotted visual stage before the controls, then provides
named chapters, a scrubber, synchronized explanations, object tracing,
controlled input changes, code highlighting, and text alternatives.

Attention, cross-attention, RoPE, and KV cache add a concept-first mode. These
views begin with words, token roles, memory ownership, and information flow.
The learner can switch to checked signed vector bars with `Inspect vectors`.
Exact values remain in accessible labels and the verified computation, but do not
dominate the visual.
Illustrative semantic labels are explicitly separated from learned embedding
dimensions.

## Verification

- Article schema validation: passed for all authored articles.
- Cross-database and media links: passed.
- Backend tests: 60 passed.
- Frontend tests: 30 passed.
- Production frontend build: passed.
- Curriculum API audit: all 72 selected questions have one complete deep lesson.
- Browser audit: 14 representative routes operated successfully; no browser
  errors and no page-width overflow at 390 px.
- Broadcasting audit: trace, input editing, reset, play/pause, 2D/3D views,
  shape failure and repair, pair subtraction, checks, solution reveal, mobile,
  and reduced motion passed.

## Source observations retained in the lessons

- Question 066's supplied function accepts `x0` but does not use it.
- Question 137 is titled as ISTA, while its supplied code performs an L1
  subgradient update rather than the proximal soft-thresholding step normally
  associated with ISTA.
- Question 241 needs careful interpretation of the BPTT scaling convention.

The site explains these facts instead of silently changing the supplied
solutions.

## Limits

The generated diagrams are instructional models, not proof that every external
implementation is correct. CUDA and Triton lessons trace their source but do
not run GPU kernels in the browser. Question 030 has a specialized 3D view;
the remaining lessons use focused 2D diagrams and state animations. Actual
screen-reader testing and dark-mode screenshot review remain future work.
