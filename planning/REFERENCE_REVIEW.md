# Source and reference review

Date: 2026-10-24.

## Local source checks

The catalog and both Markdown documents have 600 matching question IDs and titles.
The structured workbook has 600 records.
All 600 solution blocks in the LaTeX source match `workbook.json` after outer whitespace removal.
All standalone Python files have the same Python syntax trees as their workbook code.
Standalone files contain additional source header comments, so raw text comparisons differ.
This check does not establish behavioral correctness.

Extracted PDF text confirms the 600-question edition and the question 328 cross-attention contract.
PDF text contains typographic substitutions, including non-ASCII minus signs.
Use the verified generation source for code, then record PDF page references during T04.
Full PDF page mapping and CUDA equivalence checks remain pending.

## Linked attention examples

- https://www.deep-ml.com/problems/94 — source link for question 316.
- https://www.deep-ml.com/problems/959 — source link for question 317.

Both links were opened. The text browser returned a JavaScript-dependent page shell.
The question 316 page identifies Implement Multi-Head Attention.
The live playground behavior could not be inspected through this tool.
Do not claim that its controls or animations were tested.
Revisit accessible linked examples while authoring the relevant lessons.

## Additional interaction reference

https://artemiop.com/lab/attention-viz/

The page describes editable text, per-head heatmaps, token selection, and randomized projections.
These controls suggest useful ways to inspect attention values.
The page uses illustrative embeddings and weights. It is not evidence of trained semantic relationships.
The planned lessons must also show the weighted-value output and the final head combination.
Only page text was inspected. Interactive behavior was not executed.

## Writing reference

https://www.asd-ste100.org/about_STE.html

Use controlled wording and defined technical terms.
The documents follow STE principles. A full approved-dictionary review remains outside this planning pass.
