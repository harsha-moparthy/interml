# Source fidelity audit (T04)

Date: 2026-10-24. Script: `tools/audit_source.py`. Output: `planning/source-audit.json`.
Run: `python3 -I tools/audit_source.py ml-interview planning/source-audit.json`

## Source hashes (SHA-256)

| File | Hash |
| --- | --- |
| `ml-coding-patterns-500-solutions.pdf` (304 pages) | `0010235fd7224bf1051062e9f02d6d5713eebb0fa516572b7c4b19703fd25494` |
| `ml-coding-patterns-500-solutions.tex` | `48b721cfc0abf6016f2083f120e17f0ac08d5614151a60153ec36a1ca44e5757` |
| `ml-coding-workbook/workbook.json` | `1b816010d8e2bf6832529f13de714770f1c93c76bd6a2a452971fb793718a24e` |
| `ml-coding-catalog/catalog.json` | `00e2e28c1d485b2082023b616f541769f8a0114cfb9c1a8a3903512c3fd1d920` |

The PDF in `ml-coding-workbook/` has the same hash as the top-level PDF.
`ml-interview-prep.zip` was not used. The client confirmed it can be ignored.

## Results

- The PDF contains 600 question headings, IDs 001–600, with no duplicates or extra IDs.
- All 600 PDF titles match `workbook.json` after folding PDF text substitutions.
- All 600 workbook code blocks appear in their PDF sections after the same folding.
- `catalog.json` flattens to 600 items. Number, title, URL, and chapter agree with `workbook.json`.
- The 72 titles in `LESSON_SELECTION.md` match `workbook.json`.
- The workbook has 36 sections, which are the chapters.
- Each record has a PDF page range in `source-audit.json` (`pdf_pages`: first page, last page).

## Limits

- Folding removes every letter `f` and all whitespace from both sides. A change that only adds or removes `f`, or only changes whitespace, would not be detected. Code identity comes from the TeX and workbook comparison in `REFERENCE_REVIEW.md`.
- PDF text turns the `f` of `fi` ligatures into a space and uses non-ASCII dashes and quotes. Import code from `workbook.json`, never from PDF text.
- This audit does not run solutions. CUDA equivalence is still unchecked.
- `workbook.json` records have no per-record license field. Attribution and license text still need the `LICENSE-Deep-ML.txt` file and the source URLs.

Correction during T06: the first run recorded page 305 as the last page of question 600. The PDF has 304 pages. The script now drops the empty trailing split. Hashes and match counts did not change.
