# Tasks

## 1. Appendix chapter filter

- [x] 1.1 Confirm `search.py` accepts `A01`–`A18` and bare numbers still map to `C`; verify `test_appendix_chapters_filter` and the full backend suite (`python -m unittest discover -s tests`) pass.
- [x] 1.2 Add a browser check that choosing A01 and A18 in the chapter list shows 52 and 50 questions; verify it passes against the static build.

## 2. Static data export

- [x] 2.1 Make `export_static.py` write `embeddings.bin` and check it; verify its size is 1,100 × 384 × 4 bytes and its row order equals the search index order.
- [x] 2.2 Add a backend test that the export holds 1,100 questions, 1,100 solutions, every article, and 1,100 vectors; verify it fails when a file is removed and passes on a fresh export.
- [x] 2.3 Re-run the export and verify `git status` shows only intended changes under `public/static-api/`.

## 3. Search by meaning in the browser

- [x] 3.1 Commit `src/semantic.js` and the hybrid ranking in `src/staticApi.js`; verify `npm test` still passes (125 tests).
- [x] 3.2 Add Node tests for the query parser, filters (including `chapter:a03`), abbreviation expansion, typo correction and rank fusion using small fixtures; verify they pass.
- [x] 3.3 Pin `@huggingface/transformers` to an exact version in `package.json` and commit the lockfile; verify `npm ci` succeeds on a clean clone.
- [x] 3.4 Commit the model files under `public/models/Xenova/all-MiniLM-L6-v2/`; verify the four files are tracked and the largest is under 100 MB.
- [x] 3.5 Verify in a browser that, after a first search, no request goes to a host other than the site's own (network log).

## 4. Early load and visible state

- [x] 4.1 Start loading the model when the page opens (idle), raise the wait to 60 s, and show "Loading meaning search…" in the index; verify with a throttled-network browser run that a query typed during loading ends as "keywords + meaning" without a second action.
- [x] 4.2 Show the keyword-only message when the model cannot load; verify by blocking `/models/` in the browser and reading the message.
- [x] 4.3 Update the e2e suite for these states; verify the suite passes on fixtures and on the real data.
  - Result: new `app/e2e/static.mjs` passes (10 checks); `questions`, `articles`, `ui`, `python` pass on fixtures; `ui` and `questions` pass on real data. The older `search` suite already failed 6 checks before this change (it expects 600 questions and 36 chapter headings) and is not part of this change; `demo` has no checks.

## 5. Deploy workflow

- [x] 5.1 Add `npm run setup:ort` to `.github/workflows/pages.yml`, keep `public/ort/` git-ignored, and set the static build env; verify the workflow file parses (`actionlint` or a dry read) and a clean local run of the same commands produces a `dist` that serves under `/interml/`.
- [x] 5.2 Commit and push to `main`; verify the Actions run succeeds and the deploy job prints the page URL.
  - Result: pushed `c0eb475..5036784`. The Actions log could not be read (no API login on this machine; the repository is private), so success is shown by the live site serving the model, runtime and vector files that only this build produces.
- [x] 5.3 In the repository settings, set Pages source to "GitHub Actions" (manual step); verify `https://harsha-moparthy.github.io/interml/` returns 200 and `…/models/Xenova/all-MiniLM-L6-v2/config.json` returns 200.

## 6. Parity check

- [x] 6.1 Choose at least 20 queries covering concepts, abbreviations, typos, operators, phrases and exclusions; snapshot the local service's top 10 into `app/e2e/parity_reference.json`; verify the file holds one list per query.
- [x] 6.2 Write `app/e2e/parity.mjs` that runs the queries in a browser against a given base URL, prints both lists, the per-query overlap and the mean, and exits non-zero below a mean of 7 or on any empty list; verify it fails on a deliberately broken search and passes on the real one.
- [x] 6.3 Run it against the published address; record the mean overlap, per-query overlap and the first-search time on a cold cache in `STATUS.MD`. If the mean is below 7, stop and revisit Decision 4 before continuing.
  - Result on the published address: mean overlap 7.50 of 10 over 27 queries (threshold 7), worst 3.0, no empty lists, first search 4.6 s. The threshold is met, so Decision 4 stands.

## 7. Parity of the rest of the site

- [ ] 7.1 Run the existing `ui` and `questions` browser suites against the published address; verify the failures, if any, are listed with causes (differences caused by static mode are noted, not hidden).
- [ ] 7.2 Check that the Python playground runs on a lesson page of the published site (starter code fails its checks with the expected message); verify no failed request in the log.
- [ ] 7.3 Compare per-chapter counts and the 72 lessons between `/api/chapters` locally and `static-api/chapters.json`; verify they are equal.

## 8. Documentation

- [ ] 8.1 Update the root `README.md` with the live address, how to regenerate the export, how to run the parity check, and a "Differences from the local site" section with the measured overlap, the first-visit download size and the licence note on public solutions; verify every command in it runs as written.
- [ ] 8.2 Update `STATUS.MD` with the task row and measured results, stating what was not measured (browsers other than Chromium, slow connections, screen readers).

## Workflow follow-up

- Archive the change after the published site passes the parity check.
