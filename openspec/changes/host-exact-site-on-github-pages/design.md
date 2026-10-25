# Design

## Context

The local site is a FastAPI service plus a React build. The service reads two SQLite files that are not in the repository: `exercises.db` is built from the private workbook, and `content.db` is built from the lesson JSON. GitHub Pages serves static files only. See proposal.md for the reason.

State after the first publish (commit `c0eb475`, live and returning 200):

- `export_static.py` writes each read-only API response as JSON under `app/frontend/public/static-api/`.
- `src/staticApi.js` answers `/api/*` from those files and runs a keyword search in the browser.
- The build is made with `VITE_STATIC=1` and `VITE_BASE=/interml/`; the workflow in `.github/workflows/pages.yml` deploys it.
- Uncommitted work: search by meaning in the browser (`src/semantic.js`, `embeddings.bin`, the model files, `setup_ort.mjs`, the hybrid ranking in `staticApi.js`). It loaded and ranked correctly in a local browser, but the workflow does not yet fetch the runtime files.

Measured on the uncommitted build, 9 queries compared with the local service: top-10 overlap 6 to 9, mean 7.5. The tenth query ("make training use less memory") fell back to keywords because the 44 MB download did not finish inside a 20 s limit.

```
 page opens
     |
     +--> idle preload: model (23 MB) + ONNX runtime (21 MB) from /interml/
     |
 learner types ---> keywords: JS port of the local scoring (instant)
     |                         \
     |                          +--> fuse (RRF, k=60) --> list --> "keywords + meaning"
     +--> meaning: embed query in browser, dot product with 1,100 stored vectors
```

## Goals / Non-Goals

**Goals:**
- A learner sees and can do the same things online as locally, apart from differences listed in the README.
- The first meaning search works without the learner doing anything special.
- Parity is measured against the local service, not assumed.
- The deploy is reproducible from a clean checkout.

**Non-Goals:**
- Position-for-position identical ranking (see Decision 4).
- Serving the databases or any Python code online.
- Sync of progress between devices.
- A custom domain.

## Decisions

1. **Static JSON export committed to the repository.** CI cannot rebuild `exercises.db` because the workbook is private, so the export is the published data. Alternative: build the export in CI. Rejected: it needs the databases and the Python stack there.

2. **Self-host the model and the runtime.** Model `Xenova/all-MiniLM-L6-v2` (8-bit, 23 MB) is committed under `public/models/`. The ONNX runtime (21 MB `jsep` build, which is the one the library requests) is copied from `node_modules` by `npm run setup:ort` in CI and local builds, and is git-ignored. Alternatives: load both from huggingface.co and jsdelivr (smaller repo, but a runtime dependency on two hosts and a breach of the single-origin requirement); commit the runtime too (21 MB more in history).

3. **Same family of model, different precision.** The stored document vectors come from the local float32 model. The browser embeds the query with the 8-bit model. Their vectors agree closely, and the backend thresholds (floor 0.25, margin 0.12) are reused unchanged. Alternative: ship the 90 MB float32 model. Rejected for repository and download size.

4. **Keywords use a JS port, not SQLite FTS5.** The port uses the same fields, weights, synonyms and prefix rules, but not the same stemmer or BM25 length normalisation, so order differs in places. Running SQLite with FTS5 in the browser would match more closely but adds a WebAssembly build and the 5 MB database. Decision: accept the port and gate it with the parity check (mean overlap at least 7 of 10). Revisit if the check cannot pass.

5. **Single-threaded runtime.** GitHub Pages cannot send the cross-origin isolation headers that threads need. Cost: slower embedding, still well under a second per query after load.

6. **Preload on idle, with a visible state.** The model starts loading when the page opens and the index shows "Loading meaning search…". A query typed during the load waits up to 60 s, then falls back to keywords and says so. The previous 20 s limit is raised. Alternative: always answer with keywords first and update later. Rejected as more state for the list to handle and confusing for ranking.

7. **Parity check from a stored reference.** A script in `app/e2e` runs a fixed list of at least 20 queries against the built site in a browser and compares the top 10 with `parity_reference.json`, a snapshot taken from the local service. A snapshot lets the check run without the service. The snapshot is regenerated deliberately when data or ranking changes. It runs locally before release; running it in CI needs a browser install and is left optional.

8. **Deploy by Pages artifact.** The workflow runs `npm ci`, `setup:pyodide`, `setup:ort`, tests, the static build, copies `app/media`, then deploys with the official Pages actions. Pages must be set to "GitHub Actions" in the repository settings once; this cannot be done from the code.

## Risks / Trade-offs

- **[Ranking differs in places]** → measured by the parity check, disclosed in the README, threshold set at 7 of 10.
- **[First visit downloads about 44 MB]** → idle preload, browser cache, loading state; disclosed.
- **[Repository grows by about 23 MB for the model, permanently in history]** → accepted; the runtime stays out of git.
- **[Public repository exposes all supplied solutions, including 79 license-tagged questions]** → the learner asked for publication; flagged here and in the README so they can decide to make the repository private (Pages then needs a paid plan) or remove those solutions.
- **[Library versions drift]** → `@huggingface/transformers` is pinned to an exact version and its runtime files come from the matching package, so the two cannot disagree.
- **[Stale export after a data change]** → the README names the command; a test asserts that the export holds 1,100 questions and 1,100 vectors.
- **[Browser memory or WebAssembly unavailable]** → the keyword fallback and its message keep search usable.

## Migration Plan

1. Finish and commit the work in progress, then push; the workflow deploys.
2. Set Pages to "GitHub Actions" if not done.
3. Open the published address, run the parity check against it, and read the result.
4. Rollback: revert the commit and push; the previous build is redeployed. The first keyword-only build remains valid.
