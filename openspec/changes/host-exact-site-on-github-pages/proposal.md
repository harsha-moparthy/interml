# Proposal

## Why

The learner wants the same website that runs locally to be available at https://harsha-moparthy.github.io/interml/. The first publish (commit `c0eb475`) shows the same pages, but it is a keyword-only copy. Search by meaning is missing, the first meaning search is too slow to be useful, and the deploy workflow cannot yet build what that search needs. GitHub Pages serves static files only, so the copy must work with no Python service.

## What Changes

- Publish a static build of the existing React site to GitHub Pages from the `interml` repository on every push to `main`.
- Answer every read-only `/api` route from JSON exported from the databases, so questions, solutions, readings, lessons, figures and the Python playground behave as they do locally.
- Run search in the browser with the same behaviour as the local service: keywords, typo correction, abbreviations, operators (`chapter:`, `id:`, `lang:`, `lesson:`), and search by meaning with the same ranking rules.
- Serve the meaning model (all-MiniLM-L6-v2, 8-bit) and the ONNX runtime from the site itself, so no request goes to another host.
- Start loading the meaning model when the page opens, and show a visible state while it loads, so the first search does not silently fall back to keywords.
- Add a parity check that compares the hosted search with the local service on a fixed query list and fails below an agreed overlap.
- Make the deploy workflow reproducible: fetch the runtime files in CI, build in static mode under `/interml/`, and keep generated data in the repository.
- Record what differs from the local site, and keep it small and measured.

Out of scope: accounts, sync of progress between devices (progress stays in the browser), a custom domain, and changes to lesson content.

## Capabilities

### New Capabilities
- `static-site-hosting`: the published copy of the site on GitHub Pages: what it must show and do without a backend, how it is built and deployed, how in-browser search matches the local service, and how parity is checked.

### Modified Capabilities
- `question-browsing`: the chapter filter must accept appendix chapters A01–A18 as well as C01–C36 (found broken in search during this work).

## Impact

- **Frontend** (`app/frontend`): `src/staticApi.js`, `src/semantic.js`, `src/lib.js`, `src/Index.jsx`, `vite.config.js`, `scripts/setup_ort.mjs`, `package.json` (adds `@huggingface/transformers`).
- **Backend** (`app/backend/placement`): `export_static.py`, `search.py` (appendix chapter filter, already changed), `tests/test_search.py`.
- **Repository**: `.github/workflows/pages.yml`; committed model files (about 23 MB) under `app/frontend/public/models/`; committed JSON under `app/frontend/public/static-api/`.
- **First visit** downloads about 44 MB when search by meaning loads (model plus runtime); later visits use the browser cache.
- **Publicity**: a public repository exposes every supplied solution, including the 79 license-tagged questions. The learner has asked for publication; the license risk is recorded in design.
