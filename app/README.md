# ML coding learning site

Local site for one learner. Planning documents are in `../planning/`.

## Requirements

Python 3.11 or later. Node 20 or later (for the frontend, from T07).

## First-time setup

    cd app
    python3 -m venv .venv
    .venv/bin/pip install -r backend/requirements.txt

## Quick start

    app/serve.sh start     # service on 127.0.0.1:8765 and web page on 127.0.0.1:5173, both in the background
    app/serve.sh status
    app/serve.sh stop
    app/serve.sh logs      # follow both logs (written to app/.run/)

Open http://127.0.0.1:5173/ after `start` prints "Ready". The steps below do the same by hand.

## Start the service

    cd app/backend
    ../.venv/bin/uvicorn placement.main:app --host 127.0.0.1 --port 8765

Check: `curl http://127.0.0.1:8765/api/health` returns `{"status":"ok",...}`.
Build the exercise database before the first start:

    cd app/backend
    ../.venv/bin/python -m placement.importer      # add --allow-drift after an intended source change
    ../.venv/bin/python -m placement.check_links   # cross-database and media checks
    ../.venv/bin/python -m unittest discover -s tests

Import authored articles (after adding files to `app/content/articles/`):

    ../.venv/bin/python -m placement.articles

The first start creates `app/data/content.db` from its schema.
`app/data/exercises.db` appears after the import (T06).

## Start the frontend

    cd app/frontend
    npm install
    npm run setup:pyodide   # one-time download of the Python runtime (about 23 MB uncompressed with NumPy and pandas)
    npm run dev

## Use the site

The index shows your progress, a chapter list with a done count for each chapter, and the question list.
Choose a chapter in the list (a menu on narrow screens). Use the buttons to show all, not done, done, or bookmarked questions.
Mark a question done with the circle, and bookmark it with the star. Both work on the index and on the question page.
The index keeps your search, filters, and scroll position when you open a question and come back.
Progress stays in this browser only. It is never sent to the service. If the browser blocks storage, the site tells you that marks last only for this visit.
Open **Progress** in the header for totals, a bar for each chapter, your bookmarks, and recently opened questions and lessons.
On that page, **Export progress** saves a JSON file, **Import progress** restores one, and **Reset all progress** deletes everything after you confirm.
Press `/` to search, `?` for the list of shortcuts, and the left and right arrow keys on a question page to move between questions.
The theme switch in the header sets light, dark, or the system setting. A lesson page has a linked contents list, a reading progress bar, and a Continue button for a lesson you left part way. Printing a lesson gives a plain light page.

## Backup

Copy `app/data/content.db`, `app/content/`, and `app/media/`. Rebuild `exercises.db` with the import.

The frontend proxies `/api` to the service on port 8765. Open http://127.0.0.1:5173/ . Start the service first.

## Tests

    cd app/backend && ../.venv/bin/python -m unittest discover -s tests
    cd app/frontend && npm test && npm run check:content    # includes colour contrast (WCAG AA, both themes) and the progress store
    cd app/e2e && npm i && NODE_PATH=$PWD/node_modules ./run_suites.sh questions articles ui python

Browser checks are in `app/e2e/` (see its README). Regenerate NumPy reference values with `app/.venv/bin/python tools/gen_broadcast_expected.py > app/frontend/tests/broadcast_expected.json` (needs `pip install -r backend/requirements-dev.txt`).

## Build lessons

From the project root, rebuild the complete 72-lesson curriculum with:

    app/.venv-ref/bin/python tools/lessons/build_all.py expanded_01_12 expanded_13_36 c16
    cd app/backend && ../.venv/bin/python -m placement.articles          # validates and imports into content.db

`app/.venv-ref` has NumPy, pandas, SciPy, scikit-learn, matplotlib, and PyTorch for checking numbers against the supplied solutions.
Semantic search needs `python -m placement.semantic download` once (about 90 MB), then the exercise import.

## Redesigned broadcasting article

Open http://127.0.0.1:5173/#/article/lesson-q030-pairwise-differences after starting the existing services.
The index also links to this article. It replaces the previous question 030 presentation.
Run `node app/e2e/broadcast-redesign.mjs` from the project root for its browser checks.
The previous source is preserved in `backups/before-broadcast-redesign.tar.gz`.

## Complete curriculum

The index contains all 600 questions and preserves IDs 001–600. The 72 selected
questions have complete deep lessons, with two lessons in each of the 36 chapters.
Each lesson links back to its question and original solution and includes a visual,
an interactive step trace, equations or invariants, boundary cases, and checks.

Representative routes:

- Broadcasting: http://127.0.0.1:5173/#/article/lesson-q030-pairwise-differences
- Multi-head attention: http://127.0.0.1:5173/#/article/lesson-q316-multi-head-attention
- Cross-attention: http://127.0.0.1:5173/#/article/lesson-q328-cross-attention
- RoPE: http://127.0.0.1:5173/#/article/lesson-q331-rotary-positional-embeddings-rope
- KV cache: http://127.0.0.1:5173/#/article/lesson-q391-kv-cache-for-efficient-autoregressive-attention

Run the curriculum browser audit from the project root:

    node app/e2e/curriculum-lessons.mjs
