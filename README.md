# interml

Study site for 1,100 ML coding and interview questions, with ranked readings, 72 deep lessons, and a Python playground that runs in the browser.

Live site: https://harsha-moparthy.github.io/interml/

## What the online copy does

- Browse and filter all 1,100 questions by chapter (C01–C36 and A01–A18).
- Search by keywords and by meaning, with typo correction, abbreviations, and filters such as `chapter:A03`, `id:300-350`, `lang:cpp`, `lesson:complete`, quoted phrases and `-excluded` words.
- Read the 72 lessons and run the Python playgrounds (Pyodide and NumPy, in your browser).
- Show or hide each supplied solution.
- Progress and bookmarks stay in your browser only.

## Differences from the local site

Measured on 2026-10-25 against the local service, with `app/e2e/parity.mjs` (27 fixed queries):

| What | Online copy | Local site |
| --- | --- | --- |
| Pages, data, lessons, solutions, readings | Same files, byte-equal to the local API | |
| Keyword search | JavaScript port of the local scoring | SQLite FTS5 (BM25, Porter stemming) |
| Search by meaning | all-MiniLM-L6-v2, 8-bit, in your browser | Same model family, 32-bit, in the service |
| Top-10 overlap with the local site | Mean 7.5 of 10; worst query 3 of 10; no empty lists | |
| First visit | Downloads about 44 MB for search by meaning (model 23 MB, runtime 21 MB), then the browser caches it | No download |
| First meaning search | 4.6 s on a fast connection; 38.9 s at 10 Mbit/s (the list shows "Searching…", then "keywords + meaning") | Instant |

The two sites rank the same questions in a slightly different order for some queries.
For example, `handle imbalanced classes` and `train on multiple gpus` share 3 and 4 of their top 10.
If the model cannot load, the online copy says so and shows keyword results.

Not measured: browsers other than Chromium, phones, mobile data, and screen readers.

## About the solutions

Maintained, commented implementations for C01–C36 live in `app/content/solutions/`.
The importer applies them over the original workbook, preserving source attribution.
See [solution editing and verification](app/content/solutions/README.md).

The site contains every supplied solution as a JSON file (`static-api/exercises/<id>/solution.json`).
A GitHub Pages site is public even when the repository is private.
79 questions carry a license note in the data. Check those terms before you share the address.

## How it is published

`.github/workflows/pages.yml` runs on every push to `main`. It installs the packages, fetches the Python and ONNX runtimes, runs the tests, builds the site in static mode under `/interml/`, and deploys it to GitHub Pages.
In static mode the site reads JSON files from `app/frontend/public/static-api/` instead of calling the Python service.
In the repository settings, Pages must use the source "GitHub Actions" (set once).

Regenerate the JSON after any change to the databases:

    cd app/backend
    ../.venv/bin/python -m placement.export_static

Build the static site by hand:

    cd app/frontend
    npm ci --ignore-scripts
    npm run setup:pyodide && npm run setup:ort
    VITE_STATIC=1 VITE_BASE=/interml/ npm run build
    cp -r ../media dist/media

## Check that the online search matches the local one

Take a new reference from the local service only when the data or the ranking changes on purpose:

    cd app/e2e
    node parity_snapshot.mjs          # needs the local service on 127.0.0.1:8765

Run the check against any copy of the site (it exits 1 when the mean overlap is below 7 of 10 or a query shows nothing):

    SITE=https://harsha-moparthy.github.io/interml/ node parity.mjs

The browser suites also run against the online copy:

    SITE=https://harsha-moparthy.github.io/interml/ node ui.mjs
    SITE=https://harsha-moparthy.github.io/interml/ node questions.mjs
    SITE=https://harsha-moparthy.github.io/interml/ node static.mjs
    SITE=https://harsha-moparthy.github.io/interml/ node appendix.mjs

## Run it locally (full version)

See `app/README.md`. In short: Python 3.11 or later, Node 20 or later, then `app/serve.sh start`.
The databases in `app/data/` and the source workbook in `ml-interview/` are not in this repository.
The static JSON is the published copy of the data.

## Layout

- `app/backend` FastAPI service, importer, search, export, tests
- `app/frontend` React + Vite site
- `app/content` authored lessons (JSON)
- `app/media` lesson figures
- `app/e2e` browser checks
- `planning`, `openspec`, `STATUS.MD`, `TASKS.md` planning and status notes
