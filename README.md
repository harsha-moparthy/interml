# interml

Study site for 1,100 ML coding and interview questions, with ranked readings, 72 deep lessons, and a Python playground that runs in the browser.

Live site: https://harsha-moparthy.github.io/interml/

## What the online copy does

- Browse and filter all 1,100 questions by chapter (C01–C36 and A01–A18).
- Search by keywords, with typo correction, abbreviations, and filters such as `chapter:A03`, `id:300-350`, `lang:cpp`.
- Read the lessons and run the Python playgrounds (Pyodide and NumPy, in your browser).
- Show or hide each supplied solution.
- Progress and bookmarks stay in your browser only.

Search by meaning needs a local embedding model, so it works only when you run the site locally.

## How it is published

`.github/workflows/pages.yml` builds `app/frontend` with `VITE_STATIC=1` and deploys it to GitHub Pages on every push to `main`.
In this mode the site reads JSON files from `app/frontend/public/static-api/` instead of calling the Python service.
Regenerate those files after any change to the databases:

    cd app/backend
    ../.venv/bin/python -m placement.export_static

## Run it locally (full version, with search by meaning)

See `app/README.md`. In short: Python 3.11+, Node 20+, then `app/serve.sh start`.
The databases in `app/data/` and the source workbook in `ml-interview/` are not in this repository.
The static JSON is the published copy of the data.

## Layout

- `app/backend` FastAPI service, importer, search, tests
- `app/frontend` React + Vite site
- `app/content` authored lessons (JSON)
- `app/media` lesson figures
- `planning`, `openspec`, `STATUS.MD`, `TASKS.md` planning and status notes
