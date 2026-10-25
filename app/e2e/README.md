# Browser checks

Needs `playwright-core` (`npm i playwright-core`) and a Chromium binary. Edit `exe` in each script.

`questions.mjs` runs against the real data. `articles.mjs` needs demo articles so the real
`content.db` stays clean:

    D=$(mktemp -d); mkdir $D/data; cp app/data/exercises.db $D/data/
    export PLACEMENT_DATA_DIR=$D/data PLACEMENT_CONTENT_DIR=$PWD/app/e2e/fixtures/content PLACEMENT_MEDIA_DIR=$PWD/app/e2e/fixtures/media
    cd app/backend && ../.venv/bin/python -m placement.articles
    ../.venv/bin/uvicorn placement.main:app --port 8765 &   # and `npm run dev` in app/frontend
    node ../e2e/articles.mjs <screenshot-dir>

`demo.mjs` checks demonstrations, predictions, comparisons, and step help against the fixture lesson `lesson-030`.
Use the same fixture setup as `articles.mjs`. Fixture lessons are demo content and must not be imported into the real database.

`python.mjs` checks the Python playground (run, errors, output cap, isolation, timeout, Stop, Reset, memory, containment,
missing runtime, and runner behaviour with fake workers). It needs `npm run setup:pyodide` and the fixture lesson `lesson-030`.

`scene.mjs` checks the 3D player on the real lesson (needs WebGL; the scripts launch Chromium with SwiftShader). `search.mjs` checks search and the code viewer. `./run_suites.sh` runs the fixture suites in throwaway servers.

`ui.mjs` checks the interface layer: theme switch and saved choice, header and shortcuts, loading and error states, index dashboard and filters,
question page, article contents and reading progress, print, the progress page, and blocked or corrupt browser storage.
It picks its lesson and question from the API, so it runs on the real data (`node ui.mjs`, with both servers running) and on the fixture servers (`./run_suites.sh ui`).
Run the fixture suites from `app/e2e` with `NODE_PATH=$PWD/node_modules ./run_suites.sh questions articles ui python`.
`demo.mjs` and `curriculum-lessons.mjs` test demonstrations that T18 removed, and `curriculum-lessons.mjs` still expects 600 questions. They are out of date and fail.

