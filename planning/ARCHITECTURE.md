# Architecture and setup (T05)

Date: 2026-10-24. Stack: Python 3.11, FastAPI, SQLite, React with Vite. Code is under `app/`.

## Database boundaries

| Database | File | Written by | Content |
| --- | --- | --- | --- |
| Exercises | `app/data/exercises.db` | Import only. The service opens it read-only. | Chapters, subsections, 600 exercises, source hashes, search index |
| Content | `app/data/content.db` | Authoring tools and the service | Articles, prerequisites, media records, demonstrations, step help, learning checks, playgrounds, learner feedback |

Schemas: `app/backend/schema/exercises.sql` and `app/backend/schema/content.sql`.
Rebuild `exercises.db` from source at any time. Never overwrite `content.db`.
SQLite cannot enforce references between files. Content rows store `exercise_id` as a plain integer.
The import check and a `check_links` command (T06) must verify every `exercise_id`.

## Identity and order

The exercise ID (1–600) is permanent. `list_position` stores the original order.
Chapter IDs `C01`–`C36` are permanent. `chapter.position` stores display order.
The 36 workbook sections are the chapters. The 121 subsections are only grouping labels.

## Import design (built in T06)

1. Hash the four source files. Compare against `planning/source-audit.json`; stop on unexpected drift.
2. Read `ml-coding-workbook/workbook.json` for prompt, solution code, language, provenance, and URLs.
3. Read `ml-coding-catalog/catalog.json` for chapter and subsection order and for difficulty.
4. Read PDF page ranges from the audit output.
5. Build a new `exercises.db` in a temporary file. Check it. Then replace the old file.
6. Reject duplicate IDs, an empty solution, a missing PDF page range, and a title that differs from the PDF.
7. Set `license_note` on the 79 records with `provenance = 'repository'`.

Take code only from `workbook.json`. PDF text changes characters and is for page mapping only.
A re-import is safe because it touches only `exercises.db`.

## Service interface (built from T07)

The service binds to `127.0.0.1:8765`. Port 8000 is in use by another local application.
Planned read routes: `/api/health`, `/api/exercises`, `/api/exercises/{id}`, `/api/chapters`, `/api/articles/{id}`.
Authored content is written by file-based import from `content/` (version-controlled sources), not by the browser.
Media files are served from `app/media` at `/media/...`. Stored paths are relative.
The service has no route that runs learner code or reads arbitrary files.

## Browser execution boundary

Learner code runs in the browser, in a Web Worker, with Pyodide (evaluated in T11).
It has no connection to the service or the databases.

## Licensing

79 repository-derived records fall under `ml-interview/ml-coding-workbook/LICENSE-Deep-ML.txt`
(Educational Use Only; no commercial use; no model training).
Show the license note on those pages. Keep the license file with any copy of the data.
Use of this site is personal and local, which the license permits.

## Explanation components (T10)

State lives in a pure reducer (`frontend/src/demo/engine.js`). It stores inputs, step, selection, and help state only.
Frames are derived from the inputs, so stepping backward restores the earlier state exactly.
Each demonstration component supplies a spec (`sanitize`, `compute`, `defaultSelection`, `normalizeSelection`) and views.
Add a component by listing it in `app/content/components.json`, registering it in `registry.js` and `components.js`,
and adding an independent reference check. The broadcasting component (`broadcast-trace`) is the reference implementation.
`content.db` carries a schema version (`PRAGMA user_version`). `db.migrate` upgrades older files in place.

## Browser execution (T11)

Runtime: Pyodide 0.27.7 (Python 3.12) with NumPy 2.0.2 and pandas 2.2.3, self-hosted from `app/frontend/public/pyodide/`.
Required initial download: `cd app/frontend && npm install && npm run setup:pyodide`.
It copies the core from the `pyodide` npm package and downloads the packages in `app/content/pyodide-packages.json`
from the Pyodide CDN, checking each SHA-256. Measured size, uncompressed: core 13.7 MB (10.1 MB WebAssembly),
NumPy 3.1 MB, pandas 6.4 MB including python-dateutil, pytz, and six. Total 23.2 MB on disk.
Compressed transfer size was not measured. After setup the site needs no internet.
To add another package: add it to `pyodide-packages.json`, run the setup again, then list it in a playground.

Packages load per playground, not per page. A playground that lists only NumPy never downloads pandas.
The worker loads the packages and imports them before the timed run starts, with network access restored only
for that step. Importing pandas takes seconds in WebAssembly, so it must not count against the time limit.
Modules stay imported between runs. Learner variables do not: each run has a fresh namespace.
A learner who changes a library object (for example by assigning to `np`) can affect later runs until Reset or Stop.

Learner code runs in a Web Worker (`public/py-worker.js`), driven by `src/py/runner.js`.

| Control | Behaviour |
| --- | --- |
| Load | Starts on the first Run, not on page load. Packages load when a playground first needs them |
| Run | Fresh global namespace per run; output streamed; checks run after the code succeeds |
| Stop | Ends the worker and starts a new one. Works while loading or running |
| Timeout | Per playground, 1–60 s, counted from the start of the run. Ends and replaces the worker |
| Reset | Stops any run, restores the starter code, clears output and checks |
| Output | Capped at 20,000 characters per run. Code is capped at 20,000 characters |
| Errors | Traceback from the learner's code. Pyodide internal frames are removed |
| Crash | A worker that dies (for example out of memory) is reported and replaced |
| Unavailable | If the runtime files cannot load, the page says to run `npm run setup:pyodide`; Run retries |

Limits to state accurately: Pyodide is 32-bit WebAssembly, so memory cannot exceed 4 GB and the browser may allow less.
In testing, a loop that allocated 128 MiB arrays ended with NumPy's `ArrayMemoryError`, and the runtime stayed usable.
A single array above the 32-bit size limit raises `ValueError`.
Execution is single-threaded in the worker, with no `SharedArrayBuffer`, so Stop and timeout end the whole worker.

Containment: learner code has no access to the host filesystem or host commands. It sees only Pyodide's virtual
filesystem. The worker removes `fetch`, `XMLHttpRequest`, `WebSocket` and similar APIs after the runtime loads, so
`js.fetch` and `pyodide.http.pyfetch` fail. This is a precaution against accidental calls to the local service.
It is not a security boundary. The service also has no route that writes data or runs code.

PyTorch, CUDA, and Triton solutions are not run. Use `runtime: "display-only"` with a note.
