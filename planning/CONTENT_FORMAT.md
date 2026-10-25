# Article source format (T09)

Authored sources are JSON files in `app/content/articles/`. The file name equals the article ID.
Import with `python -m placement.articles` (run `... check` to validate without writing).
The import checks all files first and writes all changes in one transaction.

## Article fields

| Field | Rule |
| --- | --- |
| `id` | Lowercase slug, for example `broadcasting` |
| `kind` | `lesson` or `prerequisite` |
| `title`, `revision` | Revision is a positive integer |
| `review_state` | `draft`, `reviewed`, or `complete` |
| `exercises` | List of `{id, purpose, order}`. Purpose is `deep-lesson`, `prerequisite`, or `related`. A lesson needs one `deep-lesson` entry |
| `prerequisites` | Article IDs. No cycles |
| `sources` | List of `{title, url}` |
| `media` | List of `{path, description, alt_text, attribution}`. Paths are relative to `app/media` |
| `blocks` | Ordered content blocks |
| `demonstrations` | Interactive demonstrations (below) |
| `checks` | Predictions, new cases, and explain-back prompts (below) |
| `playgrounds` | Runnable or display-only code for an exercise (below) |

## Blocks

| Type | Fields |
| --- | --- |
| `heading` | `level` (2 or 3), `text` |
| `paragraph` | `text` |
| `list` | `items`, `ordered` |
| `equation` | `latex`, `caption` |
| `code` | `code`, `language`, `caption` |
| `image` | `path` (must appear in `media` with alt text), `caption` |
| `callout` | `kind` (`definition`, `note`, `mistake`, `boundary`, `example`), `text`, `title` |
| `table` | `header`, `rows`, `caption` |
| `demonstration` | `id` of an entry in `demonstrations`, `caption` |
| `check` | `id` of an entry in `checks` |
| `playground` | `exercise_id` of an entry in `playgrounds` |
| `compare` | `demonstration`, `changed` (the one condition that differs), `cases` (exactly two `{label, inputs}`), `takeaway` |

Inline text supports `` `code` ``, `**bold**`, `*emphasis*`, `$math$`, `[[article:id|label]]`, and `[[q:316|label]]`.

## Rules the import enforces

- Reject unknown fields, block types, prerequisites, articles, questions, and exercise IDs.
- Reject missing media files and images without alt text.
- Reject prerequisite cycles.
- Reject `complete` when any prerequisite is not `complete`.
- Reject a content change without a higher revision. Reject a lower revision.
- Never delete an article when its source file is removed.
- A lesson counts as complete on the index only when its article state is `complete`.

## Demonstrations

`{id, component, seed, input_schema: {defaults, ...}, steps: [{id, title}], expected, selection_links, help}`.
`component` must be listed in `app/content/components.json` and registered in `frontend/src/demo/registry.js`.
The component computes its frames from the inputs. Step IDs must match the frame IDs the component produces.
`expected` is a list of `{inputs, expect}` computed with an independent reference such as NumPy.
`node tests/check_content.mjs <articles-dir>` (or `npm run check:content`) compares it with the browser computation.
`help` items: `{step_id, help_type, content, prerequisite_id}`. Types: `smaller-example`, `prerequisite`, `geometry`,
`expanded-equation`, `common-mistake`, `code-connection`. The learner sees only the items for the current step.

## Checks

`{id, kind, prompt, answer_type, options, case, demonstration, answer, tolerance, explanation}`.
Kinds: `prediction`, `new-case`, `explain-back`. Answer types: `choice`, `number`, `shape`, `text`.
`explanation` has `correct`, `default` (shown for any other wrong answer), and optional `wrong` (specific text per wrong answer).
`explain-back` needs only `explanation.correct`. It shows a model explanation on request and does not grade the learner.
The answer is sent to the browser but not shown until the learner asks.

## A complete lesson also needs

A prediction check, a new-case check, a demonstration with `expected` values, authored step help,
and a `compare` block or a `mistake` or `boundary` callout.

## Playgrounds

`{exercise_id, runtime, packages, starter_code, checks, time_limit_ms, max_input_size, note}`.
The exercise must appear in the article's `exercises`. There is one playground per exercise.

- `runtime: "pyodide"` runs Python in the browser. It needs `starter_code`. `packages` must be listed in
  `app/content/pyodide-packages.json` (the setup script downloads exactly those: `numpy` and `pandas`). Default is `["numpy"]`.
  List `pandas` only for playgrounds that need it. It adds about 6 MB to the first download.
- `runtime: "display-only"` shows a label and no Run button. It needs a `note` that states why it does not run,
  for example that the solution uses PyTorch, CUDA, or Triton.
- `checks` are `{name, code}`. The check code runs after the learner's code succeeds, in the same namespace.
  An `assert` that fails marks the check as failed and shows its message.
- `time_limit_ms` is 1000–60000 (default 10000). It counts the run only, not the runtime load.
- `starter_code` must not contain the supplied solution. Solutions stay behind Show solution.

## Concept laboratory and scene demonstrations (T15)

`component: "concept-lab"` is the lesson-030-inspired interactive laboratory. It carries checked step states in `data`
(`params`, `cases`, `frames` with views of type grid, tokens, text, equation, plot, graph, or code). The visual stage appears before
the transport controls. Chapters, a scrubber, synchronized explanations, object tracing, and controlled input changes use one shared
state. One id on several cells or views selects them together. `links` mark related ids. The retired `frames` component must not be used.

`component: "scene"` is a 3D timeline. Each case holds `scene`: `duration`, `chapters` (contiguous, each with `id`, `title`, `t0`, `t1`,
`changed`, `why`, `fixed`), `objects` (types cube, text, arrow, line, box, plane; `tracks` map a property to `[time, value, ease]` keys),
`camera.keys`, named `views`, `by_sel` text for selected objects, and `alt`. The import checks structure, times, and links.
Build scenes with `tools/lessons/scene.py`. Question 030 (`tools/lessons/lessons/scenes_c02.py`) is the reference.

Authoring rule: every number in a lesson comes from code that is asserted against the supplied solution or NumPy.
