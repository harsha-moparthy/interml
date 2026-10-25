# Design

## Context

The frontend is React 18 + Vite with hash routing in `App.jsx`. It has no router library, no UI library and about 3,100 lines of source. Styling is one 264-line `styles.css` plus three component CSS files. Findings that shape the approach:

- `styles.css` defines colour variables for light and dark, but its article rules (`.article>h1`, `.article .demo`) hard-code `#14223e`, `#fff` and similar. Dark mode is therefore inconsistent on article pages. These rules target `.demo` markup that `Article.jsx` no longer renders (T18 removed demonstration blocks), so part of this CSS is dead.
- From `main.jsx`, only `demo/Check.jsx` is reachable among the `demo/`, `scene/` and `lessons/` folders. `conceptLab.css`, `conceptStories.css`, `broadcastArticle.css` and `sceneRenderer.js` hold about 86 hard-coded colours but are not loaded by the live app. They are covered by older browser suites (`demo.mjs`, `broadcast-redesign.mjs`).
- The live article blocks are heading, paragraph, list, equation, code, image, callout, table, check, walkthrough, gallery and playground. Hero is rendered in the header.
- The index already loads the full list with `limit=1100` when no query is set. Each item carries `chapter_id`. `/api/chapters` returns counts.
- The `filters` state lives in `App`, so it already survives navigation. Scroll position does not.
- The learner is one person on one machine. Service runs on 127.0.0.1:8765 and Vite on 5173; port 8000 belongs to another app and must not be used.

See proposal.md for motivation and the specs for behaviour.

**Assumptions recorded because the request was open-ended** ("very nice and enriching"):

- "Enriching" means a calmer, more motivating study flow: orientation, progress, and fewer dead ends. It does not mean new lesson content or animation.
- The change is frontend only. Progress stays in the browser.
- Direction is a calm editorial look: generous whitespace, a restrained accent, a serif for reading headings, and a clean sans for interface text.

## Goals / Non-Goals

**Goals:**
- Fix theming at the root by routing every colour through tokens, so light and dark cannot drift.
- Give the learner orientation (where am I, what is next) and momentum (done, bookmarks, resume).
- Improve long-form reading without touching lesson content or the API.
- Keep the diff reviewable: shell and tokens first, then one page at a time.

**Non-Goals:**
- Restyling code that the live app does not load (`scene/`, `lessons/`, most of `demo/`). Removing it is a separate cleanup.
- A router library, a CSS framework, a component library, or a state library.
- Server-side progress, accounts or sync.
- New animation. T18 retired it, and reduced motion must stay respected.

## Decisions

### 1. Token layer with aliases, not a rewrite of every rule
Add `tokens.css` with semantic custom properties for surface, text, border, accent, status, radius, shadow, space, type scale and motion. Keep today's names (`--bg`, `--fg`, `--muted`, `--line`, `--accent`, `--accent-bg`, `--warn-bg`, `--err`, `--code-*`, `--tok-*`, `--s0`–`--s5`) as aliases of the new tokens.
**Why:** `Check`, `Playground`, `CodeView`, `Walkthrough` and the frame styles already read these names. Aliases restyle them for free and keep the diff small.
**Alternatives:** Rename everything to a new scheme (large, risky diff); adopt Tailwind or shadcn (new build dependency, and the codebase has no utility-class habit).

### 2. Theme by `data-theme` on `<html>`, set before first paint
Three values: `light`, `dark`, `system`. A tiny inline script in `index.html` reads the saved choice and sets the attribute before React loads, so there is no flash. Tokens are defined under `:root`, `@media (prefers-color-scheme: dark) :root:not([data-theme=light])`, and `[data-theme=dark]`. All storage access is wrapped in try/catch.
**Why:** It matches how `color-scheme: light dark` already works, needs no JavaScript for the default case, and keeps CodeMirror and KaTeX on CSS variables.
**Alternative:** Toggle a class from React after mount (visible flash, and fails with JavaScript errors).

### 3. Fonts: system stacks only
Interface uses a system sans stack. Reading headings use a serif stack (`Charter, "Iowan Old Style", Georgia, serif`). Code keeps the current monospace stack.
**Why:** The site must work offline on a local machine, and the project has no font-hosting step. A self-hosted web font can be added later without changing tokens.
**Alternative:** Google Fonts (network dependency, privacy cost for a local-only tool).

### 4. Progress store: one small module on `useSyncExternalStore`
A `progress.js` module owns a single record in `localStorage` under a versioned key (for example `placement.progress.v1`):

```
{ v:1,
  done:      { "<id>": <ms> },
  bookmarks: { "<id>": <ms> },
  recent:    [ { kind:"question"|"article", id, t } ]   // max 10
  reading:   { "<articleId>": { y:<0..1>, t } } }
```

Reads are validated field by field and invalid parts are dropped. Writes are debounced for scroll position. The `storage` event keeps tabs in sync. If storage throws, the module keeps state in memory and exposes `persistent: false` so the shell can show the notice. Export is a JSON download; import validates before replacing; reset requires confirmation.
**Why:** The data is tiny, single-user and private. `useSyncExternalStore` avoids a context tree and a state library, and makes the module testable with `node --test` and a fake `localStorage`.
**Alternatives:** IndexedDB (overkill); React context with `useReducer` (more wiring, harder to test without a DOM); server storage (rejected: needs backend and schema changes, out of scope).

### 5. Per-chapter progress without a new endpoint
Progress stores only ids. Chapter totals come from `/api/chapters`. Per-chapter done counts come from the full list the index already fetches (each item has `chapter_id`). A small cached loader shares that list between the index and the progress view.
**Why:** No backend change. The list is already loaded on the home page.
**Alternative:** Store `chapter_id` with each done entry (duplicates data and drifts if chapters are renumbered). Whether the list endpoint is the right source for the progress view is to be checked in tasks.

### 6. App shell and routing
Keep hash routing. Add `#/progress` to `parseRoute`. Wrap routes in an `AppShell` that renders the header, a skip link, the shortcut help overlay and the storage notice. Index scroll position is saved in memory on route change and restored on return.
**Why:** Existing links and tests keep working. A router library would add risk for one new route.

### 7. Article navigation
Give each H2 and H3 a stable id (slug of its text, de-duplicated). Render the table of contents from those headings, with links. Mark the current section from heading positions on scroll (throttled with `requestAnimationFrame`): the last section whose heading has passed a reading line near the top, and at the bottom of the page the last visible heading. An `IntersectionObserver` was tried first and dropped, because short final sections could never reach its zone. On wide screens the table sits in a sticky side column; on narrow screens it opens from a button in a disclosure panel. A thin reading-progress bar uses scroll position. Position is saved per article with the debounced store write. Reading time is computed from block text at about 200 words per minute.
**Why:** The table is currently an unlinked ordered list. These are standard patterns and need no dependency.
**Alternative:** Always-visible table above the text (current layout; pushes content down on every article).

### 8. Chapter navigation on the index
A side list of chapters on wide screens, a select or sheet on narrow screens. It sets the existing `chapter` filter, so search and filters keep their present behaviour.
**Why:** Reuses current filter state and API; only the presentation changes.

### 9. Keep the existing search and keyboard logic
The `/`, arrow-key and `Escape` handlers in `Index.jsx` are kept. Global shortcuts move into one hook that ignores events from inputs, textareas, selects and contenteditable elements. Left and right arrows on the question page use that hook.
**Why:** Avoids regressions in behaviour the learner already relies on, and gives one place to enforce the "never hijack typing" rule.

### 10. Dead styles
Remove CSS rules that target markup `Article.jsx` no longer renders (`.article .demo*`, `.animated-frame*`) only after a grep confirms no live component uses them. Leave `scene/`, `lessons/` and unreachable `demo/` files untouched, and note them for a later cleanup.
**Why:** Keeps this change about the live experience and avoids breaking old test suites that still load those files.

### 11. Skeletons and states in CSS
Loading placeholders are CSS-only blocks with a shimmer disabled under `prefers-reduced-motion`. Errors use one shared `ErrorNotice` with a retry that bumps a key in `useFetch`.
**Why:** No dependency, and one pattern for every page.

### 12. Print styles
A `@media print` block hides the header, table of contents and interactive controls and resets to light tokens.

### 13. Verification
- Unit tests (`node --test`) for the progress store: validation, corrupt data, storage failure, export/import, recent cap.
- A new browser suite `app/e2e/ui.mjs`, run with `playwright-core` and the Chromium build under `/mnt/nas/harshamttl/.cache/ms-playwright/chromium-1148` (the Playwright MCP cannot find Chrome). It covers theme switch and persistence, header on every page, done/bookmark/resume, chapter progress, table-of-contents links and current-section marking, shortcuts not firing in inputs, 360 px no-horizontal-scroll, and a failed-request retry.
- A contrast check script for token pairs in both themes (AA).
- Screenshots of index, question, article and progress at 360, 768 and 1280 px in both themes, reviewed before the work is called done.
- All existing Node tests and browser suites (`questions`, `articles`, `python`) must pass, with selectors updated only where markup changed. Real lessons and questions are used for checks, not fixtures, and unmeasured things are reported as unmeasured.

## Risks / Trade-offs

- **Old browser suites depend on markup** → Update selectors in the same task that changes the markup, and run the suites after each page.
- **Hard-coded colours hidden in live blocks** (equation, table, callout, KaTeX, gallery) → A colour audit task greps live files for literals and tokenizes them; the dark-theme screenshot pass catches the rest.
- **Taste is subjective** → The look is a stated assumption. Screenshots go to the learner after the shell and index, before the other pages, so direction can change early.
- **Scroll-spy and sticky columns misbehave on long or short articles** → Test on the longest and shortest real lessons; fall back to a static table when fewer than three sections exist (current rule is more than two headings).
- **localStorage can be cleared or blocked** → In-memory fallback with a visible notice; export gives the learner a manual backup.
- **Restoring index scroll on a 1,100-row list can jump if the list renders late** → Restore after the list has rendered; keep the "show more" behaviour for searches.
- **Dead code left in place** → Accepted; named as a follow-up so it does not grow this change.

## Open Questions

- Accent colour and exact type scale. These change appearance only, not behaviour, so they can be settled during the first screenshot review.
- Whether the progress view should live at `#/progress` or on the index as a panel. The route is the default here; the choice does not change the specs.
