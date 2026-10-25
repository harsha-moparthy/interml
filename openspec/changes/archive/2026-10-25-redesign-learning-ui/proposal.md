# Proposal

## Why

The site exists to help one learner study ML interview material, and the explanations are the product. The interface does not yet match that. The home and question pages use unstyled `system-ui` with no shared header. Article pages use a separate serif theme with hard-coded light colours, so dark mode is likely broken there. The article "On this page" list has no links. Nothing remembers what the learner has opened, finished or wants to revisit. The result works as a reference, but it is not pleasant or motivating to study with for long sessions.

## What Changes

- Add one design system for the whole site: colour, type, spacing, radius, elevation and motion tokens. Light and dark themes both derive from it, with a manual theme switch that defaults to the system setting.
- Add a persistent app header with the site name, a global search entry, a theme switch and a link to the learner's progress view.
- Redesign the question index as a study dashboard: a progress summary, chapter navigation with per-chapter progress, and richer result cards.
- Redesign the question page for study flow: clear sections, a guided "attempt first, then reveal" solution control, a readings section with visible source and reason, and previous/next navigation with keyboard shortcuts.
- Redesign the article page for long-form reading: a linked table of contents that tracks the current section, reading progress, comfortable measure and typography, and consistent theming for figures, callouts, checks, playgrounds and code.
- Add learner progress that lives in the browser: mark questions done, bookmark questions, remember recently viewed items, resume where the learner left off, and show progress per chapter.
- Add polished states throughout: skeleton loading, empty results, and recoverable errors.
- Keep every existing route, API call and keyboard behaviour working. No backend change.
- Respect `prefers-reduced-motion` and keep keyboard and screen-reader access at least as good as today.

## Capabilities

### New Capabilities
- `app-shell`: shared design tokens, light/dark theming, persistent header, navigation, global keyboard shortcuts, and loading/empty/error states.
- `question-browsing`: the home dashboard, search and filtering presentation, chapter navigation and result cards.
- `question-study`: the single-question page, including the solution reveal flow, readings presentation and question-to-question navigation.
- `lesson-reading`: the article and deep-lesson reading experience, including table of contents, reading progress and theming of embedded blocks.
- `learning-progress`: browser-local record of done, bookmarked, recent and resume state, and its display.

### Modified Capabilities
<!-- None. The project has no existing specs. -->

## Impact

- **Code:** `app/frontend/src` only. Mainly `styles.css`, `App.jsx`, `Index.jsx`, `Article.jsx`, `Inline.jsx` and `lib.js`. Likely new files for the shell, theme, progress store and shared UI parts. Existing lesson, demo, scene, code and playground components are restyled through tokens, not rewritten.
- **APIs and data:** none. Progress uses `localStorage` and never reaches the service or `content.db`.
- **Dependencies:** none planned. Fonts are system or self-hosted so the site still works offline.
- **Tests:** existing Node tests and browser suites in `app/e2e` (`questions`, `articles`, `demo`, `python`) must keep passing, with selectors updated only where markup changes. New browser checks cover theme, progress and the table of contents.
- **Out of scope:** new lessons or readings, backend or schema changes, accounts or cross-device sync, new lesson animations (T18 retired them), and rewriting the search ranking.
