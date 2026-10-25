# Tasks

## 1. Baseline and tokens

- [x] 1.1 Capture "before" screenshots of index, question, article (a real deep lesson) at 360, 768 and 1280 px in light and dark using `playwright-core`, and verify the images exist in the scratchpad for later comparison
- [x] 1.2 Run the current suites (`npm test`, `app/e2e` `questions`, `articles`, `python`, and `pytest` in `app/backend`) and record which pass, so later failures can be attributed
- [x] 1.3 Add `tokens.css` (colour, type scale, space, radius, shadow, motion) for light, dark and `[data-theme]` overrides, with the current variable names kept as aliases; verify the site still renders unchanged in both themes
- [x] 1.4 Add a contrast-check script for every text/background token pair in both themes and verify it reports AA for all pairs
- [x] 1.5 Grep live files (`styles.css`, `Inline.jsx`, `Check.jsx`, `Playground.jsx`, `CodeView.jsx`, `Walkthrough.jsx`) for hex, `rgb` and `hsl` literals, replace them with tokens, and verify the grep finds none outside `tokens.css`
- [x] 1.6 Remove `.article .demo*` and `.animated-frame*` rules after confirming by grep that no live component uses them; verify `npm run build` succeeds and the article page looks unchanged

## 2. Progress store

- [x] 2.1 Implement `progress.js` (versioned record, field-level validation, debounced writes, `storage` sync, in-memory fallback with `persistent` flag) and verify with `node --test` using a fake `localStorage`
- [x] 2.2 Cover corrupt JSON, wrong version, wrong field types, storage that throws on read and on write, the 10-item recent cap, and done/bookmark toggling in unit tests; verify all pass
- [x] 2.3 Implement export (JSON download), import with validation, and reset; add a unit test for the export, reset, import round trip and verify it restores done and bookmarks
- [x] 2.4 Add a `useProgress` hook over `useSyncExternalStore` and verify two components stay in sync when one toggles done

## 3. App shell

- [x] 3.1 Add the inline theme script to `index.html` and a theme switch (light, dark, system) that persists with try/catch; verify there is no flash on reload and the choice survives it
- [x] 3.2 Build `AppShell` with skip link, header (site name, search entry, progress link, theme switch) and the storage-blocked notice; verify the header appears on index, question and article pages
- [x] 3.3 Add `#/progress` to `parseRoute` and verify old routes (`#/`, `#/question/N`, `#/article/ID`, `#/lesson-db/ID`) still open
- [x] 3.4 Add a global-shortcut hook that ignores inputs, textareas, selects and contenteditable, plus the `?` help overlay closing on `Escape`; verify typing `/` and `?` in the search box enters the character
- [x] 3.5 Add skeleton placeholders, a shared empty state and an `ErrorNotice` with retry (key bump in `useFetch`); verify with a stubbed failing request that retry repeats the call
- [x] 3.6 Honour `prefers-reduced-motion` for skeleton shimmer and transitions, and add print and 360 px base rules; verify with emulated reduced motion and a 360 px viewport that nothing animates and nothing scrolls sideways
- [x] 3.7 Start `app/e2e/ui.mjs` with shell checks (theme switch and persistence, header on every page, shortcuts not hijacking typing, old routes, 360 px no horizontal scroll) and verify it passes; update selectors in `questions.mjs` and `articles.mjs` where markup changed and verify they pass
- [x] 3.8 Take screenshots of the shell and index at 360 and 1280 px in both themes, show them to the learner, and record their direction on accent and type before continuing

## 4. Question browsing

- [x] 4.1 Add a shared full-list loader cached in memory, used by the index and the progress view; verify only one `limit=1100` request is made when moving between them
- [x] 4.2 Build the dashboard summary (done of total, lessons available, resume or start action) and verify first-visit and returning-learner states in the browser
- [x] 4.3 Build chapter navigation (side list on wide screens, select or sheet on narrow screens) with per-chapter done count and bar; verify choosing C05 filters to C05 and the count updates after marking a question done
- [x] 4.4 Redesign result cards with done and bookmark state shown by icon and text, keep match highlighting, and verify state does not rely on colour alone
- [x] 4.5 Add quick filters (not done, done, bookmarked, deep lessons) with the empty-bookmark message; verify each filter against known progress
- [x] 4.6 Keep `/`, arrow-key, `Escape`, operator, typo-correction, sort and "show more" behaviour; verify the existing `questions.mjs` search checks pass unchanged
- [x] 4.7 Keep the query, filters and scroll position on return from a question; verify by opening a result in the browser and going back
- [x] 4.8 Add index checks to `ui.mjs` (summary, chapter progress, quick filters, scroll restore) and verify they pass; document the new index behaviour in `app/README.md`

## 5. Question study page

- [x] 5.1 Redesign the question header with done and bookmark controls and badges; verify marking done on the page matches the card on the index
- [x] 5.2 Present the task with equations at a reading width and keep display equations scrolling inside their container; verify on a question with a long equation at 360 px
- [x] 5.3 Restyle the solution reveal (clear control text, loading and failure with retry, highlighted code) and keep the rule that the solution is requested only after the click; verify the network log shows no solution request before the click
- [x] 5.4 Show related lessons first as cards, then readings with publisher, access note and reason; verify on a question with a deep lesson and on one without
- [x] 5.5 Add previous/next with chapter position and left/right shortcuts; verify the first and last questions disable the missing direction and that arrows do nothing inside a field
- [x] 5.6 Record the question in recent items and verify the recent list caps at 10 and orders newest first
- [x] 5.7 Add question-page checks to `ui.mjs`, confirm `questions.mjs` (all 600 prompts and solutions compared with `workbook.json`) still passes, and verify both pass

## 6. Lesson reading

- [x] 6.1 Add stable slug ids to H2 and H3 headings and a linked table of contents; verify choosing an entry scrolls to the section and moves focus to the heading
- [x] 6.2 Add current-section marking with `IntersectionObserver` and the sticky side layout on wide screens; verify on the longest and the shortest real lessons
- [x] 6.3 Add the narrow-screen table-of-contents disclosure; verify it does not cover text when closed at 360 px
- [x] 6.4 Add the reading-progress bar, saved reading position and a continue action; verify by scrolling halfway, reloading and using Continue
- [x] 6.5 Redesign the article header (kind, review state, linked questions, reading time computed at about 200 words per minute) and the prerequisite and "Used by" sections; verify on a lesson with prerequisites
- [x] 6.6 Restyle equation, callout, table, gallery, check, walkthrough and playground blocks with tokens, and give each callout kind a label and marker beyond colour; verify with dark-theme screenshots of a lesson that contains all of them
- [x] 6.7 Add print styles; verify the print preview of a lesson hides the header, table of contents and controls
- [x] 6.8 Confirm checks, walkthroughs and playgrounds behave as before (run, stop, reset, timeout, error) and verify `articles.mjs`, `curriculum-lessons.mjs` and `python.mjs` pass; add article checks to `ui.mjs`

## 7. Progress view

- [x] 7.1 Build `#/progress` with overall count, per-chapter bars, bookmarks and recent items, each linking to its page; verify against a known set of done questions
- [x] 7.2 Add the empty state with a link to the first question, and the export, import and confirmed reset controls; verify the round trip in the browser and that Reset deletes nothing before confirmation
- [x] 7.3 Verify in the browser network log that marking done, bookmarking and reading send no progress data to any host
- [x] 7.4 Verify with storage blocked (browser context with storage disabled) and with a corrupted stored record that the site works and shows the notice; add both cases to `ui.mjs`

## 8. Integration and handoff

- [x] 8.1 Run every suite (`npm test`, `pytest`, `ui`, `questions`, `articles`, `curriculum-lessons`, `python`) and verify all pass; report any that fail with their output
- [x] 8.2 Take "after" screenshots of index, question, article and progress at 360, 768 and 1280 px in both themes, compare them with the 1.1 set, and review for overflow, clipping and unreadable text
- [x] 8.3 Run `npm run build` and the contrast script and verify there are no errors and every pair passes AA
- [x] 8.4 Update `app/README.md` (themes, progress storage and export, shortcuts, how to run `ui.mjs`) and `STATUS.MD` (new task row with measured results and anything not measured), and verify the documented commands run as written
