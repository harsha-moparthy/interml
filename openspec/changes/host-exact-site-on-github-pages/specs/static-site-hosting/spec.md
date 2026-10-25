# Spec Delta

## Purpose

The published copy of the site on GitHub Pages must give a learner the same pages, data and study tools as the local site, with no Python service behind it. This capability defines what the copy shows and does, how it is rebuilt, and how its parity with the local site is checked.

## ADDED Requirements

### Requirement: Published address
The site SHALL be served from `https://harsha-moparthy.github.io/interml/` and SHALL rebuild and redeploy on every push to `main` of the `interml` repository.

#### Scenario: Push redeploys
- **WHEN** a commit is pushed to `main`
- **THEN** the deploy workflow builds the site and the published address serves the new build

#### Scenario: Deep link under the sub-path
- **WHEN** a learner opens `https://harsha-moparthy.github.io/interml/#/question/601`
- **THEN** question 601 opens with its styles, figures and scripts loaded and no failed request

### Requirement: Same content as the local site
The published copy SHALL show all 1,100 questions in chapters C01–C36 and A01–A18, their ranked readings, every authored lesson with its figures, checks and code walkthroughs, and the supplied solutions behind the Show solution control.

#### Scenario: Counts match
- **WHEN** the published index loads
- **THEN** it lists 1,100 questions and the same per-chapter counts as the local site

#### Scenario: Solution hidden until requested
- **WHEN** a learner opens a question
- **THEN** the solution is not displayed until the learner chooses Show solution

### Requirement: No backend needed
The published copy SHALL work with no Python service and SHALL NOT request any resource from a host other than its own, except the links a learner chooses to follow.

#### Scenario: Offline of the service
- **WHEN** the local service is stopped and the learner uses the published address
- **THEN** browsing, search, lessons, solutions and the Python playground all work

#### Scenario: Single origin
- **WHEN** a learner searches by meaning for the first time
- **THEN** the model and its runtime are fetched from the site's own address

### Requirement: Search matches the local site
Search on the published copy SHALL support keywords, typo correction, abbreviations, the operators `chapter:`, `id:`, `lang:`, `source:`, `difficulty:`, `lesson:`, `has:` and `is:`, quoted phrases, excluded words, and search by meaning, using the same fusion rules as the local service.

#### Scenario: Operator query
- **WHEN** a learner types `chapter:a03 kv cache`
- **THEN** only questions in A03 appear and the active-filter chips show the chapter

#### Scenario: Meaning without shared words
- **WHEN** a learner types `make training use less memory`
- **THEN** the results include questions about memory-saving training whose titles do not contain those words

#### Scenario: Result mode shown
- **WHEN** meaning search has contributed to a result list
- **THEN** the page labels the list "keywords + meaning"

### Requirement: Parity with the local search
For a fixed list of at least 20 queries, the mean overlap between the top 10 results of the published copy and the local service SHALL be at least 7 of 10, and no query for which the local service returns results SHALL return an empty list on the published copy.

#### Scenario: Parity check passes
- **WHEN** the parity check runs the fixed query list against both sites
- **THEN** it reports the mean overlap and fails the build or the check if the mean is below 7 or any query is empty

#### Scenario: Parity check reports differences
- **WHEN** the check finishes
- **THEN** it prints each query with both top-10 lists so differences can be read

### Requirement: Meaning search ready early
The published copy SHALL begin loading the meaning model when the page opens and SHALL show a visible state while it loads. A search typed before the model is ready SHALL wait for it for up to 60 seconds, showing that state, before falling back to keyword results.

#### Scenario: Search while loading
- **WHEN** a learner types a query before the model has loaded
- **THEN** the page shows that meaning search is loading and then shows keyword and meaning results without a second action

#### Scenario: Model unavailable
- **WHEN** the model cannot load
- **THEN** the page shows keyword results and says that meaning search was not used

### Requirement: Browser-only progress
The published copy SHALL keep done marks, bookmarks and reading position in the learner's browser only and SHALL offer the same export, import and reset as the local site.

#### Scenario: Progress survives reload
- **WHEN** a learner marks a question done and reloads the page
- **THEN** the question is still marked done

### Requirement: Python playground works under the sub-path
The Python playgrounds SHALL run in the browser on the published copy with the same packages and limits as the local site.

#### Scenario: Run starter code
- **WHEN** a learner presses Run on a lesson playground
- **THEN** the runtime loads from the site's own address and the checks report pass or fail with their messages

### Requirement: Disclosed differences
The repository README SHALL list every known difference between the published copy and the local site, with measured figures where they exist.

#### Scenario: Differences listed
- **WHEN** a reader opens the README
- **THEN** it states the measured search overlap, the first-visit download size and any feature that is not available online
