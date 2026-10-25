# learning-progress Specification

## Purpose

Defines the learner's browser-local study record: done and bookmarked questions, recent items, resume points and their display, kept private to the browser.

## Requirements

### Requirement: Mark questions done
The learner SHALL be able to mark a question done and undo it from the question page and from result cards. The state SHALL persist across reloads in the same browser.

#### Scenario: Mark and reload
- **WHEN** the learner marks question 300 done and reloads
- **THEN** question 300 still shows as done

#### Scenario: Undo
- **WHEN** the learner marks a done question as not done
- **THEN** it no longer counts toward any progress total

### Requirement: Bookmarks
The learner SHALL be able to bookmark and unbookmark any question. Bookmarks SHALL persist across reloads and be available as an index filter.

#### Scenario: Bookmark appears in the filter
- **WHEN** the learner bookmarks a question and chooses the bookmarked filter
- **THEN** that question is listed

### Requirement: Recent and resume
The system SHALL record the most recently opened questions and articles, up to 10 items, and the last item opened. The index SHALL offer both.

#### Scenario: Recent list
- **WHEN** the learner opens twelve different questions
- **THEN** the recent list shows the latest ten, newest first

### Requirement: Progress view
The system SHALL provide a progress view with overall done count, per-chapter done counts, bookmarks and recent items. Every entry SHALL link to its page.

#### Scenario: Open progress
- **WHEN** the learner opens the progress view
- **THEN** totals, per-chapter bars, bookmarks and recent items are shown, each linking to its page

#### Scenario: Empty progress
- **WHEN** the learner has no history
- **THEN** the view explains how to start and links to the first question

### Requirement: Local and private
Progress data SHALL stay in the browser. The system SHALL NOT send it to the service or any other host.

#### Scenario: No progress traffic
- **WHEN** the learner marks a question done
- **THEN** no network request carries the progress data

### Requirement: Resilient storage
The system SHALL work when browser storage is blocked, full or holds invalid data. It SHALL tell the learner, without an error page, that progress will not be saved. Invalid stored data SHALL be ignored, not crash the page.

#### Scenario: Storage blocked
- **WHEN** the browser blocks local storage
- **THEN** the site works, and a short notice says progress is not being saved

#### Scenario: Corrupt record
- **WHEN** the stored progress cannot be parsed
- **THEN** the site starts with empty progress and keeps working

### Requirement: Export and reset
The learner SHALL be able to export their progress as a file, import a previously exported file, and reset all progress after a confirmation.

#### Scenario: Round trip
- **WHEN** the learner exports progress, resets it, and imports the file
- **THEN** done and bookmarked questions return as before

#### Scenario: Reset needs confirmation
- **WHEN** the learner chooses Reset
- **THEN** nothing is deleted until the learner confirms
