# question-browsing Specification

## Purpose

Defines how the learner finds and chooses questions: the home dashboard, search and filter controls, chapter navigation and result cards.

## Requirements

### Requirement: Study dashboard summary
The index SHALL open with a summary of the learner's position. It SHALL show questions done out of the total, deep lessons available, and a resume action when the learner has history.

#### Scenario: Returning learner
- **WHEN** the learner has viewed questions before
- **THEN** the summary shows a resume action that opens the most recently viewed question or article

#### Scenario: First visit
- **WHEN** the learner has no history
- **THEN** the summary shows the totals and a start action that opens the first question

### Requirement: Chapter navigation
The index SHALL show all chapters with their title, question count and the learner's done count. Choosing a chapter SHALL filter the list to it. The active chapter SHALL be clearly marked.

#### Scenario: Pick a chapter
- **WHEN** the learner chooses chapter C05
- **THEN** only C05 questions are listed and C05 is marked active

#### Scenario: Chapter progress
- **WHEN** the learner has marked 3 of 18 questions done in a chapter
- **THEN** that chapter shows 3 of 18 and a matching progress bar

### Requirement: Search and filters preserved
The index SHALL keep the existing search behaviour: keywords, meaning, typo correction, abbreviations, operators such as `chapter:` and `id:`, sort order and "show more". The existing filter controls SHALL remain available.

#### Scenario: Operator query
- **WHEN** the learner types `chapter:attention softmax`
- **THEN** results and active-filter chips match the behaviour before this change

#### Scenario: Clear filters
- **WHEN** any filter or query is set and the learner chooses Clear
- **THEN** the query and filters reset and the full list returns

### Requirement: Result cards
Each result SHALL show the question number, title, chapter, language, difficulty when known, a deep-lesson badge when a lesson exists, and the learner's state (done or bookmarked). Search matches SHALL stay highlighted in the title and snippet.

#### Scenario: Done and bookmarked state
- **WHEN** a question is done and bookmarked
- **THEN** its card shows both states in a way that does not rely on colour alone

#### Scenario: Highlighting
- **WHEN** the learner searches for "softmax"
- **THEN** matching words in the title and snippet are highlighted

### Requirement: Quick filters by learner state
The index SHALL offer filters for not done, done and bookmarked questions, and a filter for deep lessons.

#### Scenario: Only bookmarked
- **WHEN** the learner chooses the bookmarked filter
- **THEN** only bookmarked questions are listed

#### Scenario: Nothing bookmarked
- **WHEN** the learner chooses the bookmarked filter and has no bookmarks
- **THEN** the list explains how to bookmark a question

### Requirement: Keyboard-first list
The result list SHALL be operable by keyboard. Arrow keys move between results, `Enter` opens one, and `Escape` clears the query.

#### Scenario: Move through results
- **WHEN** focus is on a result and the learner presses the down arrow
- **THEN** focus moves to the next result

### Requirement: Filter state kept on return
The index SHALL keep the learner's query, filters and scroll position when they return from a question or article.

#### Scenario: Back to the list
- **WHEN** the learner opens a result and then returns to the index
- **THEN** the same query, filters and scroll position are shown
