# Spec Delta

## Purpose

Defines the long-form reading experience for deep lessons and prerequisite articles: layout, navigation within an article, progress, and consistent display of embedded blocks.

## ADDED Requirements

### Requirement: Reading layout
Article text SHALL use a comfortable measure and line height. Figures, equations, tables, code and interactive blocks MAY be wider than the text but SHALL stay within the page.

#### Scenario: Long paragraph
- **WHEN** an article paragraph is shown on a wide screen
- **THEN** its line length stays within a comfortable reading measure

### Requirement: Linked table of contents
An article with several sections SHALL show a table of contents whose entries link to their sections. On wide screens it SHALL stay visible while reading and mark the section in view. On narrow screens it SHALL be reachable without covering the text permanently.

#### Scenario: Jump to a section
- **WHEN** the learner chooses a table-of-contents entry
- **THEN** the page scrolls to that section and focus moves to its heading

#### Scenario: Section in view
- **WHEN** the learner scrolls into a new section
- **THEN** the matching entry is marked as current

#### Scenario: Phone width
- **WHEN** the page is shown at 360 px width
- **THEN** the table of contents opens from a control and does not cover the text when closed

### Requirement: Reading progress
The article page SHALL show how far the learner is through the article. The system SHALL remember the learner's position so a later visit can offer to continue.

#### Scenario: Progress indicator
- **WHEN** the learner scrolls halfway through an article
- **THEN** the progress indicator shows about half

#### Scenario: Continue reading
- **WHEN** the learner returns to a partly read article
- **THEN** the page offers a continue action that scrolls to the saved position

### Requirement: Article header
The header SHALL show the title, headline and subtitle when present, kind (deep lesson or prerequisite), review state and the questions the article supports. It SHALL show the estimated reading time.

#### Scenario: Lesson header
- **WHEN** the learner opens a deep lesson
- **THEN** the header shows its kind, linked question numbers and reading time

### Requirement: Prerequisites and follow-ups
The page SHALL show prerequisite articles before the body, with their review state. It SHALL show articles that depend on this one after the body. The page SHALL link back to the supported questions.

#### Scenario: Prerequisites listed
- **WHEN** an article has prerequisites
- **THEN** they appear before the first section, each linking to its article

### Requirement: Embedded blocks match the theme
Figures, equations, callouts, tables, checks, code walkthroughs and playgrounds SHALL use the shared visual system and stay legible in both themes. Callout kinds (definition, note, mistake, boundary, example) SHALL be distinguishable by label and shape as well as colour.

#### Scenario: Mistake callout
- **WHEN** an article contains a "Common mistake" callout
- **THEN** it carries a visible label and icon or marker that does not depend on colour

### Requirement: Interactive blocks keep working
Checks, code walkthroughs and Python playgrounds SHALL behave as before, including run, stop, reset, timeouts and error display.

#### Scenario: Run a playground
- **WHEN** the learner runs a playground in the restyled page
- **THEN** output streams, Stop works, and errors are shown as before

### Requirement: Print-friendly reading
Printing an article SHALL produce a readable page without the header, table of contents or interactive controls.

#### Scenario: Print preview
- **WHEN** the learner prints an article
- **THEN** the output shows the article text, figures and equations in a light, ink-friendly style
