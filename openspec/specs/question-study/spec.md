# question-study Specification

## Purpose

Defines the single-question study page: how the task, solution, readings and related lessons are presented, and how the learner moves between questions.

## Requirements

### Requirement: Question header
The question page SHALL show the question number, title, chapter, subsection, language, difficulty when known, and a deep-lesson badge when a lesson exists. It SHALL offer done and bookmark controls.

#### Scenario: Header content
- **WHEN** the learner opens question 300
- **THEN** the number, title, chapter, language and available badges are visible without scrolling

### Requirement: Task rendering
The page SHALL render the original task with its equations and paragraphs at a comfortable reading width.

#### Scenario: Equation in the task
- **WHEN** a task contains a display equation
- **THEN** the equation is rendered and scrolls inside its own container if it is wider than the screen

### Requirement: Attempt before solution
The solution SHALL stay hidden until the learner asks for it. The control SHALL state what it does, and the page SHALL keep the existing rule that the service sends the solution only after that request.

#### Scenario: Reveal
- **WHEN** the learner chooses Show solution
- **THEN** the solution loads and appears with syntax highlighting, and the control changes to Hide solution

#### Scenario: New question resets
- **WHEN** the learner moves to another question
- **THEN** its solution starts hidden

#### Scenario: Solution fails to load
- **WHEN** the solution request fails
- **THEN** the page shows the failure and a retry action and keeps the rest of the page usable

### Requirement: Readings presentation
The page SHALL list each ranked reading with its title, publisher, access note and the reason it was chosen. Links SHALL open in a new tab.

#### Scenario: Two readings
- **WHEN** a question has two readings
- **THEN** both appear in rank order, each showing its publisher and reason

### Requirement: Related lessons
When linked articles exist, the page SHALL show them prominently with their type and review state, ahead of the readings.

#### Scenario: Question with a deep lesson
- **WHEN** a question has a complete deep lesson
- **THEN** a lesson card links to it near the top of the page

### Requirement: Question navigation
The page SHALL link to the previous and next question and back to the list. Left and right arrow shortcuts SHALL move between questions when focus is not in a field. The page SHALL also show the learner's position within the chapter.

#### Scenario: Keyboard step
- **WHEN** the learner presses the right arrow on question 300
- **THEN** the next question opens

#### Scenario: First and last
- **WHEN** the learner is on the first or last question
- **THEN** the missing direction is shown as unavailable and its shortcut does nothing

### Requirement: Source and licence notes kept
The page SHALL keep the source link, repository link, solutions PDF page range and licence note.

#### Scenario: Source footer
- **WHEN** a question has a source URL and a licence note
- **THEN** both appear in the page footer
