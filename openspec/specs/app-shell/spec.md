# app-shell Specification

## Purpose

Defines the shared look, theming, navigation and system states that every page of the learning site uses, so the site reads as one product in light and dark themes.

## Requirements

### Requirement: One visual system across all pages
The system SHALL render the question index, question pages and article pages with the same colour, type, spacing and component styles. No page SHALL use colours that ignore the active theme.

#### Scenario: Article page follows the dark theme
- **WHEN** the active theme is dark and the learner opens an article
- **THEN** headings, body text, figures, callouts, checks, code and playground panels are readable against the dark background

#### Scenario: Pages feel consistent
- **WHEN** the learner moves from the index to a question and then to an article
- **THEN** the header, fonts, buttons, links and cards look like the same product

### Requirement: Theme choice
The system SHALL offer light, dark and system theme choices. The default SHALL follow the operating-system setting. A manual choice SHALL persist across reloads in the same browser.

#### Scenario: Default follows the system
- **WHEN** a first-time visitor's system prefers dark
- **THEN** the site opens in the dark theme

#### Scenario: Manual choice is remembered
- **WHEN** the learner selects the light theme and reloads the page
- **THEN** the site opens in the light theme

#### Scenario: Storage unavailable
- **WHEN** the browser blocks local storage
- **THEN** the theme switch still works for the current visit and no error is shown

### Requirement: Persistent header
The system SHALL show one header on every page. It SHALL contain the site name linking to the index, a search entry, a link to progress, and the theme switch.

#### Scenario: Header on every page
- **WHEN** the learner opens any question or article
- **THEN** the same header is visible and its site name returns to the index

#### Scenario: Search from a deep page
- **WHEN** the learner uses the header search entry on a question page
- **THEN** the index opens with the search box focused

### Requirement: Keyboard shortcuts
The system SHALL keep the existing `/` shortcut that focuses search on the index. Shortcuts SHALL NOT fire while the learner types in a field. The system SHALL list its shortcuts in a help overlay opened with `?`.

#### Scenario: Typing is not hijacked
- **WHEN** focus is in a text field and the learner types `/` or `?`
- **THEN** the character is entered and no shortcut runs

#### Scenario: Help overlay
- **WHEN** the learner presses `?` outside a field
- **THEN** an overlay lists every shortcut and closes with `Escape`

### Requirement: Loading, empty and error states
The system SHALL show a placeholder with the shape of the final content while data loads. It SHALL show a helpful message when a list is empty. It SHALL show a clear message with a retry action when a request fails.

#### Scenario: Slow load
- **WHEN** a page's data has not arrived
- **THEN** a placeholder layout appears instead of a blank page or a bare "Loading…" line

#### Scenario: Request failure
- **WHEN** a request fails
- **THEN** the page states what failed and offers a retry that repeats the request

### Requirement: Motion and accessibility
The system SHALL keep all functions available by keyboard and with a screen reader. Text and controls SHALL meet WCAG 2.2 AA contrast in both themes. The system SHALL disable non-essential animation when the learner prefers reduced motion.

#### Scenario: Reduced motion
- **WHEN** the system setting prefers reduced motion
- **THEN** page transitions, skeleton shimmer and other decorative animation do not run

#### Scenario: Focus is visible
- **WHEN** the learner tabs through the header and a page
- **THEN** each focused control shows a clear focus indicator in both themes

### Requirement: Existing routes and behaviour preserved
The system SHALL keep every existing URL working, including `#/`, `#/question/N`, `#/article/ID` and `#/lesson-db/ID`.

#### Scenario: Old links still open
- **WHEN** the learner opens a saved `#/question/300` link
- **THEN** question 300 opens

### Requirement: Responsive layout
The system SHALL work from 360 px to wide desktop widths without horizontal page scroll. Wide content such as equations, tables and code SHALL scroll inside its own container.

#### Scenario: Phone width
- **WHEN** a page is shown at 360 px width
- **THEN** the page has no horizontal scroll and the header remains usable
