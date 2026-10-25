# Spec Delta

## ADDED Requirements

### Requirement: Appendix chapters selectable
The chapter filter, the chapter list and the `chapter:` and `ch:` operators SHALL accept appendix chapters A01–A18 as well as C01–C36, in upper or lower case. A bare number SHALL keep meaning the main chapter of that number.

#### Scenario: Choose an appendix chapter
- **WHEN** the learner chooses A01 in the chapter list
- **THEN** the 52 questions of A01 are listed

#### Scenario: Operator with an appendix chapter
- **WHEN** the learner types `chapter:a18`
- **THEN** only the 50 questions of A18 are listed and the chip shows `chapter: a18`

#### Scenario: Bare number unchanged
- **WHEN** the learner types `chapter:5`
- **THEN** the questions of C05 are listed
