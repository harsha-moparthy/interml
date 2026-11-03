-- exercises.db: imported, read-only at runtime. Rebuilt from source by the import (T06).
PRAGMA foreign_keys = ON;

CREATE TABLE source_file (
  name        TEXT PRIMARY KEY,           -- 'pdf', 'tex', 'workbook', 'catalog'
  path        TEXT NOT NULL,              -- relative to the repository root
  sha256      TEXT NOT NULL,
  imported_at TEXT NOT NULL               -- ISO 8601 UTC
);

CREATE TABLE chapter (
  id        TEXT PRIMARY KEY,             -- 'C01'..'C36' original, 'A01'..'A18' supplemental
  position  INTEGER NOT NULL UNIQUE,      -- display order; separate from identity
  title     TEXT NOT NULL
);

CREATE TABLE subsection (
  id         INTEGER PRIMARY KEY,
  chapter_id TEXT NOT NULL REFERENCES chapter(id),
  position   INTEGER NOT NULL,
  title      TEXT NOT NULL,
  UNIQUE (chapter_id, position)
);

CREATE TABLE exercise (
  id             INTEGER PRIMARY KEY CHECK (id >= 1), -- permanent question ID; catalog validates bounds
  list_position  INTEGER NOT NULL UNIQUE,   -- original order; equals id in the first edition
  chapter_id     TEXT NOT NULL REFERENCES chapter(id),
  subsection_id  INTEGER NOT NULL REFERENCES subsection(id),
  title          TEXT NOT NULL,
  prompt         TEXT NOT NULL,             -- original task text, never edited
  solution_code  TEXT NOT NULL DEFAULT '', -- from workbook.json; empty for theory-only supplemental questions
  answer_text    TEXT NOT NULL DEFAULT '', -- study/answer guide for supplemental questions
  content_kind   TEXT NOT NULL DEFAULT 'coding' CHECK (content_kind IN ('coding', 'theory', 'system-design')),
  supplemental   INTEGER NOT NULL DEFAULT 0 CHECK (supplemental IN (0, 1)),
  language       TEXT NOT NULL,             -- 'Python' or 'C++'
  provenance     TEXT NOT NULL,             -- 'original' or 'repository'
  source_key     TEXT NOT NULL,             -- 'DM', 'TT', 'RESUME', 'EXT'
  source_url     TEXT,
  repository_url TEXT,
  difficulty     TEXT,                      -- from catalog.json when present
  license_note   TEXT,                      -- set for repository-derived records
  pdf_page_first INTEGER,
  pdf_page_last  INTEGER,
  record_sha256  TEXT NOT NULL,             -- hash of the source record, detects source drift
  CHECK (length(solution_code) > 0 OR length(answer_text) > 0)
);
CREATE INDEX exercise_chapter ON exercise(chapter_id, list_position);

CREATE TABLE exercise_topic (               -- search by topic: subsection title and category terms
  exercise_id INTEGER NOT NULL REFERENCES exercise(id),
  topic       TEXT NOT NULL,
  PRIMARY KEY (exercise_id, topic)
);

-- Two readings for workbook exercises; five for authored appendix answers.
-- structured data so coverage and ordering are testable.
CREATE TABLE exercise_resource (
  exercise_id INTEGER NOT NULL REFERENCES exercise(id),
  rank        INTEGER NOT NULL CHECK (rank BETWEEN 1 AND 5),
  title       TEXT NOT NULL,
  url         TEXT NOT NULL,
  publisher   TEXT NOT NULL,
  access_note TEXT NOT NULL DEFAULT 'Open access',
  why         TEXT NOT NULL,
  PRIMARY KEY (exercise_id, rank),
  UNIQUE (exercise_id, url)
);

-- Full-text search over ID, title, chapter, topics, prompt, and terms taken from the solution code.
-- Filled by the import. bm25 column weights are set in search.py.
CREATE VIRTUAL TABLE exercise_search USING fts5(
  exercise_id UNINDEXED, id_text, title, chapter, topics, prompt, code_terms, tokenize = 'porter unicode61 remove_diacritics 2'
);

-- Vocabulary of the index, used to correct misspelled query words.
CREATE VIRTUAL TABLE exercise_vocab USING fts5vocab(exercise_search, 'row');

CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Semantic search vectors. Absent when the embedding model is not installed.
CREATE TABLE exercise_embedding (
  exercise_id INTEGER PRIMARY KEY REFERENCES exercise(id),
  vector      BLOB NOT NULL          -- float32, L2-normalised
);
