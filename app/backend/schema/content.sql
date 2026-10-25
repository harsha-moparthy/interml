-- content.db: authored lessons. Imports never overwrite rows here.
-- exercise_id values refer to exercises.db. SQLite cannot enforce that; the import check and `check_links` do.
PRAGMA foreign_keys = ON;

CREATE TABLE article (
  id           TEXT PRIMARY KEY,            -- slug, for example 'broadcasting'
  kind         TEXT NOT NULL CHECK (kind IN ('lesson','prerequisite')),
  title        TEXT NOT NULL,
  body         TEXT NOT NULL,               -- structured content as JSON (blocks, see CONTENT_FORMAT.md)
  revision     INTEGER NOT NULL DEFAULT 1,
  review_state TEXT NOT NULL DEFAULT 'draft' CHECK (review_state IN ('draft','reviewed','complete')),
  sources      TEXT NOT NULL DEFAULT '[]',  -- JSON list of {title, url}
  updated_at   TEXT NOT NULL,
  source_hash  TEXT NOT NULL DEFAULT ''     -- hash of the whole authored source, detects edits without a revision bump
);

CREATE TABLE exercise_article (
  exercise_id   INTEGER NOT NULL,           -- refers to exercises.db
  article_id    TEXT NOT NULL REFERENCES article(id),
  purpose       TEXT NOT NULL CHECK (purpose IN ('deep-lesson','prerequisite','related')),
  reading_order INTEGER NOT NULL,
  PRIMARY KEY (exercise_id, article_id)
);

CREATE TABLE article_prerequisite (
  article_id      TEXT NOT NULL REFERENCES article(id),
  requires_id     TEXT NOT NULL REFERENCES article(id),
  PRIMARY KEY (article_id, requires_id),
  CHECK (article_id <> requires_id)
);

CREATE TABLE media (
  id           INTEGER PRIMARY KEY,
  path         TEXT NOT NULL UNIQUE,        -- relative to app/media; no absolute paths
  description  TEXT NOT NULL,
  alt_text     TEXT NOT NULL,
  attribution  TEXT
);

CREATE TABLE article_media (
  article_id TEXT NOT NULL REFERENCES article(id),
  media_id   INTEGER NOT NULL REFERENCES media(id),
  PRIMARY KEY (article_id, media_id)
);

CREATE TABLE demonstration (
  id           TEXT PRIMARY KEY,
  article_id   TEXT NOT NULL REFERENCES article(id),
  component    TEXT NOT NULL,               -- registered frontend component name
  input_schema TEXT NOT NULL,               -- JSON schema of editable inputs
  seed         INTEGER,
  steps        TEXT NOT NULL,               -- JSON list of step definitions
  expected     TEXT NOT NULL,               -- JSON expected states, checked against a reference computation
  selection_links TEXT NOT NULL DEFAULT '[]',
  data         TEXT                          -- component payload as JSON, for example precomputed frames
);

CREATE TABLE step_help (
  id                INTEGER PRIMARY KEY,
  demonstration_id  TEXT NOT NULL REFERENCES demonstration(id),
  step_id           TEXT NOT NULL,
  help_type         TEXT NOT NULL CHECK (help_type IN
    ('smaller-example','prerequisite','geometry','expanded-equation','common-mistake','code-connection')),
  content           TEXT NOT NULL,
  prerequisite_id   TEXT REFERENCES article(id)
);

CREATE TABLE learning_check (
  article_id  TEXT NOT NULL REFERENCES article(id),
  check_id    TEXT NOT NULL,                -- stable ID used by the article's check blocks
  kind        TEXT NOT NULL CHECK (kind IN ('prediction','new-case','explain-back')),
  inputs      TEXT NOT NULL,                -- JSON: prompt, answer_type, options, demonstration, case
  answer      TEXT NOT NULL,                -- JSON checked answer: value, tolerance
  explanation TEXT NOT NULL,                -- JSON: correct, wrong (by answer), default
  PRIMARY KEY (article_id, check_id)
);

-- Learner feedback stays apart from technical checks.
CREATE TABLE learner_feedback (
  id         INTEGER PRIMARY KEY,
  article_id TEXT NOT NULL REFERENCES article(id),
  note       TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- Authored; runtime support is a content decision, so it lives here and imports never touch it.
CREATE TABLE playground (
  exercise_id    INTEGER PRIMARY KEY,         -- refers to exercises.db
  runtime        TEXT NOT NULL CHECK (runtime IN ('pyodide','display-only')),
  packages       TEXT NOT NULL DEFAULT '["numpy"]',  -- JSON list, names from content/pyodide-packages.json
  starter_code   TEXT,
  checks         TEXT,                      -- JSON list of {name, code}
  time_limit_ms  INTEGER NOT NULL DEFAULT 10000,
  max_input_size INTEGER,
  note           TEXT                       -- shown beside the playground, for example why it is display-only
);

PRAGMA user_version = 4;
