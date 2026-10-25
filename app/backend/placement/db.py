import sqlite3
from . import config


def open_exercises() -> sqlite3.Connection:
    """Read-only connection. The service never writes exercise data."""
    con = sqlite3.connect(f"file:{config.EXERCISES_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def open_content() -> sqlite3.Connection:
    con = sqlite3.connect(config.CONTENT_DB)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def migrate(con: sqlite3.Connection) -> None:
    """Upgrade an older content.db in place. Each step runs once and refuses to drop authored rows."""
    version = con.execute("PRAGMA user_version").fetchone()[0]
    if version < 1:  # learning_check gained a text check_id and a composite key
        if con.execute("SELECT count(*) FROM learning_check").fetchone()[0]:
            raise RuntimeError("cannot migrate learning_check: it already holds rows")
        con.executescript("""
            DROP TABLE learning_check;
            CREATE TABLE learning_check (
              article_id  TEXT NOT NULL REFERENCES article(id),
              check_id    TEXT NOT NULL,
              kind        TEXT NOT NULL CHECK (kind IN ('prediction','new-case','explain-back')),
              inputs      TEXT NOT NULL,
              answer      TEXT NOT NULL,
              explanation TEXT NOT NULL,
              PRIMARY KEY (article_id, check_id)
            );
            PRAGMA user_version = 1;""")
    if version < 2:  # article gained source_hash
        if "source_hash" not in [r[1] for r in con.execute("PRAGMA table_info(article)")]:
            con.execute("ALTER TABLE article ADD COLUMN source_hash TEXT NOT NULL DEFAULT ''")
        con.execute("PRAGMA user_version = 2")
    if version < 3:  # playground: runtime names 'pyodide' / 'display-only', packages, note
        if con.execute("SELECT count(*) FROM playground").fetchone()[0]:
            raise RuntimeError("cannot migrate playground: it already holds rows")
        con.executescript("""
            DROP TABLE playground;
            CREATE TABLE playground (
              exercise_id    INTEGER PRIMARY KEY,
              runtime        TEXT NOT NULL CHECK (runtime IN ('pyodide','display-only')),
              packages       TEXT NOT NULL DEFAULT '["numpy"]',
              starter_code   TEXT,
              checks         TEXT,
              time_limit_ms  INTEGER NOT NULL DEFAULT 10000,
              max_input_size INTEGER,
              note           TEXT
            );
            PRAGMA user_version = 3;""")
    if version < 4:  # demonstration.data holds precomputed frames
        if "data" not in [r[1] for r in con.execute("PRAGMA table_info(demonstration)")]:
            con.execute("ALTER TABLE demonstration ADD COLUMN data TEXT")
        con.execute("PRAGMA user_version = 4")


def init_content() -> None:
    """Create content.db from its schema if it does not exist; migrate it if it is older."""
    if config.CONTENT_DB.exists():
        con = sqlite3.connect(config.CONTENT_DB)
        try:
            migrate(con)
        finally:
            con.close()
        return
    config.DATA_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(config.CONTENT_DB)
    con.executescript((config.SCHEMA_DIR / "content.sql").read_text())
    con.close()
