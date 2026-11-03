from placement.question_bank import QUESTION_COUNT
import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path

from placement import check_links, config, importer


def sources():
    wb, cat, items = importer.load_records()
    pages = {r["number"]: [1, 1] for r in wb}
    return wb, cat, items, pages


class ValidateTests(unittest.TestCase):
    def test_clean_sources_pass(self):
        self.assertEqual(importer.validate(*sources()), [])

    def test_duplicate_id_detected(self):
        wb, cat, items, pages = sources()
        wb = copy.deepcopy(wb)
        wb[5]["number"] = wb[4]["number"]
        self.assertTrue(any("duplicate" in e for e in importer.validate(wb, cat, items, pages)))

    def test_missing_solution_detected(self):
        wb, cat, items, pages = sources()
        wb = copy.deepcopy(wb)
        wb[9]["code"] = "  "
        self.assertTrue(any("010: missing solution" in e for e in importer.validate(wb, cat, items, pages)))

    def test_missing_page_range_detected(self):
        wb, cat, items, pages = sources()
        del pages[7]
        self.assertTrue(any("007: missing PDF page" in e for e in importer.validate(wb, cat, items, pages)))


class DriftTests(unittest.TestCase):
    def test_drift_detected_and_overridable(self):
        original = importer.AUDIT
        with tempfile.TemporaryDirectory() as d:
            import json
            audit = json.loads(original.read_text())
            audit["hashes"]["workbook"] = "0" * 64
            fake = Path(d) / "audit.json"
            fake.write_text(json.dumps(audit))
            importer.AUDIT = fake
            try:
                with self.assertRaises(importer.ImportError_):
                    importer.check_sources(False)
                importer.check_sources(True)
            finally:
                importer.AUDIT = original


class DatabaseTests(unittest.TestCase):
    def test_imported_database(self):
        con = sqlite3.connect(f"file:{config.EXERCISES_DB}?mode=ro", uri=True)
        self.assertEqual([r[0] for r in con.execute("SELECT id FROM exercise ORDER BY list_position")],
                         list(range(1, QUESTION_COUNT + 1)))
        self.assertEqual(con.execute("SELECT count(*) FROM chapter").fetchone()[0], 54)
        self.assertEqual(con.execute("SELECT count(*) FROM exercise_resource").fetchone()[0], 1200 + (QUESTION_COUNT - 600) * 5)
        self.assertEqual(con.execute("SELECT count(*) FROM exercise_resource GROUP BY exercise_id HAVING count(*) != CASE WHEN exercise_id >= 601 THEN 5 ELSE 2 END").fetchall(), [])
        self.assertEqual(con.execute("SELECT count(*) FROM exercise WHERE provenance='repository' "
                                     "AND license_note IS NULL").fetchone()[0], 0)

    def test_reimport_keeps_authored_content(self):
        con = sqlite3.connect(config.CONTENT_DB)
        con.execute("INSERT OR REPLACE INTO article (id, kind, title, body, updated_at) "
                    "VALUES ('zz-test','prerequisite','t','{}','now')")
        con.commit()
        importer.run()
        try:
            self.assertEqual(con.execute("SELECT title FROM article WHERE id='zz-test'").fetchone()[0], "t")
        finally:
            con.execute("DELETE FROM article WHERE id='zz-test'")
            con.commit()


class LinkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.content = Path(self.tmp.name) / "c.db"
        con = sqlite3.connect(self.content)
        con.executescript((config.SCHEMA_DIR / "content.sql").read_text())
        for a in "abc":
            con.execute("INSERT INTO article (id, kind, title, body, updated_at) VALUES (?, 'prerequisite','t','{}','n')", (a,))
        con.commit()
        self.con = con

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self):
        return check_links.check(content_path=self.content, media_dir=Path(self.tmp.name))

    def test_clean(self):
        self.assertEqual(self.run_check(), [])

    def test_unknown_exercise(self):
        self.con.execute("INSERT INTO exercise_article VALUES (9999,'a','related',1)")
        self.con.commit()
        self.assertTrue(any("unknown exercise_id 9999" in p for p in self.run_check()))

    def test_cycle(self):
        self.con.executemany("INSERT INTO article_prerequisite VALUES (?,?)", [("a", "b"), ("b", "c"), ("c", "a")])
        self.con.commit()
        self.assertTrue(any("cycle" in p for p in self.run_check()))

    def test_missing_and_absolute_media(self):
        self.con.execute("INSERT INTO media (path, description, alt_text) VALUES ('x.png','d','a')")
        self.con.execute("INSERT INTO media (path, description, alt_text) VALUES ('/etc/x.png','d','a')")
        self.con.commit()
        out = "\n".join(self.run_check())
        self.assertIn("media file missing: x.png", out)
        self.assertIn("not relative", out)


if __name__ == "__main__":
    unittest.main()
