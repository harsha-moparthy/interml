import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from placement import articles, config
from placement.main import app


def full_lesson(id, **kw):
    """A lesson that meets every completion rule."""
    from placement.curated import load_solutions
    code = load_solutions()[30]["code"].splitlines()
    anchor = next(i for i, line in enumerate(code, 1) if "return np.asarray(x)" in line)
    base = art(id, kind="lesson", review_state="complete", exercises=[{"id": 30, "purpose": "deep-lesson", "order": 1}],
               blocks=[{"type": "check", "id": "p1"}, {"type": "check", "id": "n1"},
                       {"type": "callout", "kind": "mistake", "text": "x"},
                       {"type": "walkthrough", "exercise_id": 30, "notes": [{"lines": [anchor, anchor], "match": "np.asarray(x)", "text": "Insert axes."}]}],
               demonstrations=[],
               checks=[{"id": "p1", "kind": "prediction", "prompt": "?", "answer_type": "choice", "options": ["a", "b"],
                        "answer": "a", "explanation": {"correct": "yes", "default": "no"}},
                       {"id": "n1", "kind": "new-case", "prompt": "?", "answer_type": "number", "answer": 3,
                        "explanation": {"correct": "yes", "default": "no"}}])
    base.update(kw)
    return base


def art(id, **kw):
    base = {"id": id, "kind": "prerequisite", "title": id.title(), "revision": 1, "review_state": "draft",
            "blocks": [{"type": "paragraph", "text": "Text."}]}
    base.update(kw)
    return base


class ArticleCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "content" / "articles").mkdir(parents=True)
        (root / "media").mkdir()
        self.root = root
        for name, val in (("CONTENT_DIR", root / "content"), ("MEDIA_DIR", root / "media"),
                          ("CONTENT_DB", root / "content.db")):
            p = mock.patch.object(config, name, val)
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, *items):
        for a in items:
            (self.root / "content" / "articles" / f"{a['id']}.json").write_text(json.dumps(a))

    def fails(self, fragment):
        with self.assertRaises(articles.ArticleError) as cm:
            articles.run()
        self.assertIn(fragment, str(cm.exception))


class ImportTests(ArticleCase):
    def test_import_and_idempotence(self):
        self.write(art("shapes", review_state="complete"),
                   art("bcast", prerequisites=["shapes"], review_state="complete",
                       blocks=[{"type": "paragraph", "text": "See [[article:shapes|shapes]] and [[q:30]]."}]),
                   art("lesson-030", kind="lesson", prerequisites=["bcast"],
                       exercises=[{"id": 30, "purpose": "deep-lesson", "order": 1}]))
        self.assertEqual(sorted(articles.run()["added"]), ["bcast", "lesson-030", "shapes"])
        self.assertEqual(sorted(articles.run()["unchanged"]), ["bcast", "lesson-030", "shapes"])

    def test_revision_rules(self):
        self.write(art("a"))
        articles.run()
        self.write(art("a", blocks=[{"type": "paragraph", "text": "Changed."}]))
        self.fails("raise the revision")
        self.write(art("a", revision=2, blocks=[{"type": "paragraph", "text": "Changed."}]))
        self.assertEqual(articles.run()["updated"], ["a"])
        self.write(art("a", revision=1))
        self.fails("lower than stored")

    def test_removing_a_source_file_keeps_the_article(self):
        self.write(art("a"), art("b"))
        articles.run()
        (self.root / "content" / "articles" / "b.json").unlink()
        articles.run()
        con = articles.db.open_content()
        self.assertEqual(con.execute("SELECT count(*) FROM article").fetchone()[0], 2)

    def test_rejections(self):
        cases = {
            "unknown prerequisite": art("a", prerequisites=["nope"]),
            "unknown exercise": art("a", exercises=[{"id": 9999, "purpose": "related", "order": 1}]),
            "unknown article": art("a", blocks=[{"type": "paragraph", "text": "[[article:nope]]"}]),
            "unknown question": art("a", blocks=[{"type": "paragraph", "text": "[[q:999999]]"}]),
            "unknown type": art("a", blocks=[{"type": "video", "src": "x"}]),
            "needs path, description, and alt_text": art("a", media=[{"path": "x.png", "description": "d", "alt_text": ""}]),
            "not listed in media": art("a", blocks=[{"type": "image", "path": "x.png"}]),
            "media file missing": art("a", media=[{"path": "x.png", "description": "d", "alt_text": "a"}],
                                      blocks=[{"type": "image", "path": "x.png"}]),
            "needs an exercises entry": art("a", kind="lesson"),
            "table row length": art("a", blocks=[{"type": "table", "header": ["a", "b"], "rows": [["1"]]}]),
            "callout kind": art("a", blocks=[{"type": "callout", "kind": "fun", "text": "x"}]),
            "lowercase slug": art("Bad_ID"),
        }
        for fragment, a in cases.items():
            with self.subTest(fragment):
                for f in (self.root / "content" / "articles").glob("*.json"):
                    f.unlink()
                self.write(a)
                self.fails(fragment)

    def test_cycle_and_complete_with_incomplete_prerequisite(self):
        self.write(art("a", prerequisites=["b"]), art("b", prerequisites=["a"]))
        self.fails("cycle")
        for f in (self.root / "content" / "articles").glob("*.json"):
            f.unlink()
        self.write(art("p"), art("q", prerequisites=["p"], review_state="complete"))
        self.fails("prerequisite 'p' is not complete")

    def test_all_or_nothing(self):
        self.write(art("good"), art("bad", prerequisites=["nope"]))
        self.fails("unknown prerequisite")
        con = articles.db.open_content()
        self.assertEqual(con.execute("SELECT count(*) FROM article").fetchone()[0], 0)

    def test_media_is_stored(self):
        (self.root / "media" / "fig.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
        self.write(art("a", media=[{"path": "fig.svg", "description": "Figure", "alt_text": "A figure"}],
                       blocks=[{"type": "image", "path": "fig.svg", "caption": "Fig"}]))
        articles.run()
        con = articles.db.open_content()
        self.assertEqual(con.execute("SELECT alt_text FROM media").fetchone()[0], "A figure")


class ApiTests(ArticleCase):
    def test_article_route(self):
        self.write(art("shapes", review_state="complete"),
                   art("lesson-030", kind="lesson", prerequisites=["shapes"],
                       exercises=[{"id": 30, "purpose": "deep-lesson", "order": 1}],
                       blocks=[{"type": "equation", "latex": "a+b"}]))
        articles.run()
        c = TestClient(app)
        d = c.get("/api/articles/lesson-030").json()
        self.assertEqual(d["blocks"], [{"type": "equation", "latex": "a+b"}])
        self.assertEqual([p["id"] for p in d["prerequisites"]], ["shapes"])
        self.assertEqual([e["id"] for e in d["exercises"]], [30])
        self.assertEqual([r["id"] for r in c.get("/api/articles/shapes").json()["required_by"]], ["lesson-030"])
        self.assertEqual(c.get("/api/articles/missing").status_code, 404)
        q = c.get("/api/exercises/30").json()
        self.assertEqual([a["id"] for a in q["articles"]], ["lesson-030"])
        stats = c.get("/api/stats").json()
        self.assertEqual(stats["completed_deep_lessons"], 0)  # draft lesson is not complete

    def test_complete_lesson_is_counted_once_everything_is_complete(self):
        self.write(art("shapes", review_state="complete"),
                   full_lesson("lesson-030", prerequisites=["shapes"]))
        articles.run()
        c = TestClient(app)
        self.assertEqual(c.get("/api/stats").json()["completed_deep_lessons"], 1)
        item = [i for i in c.get("/api/exercises", params={"q": "030"}).json()["items"] if i["id"] == 30][0]
        self.assertEqual(item["lesson"], "complete")


class InteractiveTests(ArticleCase):
    def test_full_lesson_imports_and_is_served(self):
        self.write(full_lesson("lesson-030"))
        articles.run()
        d = TestClient(app).get("/api/articles/lesson-030").json()
        self.assertEqual(d["demonstrations"], {})
        self.assertEqual(d["checks"]["p1"]["answer"]["value"], "a")
        self.assertEqual(d["checks"]["n1"]["explanation"]["default"], "no")

    def test_edit_without_revision_bump_is_caught_for_interactive_content(self):
        self.write(full_lesson("lesson-030"))
        articles.run()
        changed = full_lesson("lesson-030")
        changed["checks"][1]["answer"] = 4
        self.write(changed)
        self.fails("raise the revision")
        changed["revision"] = 2
        self.write(changed)
        articles.run()
        con = articles.db.open_content()
        self.assertIn('"value": 4', con.execute("SELECT answer FROM learning_check WHERE check_id='n1'").fetchone()[0])

    def test_rejections(self):
        def mutate(f):
            a = full_lesson("lesson-030")
            f(a)
            return a
        cases = {
            "unknown check": mutate(lambda a: a["blocks"].append({"type": "check", "id": "zzz"})),
            "answer must be one of the options": mutate(lambda a: a["checks"][0].update(answer="c")),
            "explanation.default is required": mutate(lambda a: a["checks"][1]["explanation"].pop("default")),
            "needs a prediction check": mutate(lambda a: a["checks"][0].update(kind="new-case")),
            "needs a new-case check": mutate(lambda a: a["checks"][1].update(kind="prediction")),
            "needs a comparison or a failure": mutate(lambda a: a.update(blocks=[b for b in a["blocks"] if b["type"] != "callout"])),
        }
        for fragment, a in cases.items():
            with self.subTest(fragment):
                for f in (self.root / "content" / "articles").glob("*.json"):
                    f.unlink()
                self.write(a)
                self.fails(fragment)

    def test_walkthrough_is_checked_against_the_real_solution(self):
        def wt(**note):
            a = full_lesson("lesson-030")
            a["blocks"][-1]["notes"][0].update(note)
            return a
        cases = {
            "is not in lines": wt(match="does_not_exist"),
            "solution 30 has": wt(lines=[5, 99]),
            "lines must be": wt(lines=[3, 1]),
            "needs text and match": wt(text=""),
        }
        for fragment, a in cases.items():
            with self.subTest(fragment):
                self.write(a)
                self.fails(fragment)
        self.write(wt())
        articles.run()
        blocks = TestClient(app).get("/api/articles/lesson-030").json()["blocks"]
        self.assertEqual(blocks[-1]["type"], "walkthrough")

    def test_complete_lesson_needs_a_walkthrough(self):
        a = full_lesson("lesson-030")
        a["blocks"].pop()
        self.write(a)
        self.fails("needs a walkthrough")

    def test_draft_lesson_may_be_incomplete(self):
        self.write(full_lesson("lesson-030", review_state="draft", checks=[], demonstrations=[],
                               blocks=[{"type": "paragraph", "text": "Draft."}]))
        articles.run()

    def test_migration_from_version_0(self):
        import sqlite3
        old = self.root / "old.db"
        con = sqlite3.connect(old)
        con.executescript((config.SCHEMA_DIR / "content.sql").read_text())
        con.executescript("DROP TABLE learning_check; CREATE TABLE learning_check (id INTEGER PRIMARY KEY, article_id TEXT, "
                          "kind TEXT, inputs TEXT, answer TEXT, explanation TEXT); ALTER TABLE article DROP COLUMN source_hash;"
                          "PRAGMA user_version = 0;")
        articles.db.migrate(con)
        self.assertEqual(con.execute("PRAGMA user_version").fetchone()[0], 4)
        cols = [r[1] for r in con.execute("PRAGMA table_info(learning_check)")]
        self.assertIn("check_id", cols)
        self.assertIn("source_hash", [r[1] for r in con.execute("PRAGMA table_info(article)")])


class PlaygroundTests(ArticleCase):
    def lesson(self, **pg):
        base = {"exercise_id": 30, "runtime": "pyodide", "starter_code": "print(1)",
                "checks": [{"name": "ok", "code": "assert True"}]}
        base.update(pg)
        return full_lesson("lesson-030", playgrounds=[base],
                           blocks=[*full_lesson("lesson-030")["blocks"], {"type": "playground", "exercise_id": 30}])

    def test_playground_import_and_api(self):
        self.write(self.lesson(time_limit_ms=5000, max_input_size=100))
        articles.run()
        pg = TestClient(app).get("/api/articles/lesson-030").json()["playgrounds"]["30"]
        self.assertEqual((pg["runtime"], pg["packages"], pg["time_limit_ms"], pg["max_input_size"]),
                         ("pyodide", ["numpy"], 5000, 100))
        self.assertEqual(pg["checks"][0]["name"], "ok")

    def test_pandas_is_an_allowed_package(self):
        self.write(self.lesson(packages=["numpy", "pandas"]))
        articles.run()
        pg = TestClient(app).get("/api/articles/lesson-030").json()["playgrounds"]["30"]
        self.assertEqual(pg["packages"], ["numpy", "pandas"])

    def test_display_only_needs_a_reason(self):
        self.write(self.lesson(runtime="display-only", starter_code=None))
        self.fails("needs a note")
        self.write(self.lesson(runtime="display-only", starter_code=None, note="Needs PyTorch."))
        articles.run()

    def test_rejections(self):
        cases = {
            "needs starter_code": self.lesson(starter_code=" "),
            "runtime must be": self.lesson(runtime="python"),
            "packages not downloaded": self.lesson(packages=["numpy", "torch"]),
            "time_limit_ms must be 1000": self.lesson(time_limit_ms=500),
            "time_limit_ms must be 1000-60000": self.lesson(time_limit_ms=10**6),
            "must list exercise 31": self.lesson(exercise_id=31),
            "each check needs name and code": self.lesson(checks=[{"name": "x"}]),
        }
        for fragment, a in cases.items():
            with self.subTest(fragment):
                for f in (self.root / "content" / "articles").glob("*.json"):
                    f.unlink()
                self.write(a)
                self.fails(fragment)

    def test_block_without_definition(self):
        a = full_lesson("lesson-030", blocks=[*full_lesson("lesson-030")["blocks"], {"type": "playground", "exercise_id": 30}])
        self.write(a)
        self.fails("is not defined in playgrounds")

    def test_unknown_exercise(self):
        a = self.lesson(exercise_id=9999)
        a["exercises"].append({"id": 9999, "purpose": "related", "order": 2})
        a["blocks"][-1]["exercise_id"] = 9999
        self.write(a)
        self.fails("unknown exercise")

    def test_migration_to_v3(self):
        import sqlite3
        con = sqlite3.connect(self.root / "v2.db")
        con.executescript((config.SCHEMA_DIR / "content.sql").read_text())
        con.executescript("DROP TABLE playground; CREATE TABLE playground (exercise_id INTEGER PRIMARY KEY, runtime TEXT "
                          "CHECK (runtime IN ('pyodide-numpy','display-only')), starter_code TEXT, checks TEXT, "
                          "time_limit_ms INTEGER, max_input_size INTEGER); PRAGMA user_version = 2;")
        articles.db.migrate(con)
        self.assertEqual(con.execute("PRAGMA user_version").fetchone()[0], 4)
        self.assertIn("packages", [r[1] for r in con.execute("PRAGMA table_info(playground)")])


if __name__ == "__main__":
    unittest.main()
