import unittest
from fastapi.testclient import TestClient
from placement.main import app

client = TestClient(app)


_tmp = None
_patch = None


def setUpModule():
    """Use an empty content database so that these tests do not depend on the lessons authored so far."""
    global _tmp, _patch
    import tempfile
    from pathlib import Path
    from unittest import mock
    from placement import config, db
    _tmp = tempfile.TemporaryDirectory()
    _patch = mock.patch.object(config, "CONTENT_DB", Path(_tmp.name) / "content.db")
    _patch.start()
    db.init_content()


def tearDownModule():
    _patch.stop()
    _tmp.cleanup()


class ApiTests(unittest.TestCase):
    def get(self, **params):
        r = client.get("/api/exercises", params=params)
        self.assertEqual(r.status_code, 200)
        return r.json()

    def test_answer_types(self):
        for number, expected in [(316, "solution"), (601, "solution"), (653, "answer"), (685, "answer"), (1100, "answer")]:
            response = client.get(f"/api/exercises/{number}")
            self.assertEqual(response.json()["answer_type"], expected)

    def test_all_in_order(self):
        data = self.get()
        self.assertEqual([i["id"] for i in data["items"]], list(range(1, 1101)))

    def test_search_by_id(self):
        self.assertIn(316, [i["id"] for i in self.get(q="316")["items"]])

    def test_search_by_title_word_prefix(self):
        ids = [i["id"] for i in self.get(q="mult atten")["items"]]
        self.assertIn(316, ids)

    def test_search_by_chapter_name_and_filter(self):
        self.assertTrue(self.get(q="attention mechanisms")["total"] > 0)
        self.assertEqual({i["chapter_id"] for i in self.get(chapter="C16")["items"]}, {"C16"})

    def test_topic_search(self):
        self.assertTrue(self.get(q="joins")["total"] > 0)

    def test_special_characters_are_safe(self):
        self.assertEqual(self.get(q='" * ( -- ^ :')["total"], 0)  # punctuation only: no tokens, no error
        self.assertTrue(self.get(q='kl" (divergence')["total"] > 0)  # punctuation is ignored

    def test_lesson_status(self):
        by_id = {i["id"]: i["lesson"] for i in self.get()["items"]}
        self.assertEqual(by_id[316], "planned")
        self.assertEqual(by_id[1], "none")
        self.assertEqual(self.get(lesson="planned")["total"], 72)

    def test_stats_and_chapters(self):
        s = client.get("/api/stats").json()
        self.assertEqual(s, {"imported_questions": 1100, "planned_deep_lessons": 72, "completed_deep_lessons": 0})
        ch = client.get("/api/chapters").json()
        self.assertEqual(len(ch), 54)
        self.assertEqual(sum(c["count"] for c in ch), 1100)


if __name__ == "__main__":
    unittest.main()


class QuestionTests(unittest.TestCase):
    def test_every_question_page_matches_workbook(self):
        import json
        from placement import importer
        from placement.curated import load_solutions
        curated = load_solutions()
        wb = json.loads((importer.REPO / importer.SOURCES["workbook"]).read_text())
        for r in wb:
            page = client.get(f"/api/exercises/{r['number']}")
            self.assertEqual(page.status_code, 200)
            body = page.json()
            self.assertEqual((body["id"], body["title"], body["prompt"]), (r["number"], r["title"], r["question"]))
            self.assertNotIn("solution_code", body)
            self.assertNotIn("code", body)
            sol = client.get(f"/api/exercises/{r['number']}/solution").json()
            self.assertEqual(sol["code"], curated.get(r["number"], {}).get("code") or r["code"])

    def test_unknown_question(self):
        self.assertEqual(client.get("/api/exercises/1101").status_code, 404)
        self.assertEqual(client.get("/api/exercises/0/solution").status_code, 404)

    def test_license_note_and_neighbours(self):
        ids = [i["id"] for i in client.get("/api/exercises").json()["items"]]
        notes = [client.get(f"/api/exercises/{i}").json() for i in (1, 16, 1100)]
        self.assertEqual((notes[0]["previous_id"], notes[0]["next_id"]), (None, 2))
        self.assertIsNone(notes[2]["next_id"])
        self.assertEqual(len(ids), 1100)

    def test_two_ranked_readings_and_supplemental_answer(self):
        original = client.get("/api/exercises/316").json()
        extra = client.get("/api/exercises/601").json()
        self.assertEqual([r["rank"] for r in original["resources"]], [1, 2])
        self.assertEqual([r["rank"] for r in extra["resources"]], [1, 2, 3, 4, 5])
        self.assertTrue(all(r["url"].startswith("https://") for r in extra["resources"]))
        answer = client.get("/api/exercises/601/solution").json()
        self.assertEqual(answer["content_kind"], "coding")
        self.assertTrue(answer["answer"])
