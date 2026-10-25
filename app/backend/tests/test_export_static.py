import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from placement import export_static


class ExportStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name) / "static-api"
        with mock.patch.object(export_static, "OUT", cls.out):
            export_static.main()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def load(self, rel):
        return json.loads((self.out / rel).read_text(encoding="utf-8"))

    def test_every_question_has_a_page_and_a_solution(self):
        ids = [e["id"] for e in self.load("all.json")["items"]]
        self.assertEqual(ids, list(range(1, 1101)))
        for i in (1, 316, 601, 1100):
            self.assertEqual(self.load(f"exercises/{i}.json")["id"], i)
            self.assertEqual(self.load(f"exercises/{i}/solution.json")["id"], i)
        self.assertEqual(len(list((self.out / "exercises").glob("*.json"))), 1100)
        self.assertEqual(len(list((self.out / "exercises").glob("*/solution.json"))), 1100)

    def test_chapters_cover_main_and_appendix(self):
        chapters = {c["id"]: c["count"] for c in self.load("chapters.json")}
        self.assertEqual(len(chapters), 54)
        self.assertEqual(sum(chapters.values()), 1100)
        self.assertEqual((chapters["A01"], chapters["A18"]), (52, 50))

    def test_every_article_is_exported(self):
        listing = self.load("articles.json")
        self.assertTrue(listing)
        for a in listing:
            self.assertEqual(self.load(f"articles/{a['id']}.json")["id"], a["id"])

    def test_vectors_match_the_search_index(self):
        index = self.load("search-index.json")
        self.assertEqual([d["id"] for d in index["docs"]], list(range(1, 1101)))
        self.assertEqual((self.out / "embeddings.bin").stat().st_size, 1100 * 384 * 4)

    def test_solutions_stay_out_of_the_question_files(self):
        self.assertNotIn("solution_code", self.load("exercises/316.json"))


if __name__ == "__main__":
    unittest.main()
