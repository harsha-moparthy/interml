import unittest
from fastapi.testclient import TestClient
from placement import search
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


def ids(q, **kw):
    r = client.get("/api/search", params={"q": q, **kw})
    assert r.status_code == 200, r.text
    return [i["id"] for i in r.json()["items"]], r.json()


class ParseTests(unittest.TestCase):
    def test_filters_phrases_and_exclusions(self):
        p = search.parse('chapter:attention id:300-350 "causal mask" -dropout softmax lang:cpp')
        self.assertEqual(p["filters"], {"chapter": ["attention"], "id": ["300-350"], "lang": ["cpp"]})
        self.assertEqual(p["phrases"], ["causal mask"])
        self.assertEqual(p["exclude"], ["dropout"])
        self.assertEqual(p["words"], ["softmax"])

    def test_unknown_key_is_plain_text(self):
        self.assertEqual(search.parse("foo:bar")["filters"], {})

    def test_spelling_correction(self):
        self.assertEqual(search.correct_spelling(["attenion"]), {"attenion": "attention"})
        self.assertEqual(search.correct_spelling(["atten"]), {})     # still typing a real word
        self.assertEqual(search.correct_spelling(["gaussian"]), {})  # correct already

    def test_abbreviation_expansion_both_ways(self):
        groups = search.expand(["kullback", "leibler"])
        self.assertTrue(any(g == ["kl"] for g in groups))
        self.assertIn("kullback", search.expand(["kl"])[0])


class SearchApiTests(unittest.TestCase):
    def test_id_first(self):
        got, _ = ids("316")
        self.assertEqual(got[0], 316)

    def test_typo_is_corrected_and_reported(self):
        got, body = ids("attenion")
        self.assertEqual(body["corrected"], {"attenion": "attention"})
        self.assertTrue({309, 310, 316} & set(got[:5]))

    def test_acronym_and_spelled_out_find_the_same_question(self):
        for q in ("kl divergence", "kullback leibler", "kl"):
            top = ids(q)[0][:10]
            self.assertIn(100, top, q)
            titles = [client.get(f"/api/exercises/{i}").json()["title"] for i in top]
            self.assertGreaterEqual(sum(("kl" in t.lower() or "kullback" in t.lower()) for t in titles), 3, (q, titles))

    def test_meaning_search_finds_titles_without_the_words(self):
        got, body = ids("make training use less memory")
        self.assertTrue(body["semantic"])
        self.assertTrue({57, 55, 56, 710, 852} & set(got[:8]))

    def test_appendix_chapters_filter(self):
        for n in range(1, 19):
            cid = f"A{n:02d}"
            expected = client.get(f"/api/exercises?chapter={cid}&limit=1100").json()["total"]
            for q in (f"chapter:{cid}", f"ch:{cid.lower()}"):
                _, body = ids(q)
                self.assertEqual(body["total"], expected, q)
                self.assertGreater(expected, 0, cid)

    def test_filters(self):
        got, _ = ids("attention chapter:c16")
        self.assertTrue(got)
        for i in got:
            self.assertEqual(client.get(f"/api/exercises/{i}").json()["chapter_id"], "C16")
        r, _ = ids("id:310-312")
        self.assertEqual(r, [310, 311, 312])
        _, body = ids("lesson:planned")
        self.assertEqual(body["total"], 72)
        cpp, _ = ids("lang:cpp")
        self.assertEqual(len(cpp), 9)

    def test_exclusion_and_phrase(self):
        with_dropout, _ = ids("attention")
        without, _ = ids("attention -dropout")
        self.assertNotIn(315, without)
        phrase, _ = ids('"multi head"')
        self.assertIn(316, phrase)

    def test_segments_mark_matches_without_html(self):
        _, body = ids("attention")
        first = body["items"][0]
        self.assertTrue(any(p["hit"] for p in first["title_parts"] + first["snippet_parts"]))
        self.assertEqual("".join(p["text"] for p in first["title_parts"]), first["title"])

    def test_empty_and_odd_queries_do_not_fail(self):
        for q in ("", "   ", '"', "(((", "-", "a:b:c", "' OR 1=1 --", "x" * 500, "éè"):
            r = client.get("/api/search", params={"q": q})
            self.assertEqual(r.status_code, 200, q)

    def test_keyword_only_mode_works_without_the_model(self):
        _, body = ids("softmax", mode="keyword")
        self.assertEqual(body["mode"], "keyword")
        self.assertTrue(body["total"] > 0)


class QualityTests(unittest.TestCase):
    """Guard the measured quality. Numbers come from tests/search_eval.py (tuning set) and the held-out set."""
    def run_set(self, name):
        import importlib, os
        os.environ["EVAL_SET"] = name
        import search_eval
        importlib.reload(search_eval)
        return search_eval.evaluate("hybrid")

    @classmethod
    def setUpClass(cls):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))

    def test_tuning_set(self):
        hit, mrr, _ = self.run_set("search_eval.json")
        self.assertGreaterEqual(hit, 0.85)
        self.assertGreaterEqual(mrr, 0.60)

    def test_holdout_set(self):
        hit, mrr, _ = self.run_set("search_holdout.json")
        self.assertGreaterEqual(hit, 0.90)
        self.assertGreaterEqual(mrr, 0.85)


if __name__ == "__main__":
    unittest.main()
