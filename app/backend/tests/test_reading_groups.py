import unittest
from collections import Counter
from urllib.parse import urlparse

from placement.reading_groups import all_groups, overrides

HTTP_ONLY_HOSTS = ("neuralnetworksanddeeplearning.com",)  # Nielsen's book has no valid certificate
PAPER_HOSTS = ("arxiv.org", "ar5iv", "openreview.net", "aclanthology.org", "semanticscholar.org")


class ReadingGroupTests(unittest.TestCase):
    def test_every_group_has_two_distinct_readings(self):
        for g in all_groups():
            first, second = g.readings
            self.assertNotEqual(first.url, second.url, g.ids)
            for r in g.readings:
                parsed = urlparse(r.url)
                if parsed.netloc not in HTTP_ONLY_HOSTS:
                    self.assertEqual(parsed.scheme, "https", (g.ids, r.url))
                self.assertTrue(r.title and r.publisher and r.why, (g.ids, r.url))

    def test_no_paper_hosts(self):
        for g in all_groups():
            for r in g.readings:
                host = urlparse(r.url).netloc.lower()
                self.assertFalse(any(h in host for h in PAPER_HOSTS), (g.ids, r.url))

    def test_ids_are_unique_and_in_range(self):
        counts = Counter(i for g in all_groups() for i in g.ids)
        self.assertEqual([i for i, n in counts.items() if n > 1], [])
        table = overrides()
        self.assertTrue(all(1 <= i <= 1100 for i in table))

    def test_known_topics_get_their_own_readings(self):
        table = overrides()
        self.assertNotEqual(table[320][0].url, "https://jalammar.github.io/illustrated-transformer/")
        online = " ".join(r.url for r in table[618])
        self.assertIn("flash-attention", online)


if __name__ == "__main__":
    unittest.main()
