import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from placement import importer, question_bank
from placement.main import app


class InterviewAdditionsTests(unittest.TestCase):
    def test_appendix_mapping_and_original_ids(self):
        rows, chapters = importer.load_supplemental()
        self.assertEqual([r['id'] for r in rows[:500]], list(range(601, 1101)))
        self.assertEqual(len(chapters), 18)
        expected = {1101: 3, 1112: 10, 1113: 4, 1121: 6, 1127: 7, 1129: 5, 1136: 8}
        by_id = {r['id']: r for r in rows}
        for number, chapter in expected.items():
            self.assertEqual(by_id[number]['chapter_number'], chapter)
        for item in question_bank.load_additions():
            self.assertNotIn('fde', item['prompt'].lower())
            self.assertIn('/blob/', item['source_url'])

    def test_catalog_rejects_duplicate_ids_and_invalid_chapters(self):
        for change in ({'id': 1102}, {'chapter_number': 19}, {'prompt': ''}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                items = question_bank.load_additions()
                items[0].update(change)
                path = Path(d) / 'additions.json'
                path.write_text(json.dumps(items))
                with patch.object(question_bank, 'ADDITIONS_PATH', path):
                    with self.assertRaises(ValueError):
                        question_bank.load_additions()

    def test_new_answers_search_and_navigation(self):
        with TestClient(app) as client:
            self.assertEqual(client.get('/api/exercises/1100').json()['next_id'], 1101)
            final = client.get(f'/api/exercises/{question_bank.QUESTION_COUNT}').json()
            self.assertIsNone(final['next_id'])
            self.assertEqual(final['previous_id'], question_bank.QUESTION_COUNT - 1)
            for number, term in [(1101, 'MLM'), (1103, 'classification'), (1135, 'telescop')]:
                answer = client.get(f'/api/exercises/{number}/solution').json()
                self.assertIn(term.lower(), json.dumps(answer).lower())
                detail = client.get(f'/api/exercises/{number}').json()
                self.assertEqual(len(detail['resources']), 5)
                self.assertTrue(detail['source_url'].startswith('https://github.com/'))
            results = client.get('/api/search?q=ELECTRA&mode=keyword').json()
            self.assertIn(1107, [item['id'] for item in results['items']])
