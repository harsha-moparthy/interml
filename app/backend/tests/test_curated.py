from placement.question_bank import QUESTION_COUNT
import json
from pathlib import Path
import tempfile
import unittest

from placement.curated import load_solutions


class CuratedSolutionsTests(unittest.TestCase):
    def test_complete_main_workbook_is_loadable(self):
        solutions = load_solutions()
        self.assertTrue(set(range(1, 601)) <= set(solutions))
        self.assertTrue(all(solutions[n]["code"].strip() for n in range(1, 601)))
        self.assertIn("def multi_head_attention", solutions[316]["code"])

    def test_authored_appendix_has_code_or_substantive_answers(self):
        solutions = load_solutions()
        for number in range(601, QUESTION_COUNT + 1):
            item = solutions[number]
            self.assertNotIn("A complete answer must", item["answer"])
            self.assertNotIn("A complete solution must", item["answer"])
            self.assertEqual(len(item["readings"]), 5)
            self.assertTrue(item["answer"].strip())
            if number <= 652:
                self.assertTrue(item["code"].strip())
            else:
                self.assertFalse(item["code"])
                self.assertGreaterEqual(len(item["answer"].split()), 60)
            self.assertRegex(item["answer"], r"\[\d\]\(https://")

    def test_code_path_cannot_escape_content_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "q001.json").write_text(
                json.dumps({"id": 1, "code_file": "../outside.py"})
            )
            with self.assertRaisesRegex(ValueError, "inside solutions"):
                load_solutions(root)

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("first", "second"):
                (root / f"{name}.json").write_text(
                    json.dumps({"id": 1, "answer": "An answer."})
                )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_solutions(root)

    def test_invalid_python_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "q001.py").write_text("def broken(:")
            (root / "q001.json").write_text(
                json.dumps({"id": 1, "code_file": "q001.py"})
            )
            with self.assertRaises(SyntaxError):
                load_solutions(root)

    def test_database_uses_curated_code(self):
        from placement import db

        with db.open_exercises() as connection:
            for number, solution in load_solutions().items():
                row = connection.execute(
                    "SELECT solution_code FROM exercise WHERE id=?", (number,)
                ).fetchone()
                self.assertEqual(row[0], solution["code"], number)
