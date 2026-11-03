"""Load reviewed solutions independently of the immutable imported workbook."""

import ast
import json
from pathlib import Path

from . import config
from .question_bank import QUESTION_COUNT
from .resources import Reading

DIRECTORY = config.APP_DIR / "content" / "solutions"


def load_solutions(directory: Path = DIRECTORY) -> dict[int, dict]:
    """Fail the import on malformed content instead of silently losing a review."""
    result = {}
    for path in sorted(directory.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        number = item["id"]
        if type(number) is not int or not 1 <= number <= QUESTION_COUNT or number in result:
            raise ValueError(f"{path}: invalid or duplicate exercise id")
        code_path = item.get("code_file")
        code = ""
        if code_path:
            source = (directory / code_path).resolve()
            if not source.is_relative_to(directory.resolve()):
                raise ValueError(f"{path}: code_file must stay inside solutions")
            code = source.read_text(encoding="utf-8")
            if source.suffix == ".py":
                ast.parse(code, filename=str(source))
        answer = item.get("answer", "").strip()
        if not code.strip() and not answer:
            raise ValueError(f"{path}: empty solution")
        readings = tuple(Reading(**r) for r in item.get("readings", []))
        expected = 5 if number >= 601 else 2
        if readings and (len(readings) != expected or len({r.url for r in readings}) != expected):
            raise ValueError(f"{path}: provide {expected} distinct, relevant readings")
        result[number] = {"code": code, "answer": answer, "readings": readings}
    return result
