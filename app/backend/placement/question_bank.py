"""Validated additions to the immutable 1..1100 question bank."""
import json
from . import config

ADDITIONS_PATH = config.APP_DIR / "content" / "interview_additions.json"


def load_additions():
    items = json.loads(ADDITIONS_PATH.read_text())
    if not isinstance(items, list):
        raise ValueError("interview additions must be a list")
    for expected, item in enumerate(items, 1101):
        if type(item.get("id")) is not int or item["id"] != expected:
            raise ValueError("interview additions must have contiguous IDs starting at 1101")
        if type(item.get("chapter_number")) is not int or not 1 <= item["chapter_number"] <= 18:
            raise ValueError("interview addition has invalid appendix chapter")
        for field in ("title", "prompt", "source_url", "repository_url", "discovery_note"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"interview addition missing {field}")
        if not item["source_url"].startswith(item["repository_url"] + "/blob/"):
            raise ValueError("interview discovery source must be a repository file")
    return items


QUESTION_COUNT = 1100 + len(load_additions())
