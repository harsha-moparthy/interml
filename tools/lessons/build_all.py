"""Build lesson articles. Usage: python tools/lessons/build_all.py [module ...]   (module = file name without .py)"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lessons"))
import kit  # noqa: E402

names = sys.argv[1:] or sorted(p.stem for p in (HERE / "lessons").glob("*.py") if not p.stem.startswith("_"))
for name in names:
    mod = importlib.import_module(name)
    for article in mod.build():
        print(f"{name}: {article.id}: {kit.write(article)}")
