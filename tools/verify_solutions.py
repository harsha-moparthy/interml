"""Run original workbook regression suites against the curated standalone files.

Usage: app/.venv-ref/bin/python tools/verify_solutions.py
Requires the local (untracked) source workbook and its test fixtures.
All generated test reports stay in a temporary directory.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ml-interview" / "ml-coding-workbook"
CURATED = ROOT / "app" / "content" / "solutions"


def main():
    with tempfile.TemporaryDirectory(prefix="placement-solutions-") as temporary:
        parent = Path(temporary)
        workbook = parent / "ml-coding-workbook"
        workbook.mkdir()
        solutions = workbook / "solutions"
        solutions.mkdir()
        records = json.loads((SOURCE / "workbook.json").read_text())
        for record in records:
            suffix = ".py" if record["language"] == "Python" else ".cu"
            reviewed = CURATED / f"q{record['number']:03d}{suffix}"
            shutil.copyfile(reviewed, solutions / record["file"])
        for name in (
            "workbook.json",
            "verify_original.py",
            "verify_resume.py",
            "verify_bank.py",
            "resume_catalog.py",
        ):
            shutil.copyfile(SOURCE / name, workbook / name)
        bank = json.loads((SOURCE / "bank-map.json").read_text())
        # The upstream fixture stores Windows path separators.
        bank = {key: value.replace("\\", "/") for key, value in bank.items()}
        (workbook / "bank-map.json").write_text(json.dumps(bank))
        (workbook / "deep-ml-open").symlink_to(
            SOURCE / "deep-ml-open", target_is_directory=True
        )
        (parent / "ml-coding-catalog").symlink_to(
            SOURCE.parent / "ml-coding-catalog", target_is_directory=True
        )
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        for script in ("verify_original.py", "verify_resume.py", "verify_bank.py"):
            subprocess.run(
                [sys.executable, str(workbook / script)], check=True, env=env
            )
        # The bank runner only prints failures: explicitly propagate its report.
        bank_results = json.loads((workbook / "bank-test-results.json").read_text())
        if not bank_results or not all(row["passed"] for row in bank_results):
            raise SystemExit("Repository fixture tests are missing or failing")


if __name__ == "__main__":
    main()
