import subprocess
import sys
import tempfile
from pathlib import Path


def q382(trusted_solution, trusted_tests, timeout=3):
    """Run trusted local Python solution and assertion strings in a temporary subprocess.
    Return passed, stdout, stderr, and timed_out. A timeout is not a security sandbox:
    inputs must already be trusted.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "check.py"
        path.write_text(trusted_solution + "\n" + trusted_tests, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "passed": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "stdout": "",
                "stderr": "timeout",
                "timed_out": True,
            }
