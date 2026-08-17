"""Run the frontend unit tests under pytest, so ``pytest tests -q`` covers them.

The frontend has no build step and no package.json — adding one to run a
handful of pure-function tests would be a worse trade than shelling out to
Node's own test runner, which needs neither. When Node is missing the tests
skip rather than fail: a Python-only checkout must still be able to run the
suite (CLAUDE.md §5, degrade rather than crash).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SUITES = ["bar.test.js", "common.test.js"]


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not installed; the frontend tests need it")
@pytest.mark.parametrize("suite", SUITES)
def test_frontend_unit_suite_passes(suite: str):
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).parent / suite)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
