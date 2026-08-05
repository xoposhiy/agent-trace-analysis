"""Run the bar.js unit tests under pytest, so ``pytest tests -q`` covers them.

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

BAR_TEST = Path(__file__).parent / "bar.test.js"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not installed; bar.js tests need it")
def test_bar_js_layout_suite_passes():
    result = subprocess.run(
        ["node", "--test", str(BAR_TEST)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
