"""Pins the import package name to the directory name.

These drifted once already: ``pyproject.toml`` declared ``tracelens`` while the
directory on disk was ``Final_app``, so ``pip install -e .`` matched zero
packages and the whole suite died at collection with ``ModuleNotFoundError``.
Nothing caught it, because a suite that cannot be imported reports no failures.

The distribution name stays ``tracelens`` (that is the product); only the import
path follows the folder.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import Final_app

PACKAGE_ROOT = Path(Final_app.__file__).parent


# ----------------------------------------------------------------------
# Package name
# ----------------------------------------------------------------------


def test_import_package_name_matches_the_directory_on_disk() -> None:
    assert Final_app.__name__ == PACKAGE_ROOT.name


def test_every_submodule_is_importable_under_the_package_name() -> None:
    from Final_app.adapters import claude_code
    from Final_app.analysis import blocks, classify
    from Final_app.api import app as api
    from Final_app.ir import models
    from Final_app.judge import summary

    modules = [claude_code, blocks, classify, api, models, summary]
    assert [module.__name__.split(".")[0] for module in modules] == [
        Final_app.__name__
    ] * 6


# ----------------------------------------------------------------------
# Build configuration
# ----------------------------------------------------------------------


def test_setuptools_include_glob_matches_the_package_directory() -> None:
    """The glob in ``pyproject.toml`` must actually match, or the wheel is empty."""
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    globs = re.search(r'include\s*=\s*\[([^\]]*)\]', pyproject).group(1)
    patterns = re.findall(r'"([^"]+)"', globs)

    assert patterns, "no include patterns found in pyproject.toml"
    assert any(fnmatch.fnmatch(PACKAGE_ROOT.name, pattern) for pattern in patterns)
