"""The pure/IO split, enforced rather than documented.

Business logic lives in pure functions so it can be tested without a network.
A rule that is only written down lasts about three weeks, so this test walks
the AST of every module that claims to be pure and fails if one of them grows
an import that could reach the outside world.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PURE_MODULES = [
    "types",
    "questions",
    "chunking",
    "values",
    "coalitions",
    "estimators",
    "faithfulness",
    "calibration",
    "thresholds",
    "drift",
    "budget",
    "ratelimit",
]

FORBIDDEN = {
    "asyncio",
    "httpx",
    "httpx2",
    "typesafe_sdk",
    "sqlite3",
    "socket",
    "urllib",
    "requests",
    "subprocess",
    "os",
    "pathlib",
    "shutil",
    "tempfile",
    "threading",
    "multiprocessing",
    "time",
}

PACKAGE = pathlib.Path(__file__).parent


def _imported_roots(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("module", PURE_MODULES)
def test_pure_modules_cannot_reach_the_outside_world(module: str) -> None:
    path = PACKAGE / f"{module}.py"
    if not path.exists():
        pytest.skip(f"{module} not implemented yet")
    offending = _imported_roots(path) & FORBIDDEN
    assert not offending, (
        f"{module}.py imports {sorted(offending)}. Pure modules must stay testable "
        f"with no network, no clock and no filesystem -- move the I/O to client.py, "
        f"executor.py or cache.py and inject the result."
    )


def test_importing_the_package_does_not_pull_matplotlib() -> None:
    """Rendering is an optional extra. Importing jev_why must work, and stay
    fast, for someone who never draws a diagram."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import sys, jev_why; assert 'matplotlib' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
