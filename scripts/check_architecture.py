#!/usr/bin/env python3
"""Offline architecture gates (stdlib only).

1. Line budget: every owned production .py/.ts/.tsx file has <= 300 lines.
   Excluded: tests, venv, node_modules, __pycache__, .next, dist, build, data.
2. Cycle ban: no pipeline ``online/*`` module imports ``pipeline.service``.

Exit status: 0 = all gates pass, 1 = at least one violation.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_LINES = 300
SUFFIXES = {".py", ".ts", ".tsx"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".next",
    "dist",
    "build",
    "out",
    "coverage",
    "tests",
    "test",
    "__tests__",
    "data",
    "recordings",
    "batch-results",
    "online-models",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
ONLINE_DIR = REPO / "pipeline" / "src" / "pipeline" / "online"


def is_test_file(path: Path) -> bool:
    name = path.name
    return (
        name.startswith("test_")
        or name == "conftest.py"
        or ".test." in name
        or ".spec." in name
        or name.endswith(".d.ts")
    )


def owned_files() -> list[Path]:
    found: list[Path] = []
    for path in REPO.rglob("*"):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        parts = path.relative_to(REPO).parts[:-1]
        if any(part in SKIP_DIRS for part in parts) or is_test_file(path):
            continue
        found.append(path)
    return sorted(found)


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def check_line_budget(files: list[Path]) -> list[str]:
    return [
        f"{path.relative_to(REPO).as_posix()}: {count} lines (max {MAX_LINES})"
        for path in files
        if (count := line_count(path)) > MAX_LINES
    ]


def imports_service(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "pipeline.service" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = {alias.name for alias in node.names}
            if node.level == 0:
                if module == "pipeline.service" or (
                    module == "pipeline" and "service" in names
                ):
                    return True
            elif module == "service" or (module == "" and "service" in names):
                # Relative import of the top-level service module from online/*.
                # level 2 (``..service``) resolves to pipeline.service.
                if node.level == 2:
                    return True
    return False


def check_online_cycle_ban() -> list[str]:
    if not ONLINE_DIR.is_dir():
        return [f"missing directory: {ONLINE_DIR.relative_to(REPO).as_posix()}"]
    return [
        f"{path.relative_to(REPO).as_posix()}: imports pipeline.service"
        for path in sorted(ONLINE_DIR.rglob("*.py"))
        if "__pycache__" not in path.parts and imports_service(path)
    ]


def main() -> int:
    files = owned_files()
    failures = {
        f"line budget (<= {MAX_LINES})": check_line_budget(files),
        "online -> service import ban": check_online_cycle_ban(),
    }
    failed = False
    for gate, problems in failures.items():
        if problems:
            failed = True
            print(f"FAIL  {gate}")
            for problem in problems:
                print(f"      {problem}")
        else:
            print(f"OK    {gate}")
    print(f"checked {len(files)} owned production files")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
