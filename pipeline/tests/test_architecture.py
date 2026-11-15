"""Architecture guards with a temporary allowlist of today's violations.

WP1–WP2 / WP6 must shrink ALLOWED_OVERSIZE and FORBIDDEN_IMPORTS to empty.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "pipeline"
MAX_LINES = 300

# Current god files — remove each entry only after that module is split.
ALLOWED_OVERSIZE: frozenset[str] = frozenset()

# Domain/online modules that still import the HTTP service module.
# Must stay empty after WP1: online modules must not import pipeline.service.
FORBIDDEN_SERVICE_IMPORTERS: frozenset[str] = frozenset()


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _owned_python_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.py")
        if "__pycache__" not in path.parts and path.name != "__init__.py"
    )


def test_owned_modules_respect_line_budget_or_allowlist() -> None:
    offenders: list[str] = []
    unexpected: list[str] = []
    for path in _owned_python_files():
        lines = path.read_text(encoding="utf-8").count("\n") + 1
        rel = _rel(path)
        if lines <= MAX_LINES:
            continue
        if rel in ALLOWED_OVERSIZE:
            continue
        offenders.append(f"{rel}={lines}")
    for allowed in sorted(ALLOWED_OVERSIZE):
        path = ROOT / allowed
        if not path.exists():
            unexpected.append(f"missing allowlist entry: {allowed}")
            continue
        lines = path.read_text(encoding="utf-8").count("\n") + 1
        if lines <= MAX_LINES:
            unexpected.append(f"allowlist stale (now ≤{MAX_LINES}): {allowed}={lines}")
    assert not offenders, "oversize modules not on allowlist:\n" + "\n".join(offenders)
    assert not unexpected, "allowlist needs update:\n" + "\n".join(unexpected)


def _imports_pipeline_service(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pipeline.service":
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pipeline.service":
                    return True
    return False


def test_online_service_imports_are_allowlisted_only() -> None:
    """online/* must not depend on service.py except for the known debt list."""
    found: set[str] = set()
    online = ROOT / "online"
    for path in online.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        if _imports_pipeline_service(path):
            found.add(_rel(path))
    assert found == set(FORBIDDEN_SERVICE_IMPORTERS), (
        f"online→service imports changed.\n"
        f"expected={sorted(FORBIDDEN_SERVICE_IMPORTERS)}\n"
        f"found={sorted(found)}"
    )


def test_named_ws_ports_live_in_config() -> None:
    from pipeline.config import CONTROL_WS_PORT, OBSERVE_WS_PORT

    assert CONTROL_WS_PORT == 8090
    assert OBSERVE_WS_PORT == 8091
    config_text = (ROOT / "config.py").read_text(encoding="utf-8")
    assert "CONTROL_WS_PORT = 8090" in config_text
    assert "OBSERVE_WS_PORT = 8091" in config_text
