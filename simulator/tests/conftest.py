from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def generic_sample(fixtures_dir: Path) -> Path:
    return fixtures_dir / "generic_sample.txt"
