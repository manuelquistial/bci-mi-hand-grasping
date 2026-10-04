from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def bci_sample(fixtures_dir: Path) -> Path:
    return fixtures_dir / "bci_iv2a" / "A01T.txt"


@pytest.fixture
def physionet_sample(fixtures_dir: Path) -> Path:
    return fixtures_dir / "physionet_eegbci" / "S001.txt"


@pytest.fixture
def generic_sample(fixtures_dir: Path) -> Path:
    return fixtures_dir / "generic_sample.txt"
