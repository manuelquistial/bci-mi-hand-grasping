"""Tests for concrete dataset loaders."""
from pathlib import Path

from dataset_loaders import BCICompetitionIV2aLoader


def test_bci_loader_skips_time_column(bci_sample: Path):
    loader = BCICompetitionIV2aLoader()
    samples = list(loader.iter_samples(bci_sample))
    assert len(samples) == 3
    assert len(samples[0]) == 22
    assert samples[0][0] == 1.0
    assert samples[0][-1] == 22.0


def test_bci_loader_channel_subset(bci_sample: Path):
    loader = BCICompetitionIV2aLoader(n_channels=8)
    samples = list(loader.iter_samples(bci_sample))
    assert len(samples[0]) == 8
    assert samples[0] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert loader.metadata.channel_names == [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3",
    ]

