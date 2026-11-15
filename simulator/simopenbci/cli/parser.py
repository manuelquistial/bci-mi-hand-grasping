"""Argument parser for the SimOpenBCI command line."""
from __future__ import annotations

import argparse

from dataset_loaders import DATASET_IDS

# Default dataset when starting without --file / --pick-file
DEFAULT_DATASET = "shuqfa2024"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SimOpenBCI: stream EEG data from a dataset to LSL. "
            f"With no args, streams the first subject of {DEFAULT_DATASET} "
            "(data/<dataset>/)."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_IDS),
        default=None,
        help=f"Dataset type (default: {DEFAULT_DATASET} when not using --file/--pick-file)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="EEG file path (optional; prefer --dataset/--subject)",
    )
    parser.add_argument(
        "--pick-file",
        action="store_true",
        help="Open a file dialog to choose an EEG file (instead of using a dataset subject)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after sending this many samples (for testing; default: send all)",
    )
    parser.add_argument(
        "--no-dialog",
        action="store_true",
        help="Do not show GUI finish/error dialogs (console only)",
    )
    parser.add_argument(
        "--lsl-name",
        type=str,
        default=None,
        help="LSL stream name (default: from dataset). Use e.g. OpenBCI_EEG or BioSemi to connect to articulo_1.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help="Use only first N channels (e.g. 8 for articulo_1).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root folder of the dataset (for --subject/--list-subjects).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Subject ID (e.g. A01, S001). Use with --dataset and optional --run. Overrides --file.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run/session (e.g. T, E). Only with --subject.",
    )
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="List available subjects and runs for the dataset, then exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --list-subjects, print subjects as JSON.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print the datasets this simulator can play as JSON, then exit.",
    )
    parser.add_argument(
        "--dump-seconds",
        type=float,
        default=None,
        help="Print the first seconds of the resolved file as JSON and exit. No LSL.",
    )
    parser.add_argument(
        "--dump-trials",
        action="store_true",
        help="Print each marker's following window as JSON and exit. No LSL.",
    )
    parser.add_argument(
        "--trial-seconds",
        type=float,
        default=4.0,
        help="Length of each window printed by --dump-trials.",
    )
    parser.add_argument(
        "--events",
        type=str,
        default=None,
        help=(
            "Path to trial/events CSV (sample_index,code,label[,kind]). "
            "Default: sidecar <eeg_file>.events.csv next to the EEG file."
        ),
    )
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Do not open an LSL Markers stream even if an events file is present.",
    )
    return parser
