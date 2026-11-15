"""Argument parsers for batch scoring commands (batch, paper-csp)."""

from __future__ import annotations

import argparse

from pipeline.spatial import METHODS as SPATIAL_METHODS


def register_batch_commands(sub: argparse._SubParsersAction) -> None:
    batch_p = sub.add_parser(
        "batch",
        help="Score every classifier on one shared cohort per dataset",
    )
    batch_p.add_argument(
        "--datasets",
        default="shuqfa2024,liu2024_stroke",
        help="Comma-separated dataset ids (default: both)",
    )
    batch_p.add_argument(
        "--classifiers",
        default="",
        help="Comma-separated classifiers (default: all)",
    )
    batch_p.add_argument(
        "--features",
        default="ts,tdp,psd,csp,fbcsp",
        help="Comma-separated descriptors (default: ts,tdp,psd,csp,fbcsp)",
    )
    batch_p.add_argument(
        "--evaluations",
        default="nested",
        help="Comma-separated schemes (default: nested)",
    )
    batch_p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel folds and subject reads (default: CPU count)",
    )
    batch_p.add_argument(
        "--spatial",
        default=",".join(SPATIAL_METHODS),
        help="Comma-separated spatial methods (default: none,car,hjorth,large_laplacian,spherical_spline)",
    )
    batch_p.add_argument(
        "--alignment",
        default="none,euclidean",
        help="Comma-separated alignments (default: none,euclidean)",
    )
    batch_p.add_argument("--notch", type=float, default=50.0)
    batch_p.add_argument("--band-low", type=float, default=8.0)
    batch_p.add_argument("--band-high", type=float, default=30.0)
    batch_p.add_argument("--order", type=int, default=4)
    batch_p.add_argument(
        "--run",
        action="store_true",
        help="Train the nested comparison. Without this flag a nested batch only estimates the fits",
    )
    batch_p.add_argument(
        "--output",
        default=None,
        help="JSON path (default: batch-results/<timestamp>.json)",
    )

    paper_p = sub.add_parser(
        "paper-csp",
        help="Score MDM and TWFB after every spatial method, both alignments, and CSP",
    )
    paper_p.add_argument("--datasets", default="shuqfa2024,liu2024_stroke")
    paper_p.add_argument("--models", default="mdm,twfb_dgfmdr")
    paper_p.add_argument(
        "--spatial",
        default="none,car,hjorth,large_laplacian,spherical_spline",
    )
    paper_p.add_argument("--alignment", default="none,euclidean")
    paper_p.add_argument(
        "--output",
        default=None,
        help="JSON path (default: batch-results/paper-csp.json). The same path resumes.",
    )
    paper_p.add_argument(
        "--run",
        action="store_true",
        help="Train the models. Without this flag only the job list is printed",
    )
