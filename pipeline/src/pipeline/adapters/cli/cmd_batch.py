"""Batch scoring commands: batch, paper-csp."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.adapters.cli.common import csv_list


def cmd_batch(args: argparse.Namespace) -> int:
    import os
    from datetime import datetime, timezone

    from pipeline.batch import dry_run_estimate, run_batch
    from pipeline.service import PipelineService, default_recordings_root

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    output = args.output
    if not output:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = str(default_recordings_root().parent / "batch-results" / f"{stamp}.json")
    service = PipelineService(default_recordings_root())

    def report(row: dict) -> None:
        if row.get("error"):
            print(
                f"{row['dataset']} · {row.get('spatial') or 'none'} · "
                f"{row.get('alignment') or 'none'} · {row['features']} · {row['evaluation']} · "
                f"{row['classifier']} · {row['error']}",
                flush=True,
            )
            return
        accuracy = row.get("accuracy")
        shown = f"{accuracy:.0%}" if isinstance(accuracy, float) else ""
        print(
            f"{row['dataset']} · {row.get('spatial') or 'none'} · "
            f"{row.get('alignment') or 'none'} · {row['features']} · {row['evaluation']} · "
            f"{row['classifier']} · {shown} · {row['seconds']:.0f}s",
            flush=True,
        )

    destination = Path(output)
    evaluations = csv_list(args.evaluations)
    nested = any(item in {"nested", "nested_loso"} for item in evaluations)
    if nested and not args.run:
        try:
            estimate = dry_run_estimate(
                service,
                datasets=csv_list(args.datasets),
                classifiers=csv_list(args.classifiers),
                features=csv_list(args.features),
                evaluations=evaluations,
                spatial=csv_list(args.spatial),
                alignment=csv_list(args.alignment),
            )
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        estimate_path = destination.with_name(f"{destination.stem}-estimate.json")
        estimate_path.parent.mkdir(parents=True, exist_ok=True)
        estimate_path.write_text(json.dumps(estimate, indent=2), encoding="utf-8")
        print(
            f"Dry run: {estimate['total_fits']} fits across {estimate['pipelines']} pipelines.",
            flush=True,
        )
        print("No model was trained. Pass --run to start, and power the machine off when it finishes.", flush=True)
        print(f"Wrote the estimate to {estimate_path}", flush=True)
        return 0
    try:
        results = run_batch(
            service,
            datasets=csv_list(args.datasets),
            classifiers=csv_list(args.classifiers),
            features=csv_list(args.features),
            evaluations=evaluations,
            workers=workers,
            spatial=csv_list(args.spatial),
            alignment=csv_list(args.alignment),
            notch_hz=args.notch,
            band_low_hz=args.band_low,
            band_high_hz=args.band_high,
            order=args.order,
            on_result=report,
            output=destination,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    table = destination.with_name(f"{destination.stem}-comparison.csv")
    print(f"Wrote {len(results)} rows to {destination}", flush=True)
    print(f"Comparison, best accuracy first: {table}", flush=True)
    return 0


def cmd_paper_csp(args: argparse.Namespace) -> int:
    import os

    from pipeline.classify.paper_models import PAPER_MODELS, score_csp_grid
    from pipeline.service import PipelineService, default_recordings_root
    from pipeline.spatial import METHODS as SPATIAL_METHODS

    datasets = csv_list(args.datasets)
    models = csv_list(args.models)
    spatials = csv_list(args.spatial)
    alignments = csv_list(args.alignment)
    unknown_models = [name for name in models if name not in PAPER_MODELS]
    unknown_spatial = [name for name in spatials if name not in SPATIAL_METHODS]
    unknown_alignment = [name for name in alignments if name not in {"none", "euclidean"}]
    if unknown_models or unknown_spatial or unknown_alignment or not datasets:
        print("error: unknown dataset, model, spatial method, or alignment", file=sys.stderr)
        return 1
    variants = [(spatial, alignment) for spatial in spatials for alignment in alignments]
    jobs = len(datasets) * len(models) * len(variants)
    destination = Path(args.output) if args.output else default_recordings_root().parent / "batch-results" / "paper-csp.json"
    cores = os.cpu_count() or 1
    print(
        f"{jobs} jobs · {len(models)} models · {len(variants)} signals · {cores} cores, one BLAS thread each.",
        flush=True,
    )
    print(f"Results: {destination}", flush=True)
    print(f"Comparison: {destination.with_name(destination.stem + '-comparison.csv')}", flush=True)
    if not args.run:
        print("No model was trained. Pass --run to start.", flush=True)
        return 0
    service = PipelineService(default_recordings_root())
    try:
        for dataset in datasets:
            cohort = service.load_cohort(
                dataset=dataset,
                spatial="none",
                alignment="none",
            )
            score_csp_grid(
                cohort["groups"],
                float(cohort["rate"]),
                list(cohort["channels"]),
                destination,
                dataset=str(cohort["dataset"]),
                models=models,
                variants=variants,
            )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {destination}", flush=True)
    return 0
