"""Offline registry commands: offline-export, offline-list."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_offline_export(args: argparse.Namespace) -> int:
    from pipeline.online.registry import (
        OfflineModeSpec,
        OnlineModelRegistry,
        export_offline_modes,
        load_offline_modes,
    )

    try:
        if args.config:
            modes = load_offline_modes(args.config)
        else:
            if not args.dataset or not args.name:
                print(
                    "error: provide --config, or both --name and --dataset",
                    file=sys.stderr,
                )
                return 2
            online_subject = args.online_subject or args.subject
            modes = [
                OfflineModeSpec(
                    name=args.name,
                    dataset=args.dataset,
                    online_subject=online_subject,
                    subject=online_subject,
                    session_dir=args.session_dir,
                    classifiers=tuple(
                        part.strip()
                        for part in str(args.classifiers).split(",")
                        if part.strip()
                    ),
                    control_classifier=args.control_classifier,
                    cross_subject=bool(args.cross_subject),
                    hold_out_online_subject=bool(args.cross_subject),
                    notch_hz=args.notch,
                    band_low_hz=args.band_low,
                    band_high_hz=args.band_high,
                    order=args.order,
                    csp_pairs=args.csp_pairs,
                )
            ]
        registry = OnlineModelRegistry(
            Path(args.registry_root) if args.registry_root else None
        )
        print(f"Exporting {len(modes)} offline mode(s) → {registry.root}", flush=True)
        entries = export_offline_modes(modes, registry=registry, replace=args.replace)
    except (TimeoutError, ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for entry in entries:
        offline = ", ".join(
            f"{name}={acc:.0%}" if acc is not None else f"{name}=n/a"
            for name, acc in entry.offline_accuracies.items()
        )
        n_train = entry.offline.get("n_train_subjects")
        verify = entry.offline.get("online_subject") or entry.offline.get("subject") or "-"
        held = entry.offline.get("held_out_online_subject")
        print(
            f"registered {entry.id} · name={entry.name} · "
            f"dataset={entry.offline.get('dataset')} · "
            f"cross_subject={entry.offline.get('cross_subject')} · "
            f"held_out={held} · n_train={n_train} · online_subject={verify} · "
            f"heads={entry.classifiers} · offline=[{offline}]",
            flush=True,
        )
    print(
        "Use: uv run pipeline online-sim --registry <id-or-name>",
        flush=True,
    )
    return 0


def cmd_offline_list(args: argparse.Namespace) -> int:
    from pipeline.online.registry import OnlineModelRegistry

    registry = OnlineModelRegistry(Path(args.registry_root) if args.registry_root else None)
    entries = registry.list_entries()
    if args.dataset:
        entries = [
            item for item in entries if str(item.offline.get("dataset") or "") == args.dataset
        ]
    if args.subject is not None:
        entries = [
            item
            for item in entries
            if str(
                item.offline.get("online_subject") or item.offline.get("subject") or ""
            )
            == str(args.subject)
        ]
    if not entries:
        print("No registered offline models.")
        return 0
    for entry in entries:
        offline = ", ".join(
            f"{name}={acc:.0%}" if acc is not None else f"{name}=n/a"
            for name, acc in entry.offline_accuracies.items()
        )
        verify = entry.offline.get("online_subject") or entry.offline.get("subject") or "-"
        n_train = entry.offline.get("n_train_subjects")
        xs = "xs" if entry.offline.get("cross_subject") else "ss"
        print(
            f"{entry.id}\t{entry.name}\t"
            f"{entry.offline.get('dataset')}/online={verify}"
            f"{f'/n_train={n_train}' if n_train is not None else ''}/{xs}\t"
            f"{','.join(entry.classifiers)}\t[{offline}]"
        )
    return 0
