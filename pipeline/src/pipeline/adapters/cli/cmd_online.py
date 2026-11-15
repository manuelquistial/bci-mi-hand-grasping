"""Online classification commands: online-sim, online-replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_online_replay(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    from pipeline.online.registry import OnlineModelRegistry
    from pipeline.online.replay_session import ReplayOnlineConfig, ReplayOnlineSession
    from pipeline.service import default_recordings_root

    log_path = Path(args.log) if args.log else (
        default_recordings_root().parent
        / "batch-results"
        / f"online-replay-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jsonl"
    )
    try:
        registry = OnlineModelRegistry()
        entry = registry.get(args.registry)
        model = registry.load_model(args.registry)
        dataset = str(entry.offline.get("dataset") or args.dataset)
        subject = (
            args.subject
            or str(entry.offline.get("online_subject") or "")
            or str(entry.offline.get("subject") or "")
            or None
        )
        session_dir = args.session_dir or str(entry.offline.get("session_dir") or "") or None
        n_train = entry.offline.get("n_train_subjects")
        print(
            f"Replay registry {entry.id} · dataset={dataset} · "
            f"online_subject={subject or '-'} · n_train={n_train} · "
            f"heads={list(model.classifier_names)}",
            flush=True,
        )
        summary = ReplayOnlineSession(
            model,
            ReplayOnlineConfig(
                dataset=dataset,
                subject=subject,
                session_dir=session_dir,
                max_trials=args.max_trials,
                log_path=log_path,
                dump_seconds=args.dump_seconds,
            ),
        ).run()
    except (TimeoutError, ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Log: {log_path}", flush=True)
    return 0


def cmd_online_sim(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    from pipeline.online.dataset_model import train_frozen_from_dataset
    from pipeline.online.model import load_frozen_model, save_frozen_model
    from pipeline.online.registry import OfflineModeSpec, OnlineModelRegistry
    from pipeline.online.simulator_session import (
        SimulatorOnlineConfig,
        SimulatorOnlineSession,
    )
    from pipeline.preprocess import FilterSpec
    from pipeline.service import default_recordings_root

    log_path = Path(args.log) if args.log else (
        default_recordings_root().parent
        / "batch-results"
        / f"online-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jsonl"
    )
    try:
        registry = OnlineModelRegistry()
        dataset = args.dataset
        subject = args.subject
        session_dir = args.session_dir
        if args.registry or args.registry_latest:
            if args.registry:
                entry = registry.get(args.registry)
                model = registry.load_model(args.registry)
            else:
                entry = registry.latest(dataset=args.dataset, subject=args.subject)
                model = registry.load_model(entry.id)
            dataset = str(entry.offline.get("dataset") or dataset)
            subject = (
                subject
                or str(entry.offline.get("online_subject") or "")
                or str(entry.offline.get("subject") or "")
                or None
            )
            session_dir = str(entry.offline.get("session_dir") or "") or session_dir
            print(
                f"Loaded registry {entry.id} · name={entry.name} · "
                f"online_subject={subject or '-'} · "
                f"n_train={entry.offline.get('n_train_subjects')} · "
                f"heads={list(model.classifier_names)}",
                flush=True,
            )
        elif args.model:
            model = load_frozen_model(args.model)
            print(
                f"Loaded frozen model from {args.model} · heads={list(model.classifier_names)}",
                flush=True,
            )
        else:
            classifiers = args.classifier or args.classifiers
            print(
                f"Training frozen [{classifiers}] on {dataset}"
                + (f" / {subject}" if subject else "")
                + "…",
                flush=True,
            )
            model = train_frozen_from_dataset(
                dataset=dataset,
                subject=subject,
                session_dir=session_dir,
                classifier_names=None if args.classifier else args.classifiers,
                classifier_name=args.classifier,
                control_classifier=args.control_classifier,
                filter=FilterSpec(
                    notch_hz=args.notch,
                    band_low_hz=args.band_low,
                    band_high_hz=args.band_high,
                    order=args.order,
                ),
                csp_pairs=args.csp_pairs,
            )
            offline = ", ".join(
                f"{name}={acc:.0%}" if acc is not None else f"{name}=n/a"
                for name, acc in model.offline_accuracies.items()
            )
            print(
                f"Offline train accuracy: {offline} · "
                f"{model.channel_count} ch · control={model.control_name} · {model.meta}",
                flush=True,
            )
            if args.save_model:
                saved = save_frozen_model(model, args.save_model)
                print(f"Wrote frozen model to {saved}", flush=True)
            if args.register:
                reg_name = args.register_as or (
                    f"{dataset}-{subject or 'all'}-online"
                )
                entry = registry.register(
                    model,
                    OfflineModeSpec(
                        name=reg_name,
                        dataset=dataset,
                        online_subject=subject,
                        subject=subject,
                        session_dir=session_dir,
                        classifiers=tuple(model.classifier_names),
                        control_classifier=model.control_name,
                        cross_subject=bool((model.meta or {}).get("cross_subject", True)),
                        notch_hz=args.notch,
                        band_low_hz=args.band_low,
                        band_high_hz=args.band_high,
                        order=args.order,
                        csp_pairs=args.csp_pairs,
                    ),
                )
                print(f"Registered offline export as {entry.id}", flush=True)

        config = SimulatorOnlineConfig(
            dataset=dataset,
            subject=subject,
            session_dir=session_dir,
            wait_seconds=args.wait,
            max_trials=args.max_trials,
            log_path=log_path,
        )
        print(
            f"Starting simulator playback → online windows → {list(model.classifier_names)}",
            flush=True,
        )
        summary = SimulatorOnlineSession(model, config).run()
    except (TimeoutError, ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Log: {log_path}", flush=True)
    return 0
