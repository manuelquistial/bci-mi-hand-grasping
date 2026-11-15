"""Argument parsers for online and offline-export commands."""

from __future__ import annotations

import argparse


def _register_online_sim(sub: argparse._SubParsersAction) -> None:
    online_p = sub.add_parser(
        "online-sim",
        help="Play a dataset on the simulator and classify each imagery window online",
    )
    online_p.add_argument(
        "--dataset",
        default="shuqfa2024",
        help="Dataset id for the simulator (default: shuqfa2024)",
    )
    online_p.add_argument("--subject", default=None, help="Participant id (default: first listed)")
    online_p.add_argument(
        "--session-dir",
        default=None,
        help="Recorded session folder when dataset=recorded_session",
    )
    online_p.add_argument(
        "--classifiers",
        default="linear_svm,logistic_regression",
        help=(
            "Comma-separated frozen heads scored on every window "
            "(default: linear_svm,logistic_regression)"
        ),
    )
    online_p.add_argument(
        "--classifier",
        default=None,
        choices=("linear_svm", "logistic_regression"),
        help="Single classifier (alias). Overrides --classifiers when set",
    )
    online_p.add_argument(
        "--control-classifier",
        default=None,
        help="Which head drives the stability gate / avatar (default: first listed)",
    )
    online_p.add_argument(
        "--model",
        default=None,
        help="Path to a saved frozen model (.joblib). When omitted, train from the dataset",
    )
    online_p.add_argument(
        "--registry",
        default=None,
        help="Load a registered offline export by id or name (from pipeline offline-export)",
    )
    online_p.add_argument(
        "--registry-latest",
        action="store_true",
        help="Load the latest registered offline export (optionally filtered by --dataset/--subject)",
    )
    online_p.add_argument(
        "--save-model",
        default=None,
        help="Where to write the frozen model after training",
    )
    online_p.add_argument(
        "--register",
        action="store_true",
        help="After training, also register the model under online-models/ for later online runs",
    )
    online_p.add_argument(
        "--register-as",
        default=None,
        help="Name used when --register is set (default: <dataset>-<subject>-online)",
    )
    online_p.add_argument("--max-trials", type=int, default=None, help="Stop after N hand trials")
    online_p.add_argument(
        "--log",
        default=None,
        help="JSONL path for per-window decisions (default: batch-results/online-<stamp>.jsonl)",
    )
    online_p.add_argument("--wait", type=float, default=8.0, help="LSL discover timeout")
    online_p.add_argument("--notch", type=float, default=50.0)
    online_p.add_argument("--band-low", type=float, default=8.0)
    online_p.add_argument("--band-high", type=float, default=30.0)
    online_p.add_argument("--order", type=int, default=4)
    online_p.add_argument("--csp-pairs", type=int, default=2)


def _register_online_replay(sub: argparse._SubParsersAction) -> None:
    replay_p = sub.add_parser(
        "online-replay",
        help="Classify a dataset file online without LSL (Slurm / headless)",
    )
    replay_p.add_argument("--dataset", default="shuqfa2024")
    replay_p.add_argument("--subject", default=None)
    replay_p.add_argument("--session-dir", default=None)
    replay_p.add_argument(
        "--registry",
        required=True,
        help="Registered offline export id or name",
    )
    replay_p.add_argument("--max-trials", type=int, default=None)
    replay_p.add_argument("--log", default=None)
    replay_p.add_argument(
        "--dump-seconds",
        type=float,
        default=200.0,
        help="How many seconds of the recording to replay (default: 200)",
    )


def _register_offline_export(sub: argparse._SubParsersAction) -> None:
    export_p = sub.add_parser(
        "offline-export",
        help="Train offline mode(s) and register frozen models for online-sim",
    )
    export_p.add_argument(
        "--config",
        default=None,
        help="JSON file with one mode or {modes: [...]} offline recipes",
    )
    export_p.add_argument("--name", default=None, help="Mode name when not using --config")
    export_p.add_argument("--dataset", default=None)
    export_p.add_argument(
        "--subject",
        default=None,
        help="Legacy alias for --online-subject (verification subject only)",
    )
    export_p.add_argument(
        "--online-subject",
        default=None,
        help="Subject used later for online verification (training uses all subjects)",
    )
    export_p.add_argument("--session-dir", default=None)
    export_p.add_argument(
        "--classifiers",
        default="linear_svm,logistic_regression",
        help="Comma-separated classifiers (default: linear_svm,logistic_regression)",
    )
    export_p.add_argument("--control-classifier", default=None)
    export_p.add_argument(
        "--cross-subject",
        dest="cross_subject",
        action="store_true",
        default=True,
        help="Train on all subjects (default)",
    )
    export_p.add_argument(
        "--single-subject",
        dest="cross_subject",
        action="store_false",
        help="Legacy smoke: train on --online-subject / --subject only",
    )
    export_p.add_argument("--notch", type=float, default=50.0)
    export_p.add_argument("--band-low", type=float, default=8.0)
    export_p.add_argument("--band-high", type=float, default=30.0)
    export_p.add_argument("--order", type=int, default=4)
    export_p.add_argument("--csp-pairs", type=int, default=2)
    export_p.add_argument(
        "--registry-root",
        default=None,
        help="Registry directory (default: pipeline/online-models)",
    )
    export_p.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite an existing registry id with the same name stamp collision",
    )


def _register_offline_list(sub: argparse._SubParsersAction) -> None:
    list_p = sub.add_parser(
        "offline-list",
        help="List offline models registered for online use",
    )
    list_p.add_argument(
        "--registry-root",
        default=None,
        help="Registry directory (default: pipeline/online-models)",
    )
    list_p.add_argument("--dataset", default=None, help="Filter by dataset id")
    list_p.add_argument("--subject", default=None, help="Filter by subject id")


def register_online_commands(sub: argparse._SubParsersAction) -> None:
    _register_online_sim(sub)
    _register_online_replay(sub)
    _register_offline_export(sub)
    _register_offline_list(sub)
