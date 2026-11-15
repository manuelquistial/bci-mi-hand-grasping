#!/usr/bin/env python3
"""Compare control strategies A–D on identical synthetic margin streams.

This does not retrain the frozen model. It evaluates decoder behavior on the
same margin sequences so strategies can be compared fairly without held-out
parameter tuning.
"""

from __future__ import annotations

import json
from collections import Counter

from pipeline.online.control_decoder import ControlDecoder, ControlDecoderConfig

# Synthetic trial margins: cue right, then left, with ambiguous middle windows.
TRIALS = [
    {"cue": "right", "margins": [0.8, 1.1, 0.9, 0.7, 0.6]},
    {"cue": "left", "margins": [-0.2, 0.4, -0.1, -0.6, -0.9]},  # ambiguous early
    {"cue": "left", "margins": [-1.0, -0.8, -0.7, -0.5, -0.4]},
    {"cue": "right", "margins": [0.1, -0.3, 0.2, 0.5, 0.8]},  # ambiguous early
]


def fused_direction(controls: list[float]) -> str | None:
    if not controls:
        return None
    mean = sum(controls) / len(controls)
    if mean > 0:
        return "right"
    if mean < 0:
        return "left"
    return None


def run_strategy(name: str, **kwargs) -> dict:
    dec = ControlDecoder(ControlDecoderConfig(strategy=name, **kwargs))  # type: ignore[arg-type]
    correct = 0
    rows = []
    for trial_i, trial in enumerate(TRIALS):
        dec.reset_trial()
        controls = []
        for margin in trial["margins"]:
            label = "right" if margin >= 0 else "left"
            out = dec.decode(label=label, score=abs(margin), decision_margin=margin)
            if out.valid:
                controls.append(out.control)
        fused = fused_direction(controls)
        ok = fused == trial["cue"]
        correct += int(ok)
        rows.append(
            {
                "trial": trial_i,
                "cue": trial["cue"],
                "fused": fused,
                "correct": ok,
                "mean_control": (sum(controls) / len(controls)) if controls else 0.0,
                "n_commands": len(controls),
            }
        )
    return {
        "strategy": name,
        "kwargs": kwargs,
        "trial_accuracy": correct / len(TRIALS),
        "n_correct": correct,
        "n_trials": len(TRIALS),
        "rows": rows,
    }


def main() -> None:
    results = [
        run_strategy("baseline_proba"),
        run_strategy("discrete"),
        run_strategy("temporal_discrete", window_k=1),
        run_strategy("temporal_discrete", window_k=2),
        run_strategy("temporal_discrete", window_k=3),
        run_strategy("temporal_proportional", window_k=1, proportional_scale=1.0),
        run_strategy("temporal_proportional", window_k=2, proportional_scale=1.0),
        run_strategy("temporal_proportional", window_k=3, proportional_scale=1.0),
    ]
    print(json.dumps(results, indent=2))
    print("\nSummary")
    for item in results:
        print(
            f"{item['strategy']:22} k={item['kwargs'].get('window_k', '-')} "
            f"acc={item['trial_accuracy']:.2f} ({item['n_correct']}/{item['n_trials']})"
        )


if __name__ == "__main__":
    main()
