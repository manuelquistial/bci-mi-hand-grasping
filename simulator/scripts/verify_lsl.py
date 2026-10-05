"""Verify that an EEG LSL stream is visible and delivering samples.

Usage (terminal 1 — leave running):
  uv run python scripts/verify_lsl.py

Then (terminal 2):
  uv run simopenbci --dataset shuqfa2024 --subject S001 --run 2 --no-dialog
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Listen for an EEG LSL stream and print samples.")
    parser.add_argument(
        "--name",
        default=None,
        help="Optional stream name filter (e.g. Shuqfa2024_EEG, Liu2024_Stroke_MI)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for a stream to appear (default: 30)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="How many samples to print before exiting (default: 10)",
    )
    args = parser.parse_args()

    from pylsl import StreamInlet, resolve_byprop, resolve_streams

    print(f"Waiting up to {args.timeout}s for an EEG LSL stream...")
    if args.name:
        streams = resolve_byprop("name", args.name, timeout=args.timeout)
    else:
        streams = resolve_byprop("type", "EEG", timeout=args.timeout)

    if not streams:
        # Fallback: show whatever is on the network
        all_streams = resolve_streams(wait_time=1.0)
        if all_streams:
            print("No EEG stream matched. Streams currently visible:")
            for s in all_streams:
                print(f"  - {s.name()} type={s.type()} ch={s.channel_count()} @ {s.nominal_srate()} Hz")
        else:
            print("No LSL streams found. Start the simulator in another terminal, then retry.")
        return 1

    info = streams[0]
    print(
        f"Found: name={info.name()!r} type={info.type()} "
        f"channels={info.channel_count()} rate={info.nominal_srate()} Hz"
    )
    inlet = StreamInlet(info)
    got = 0
    while got < args.samples:
        sample, ts = inlet.pull_sample(timeout=5.0)
        if sample is None:
            print("Timed out waiting for samples (is the simulator still sending?).")
            return 1
        got += 1
        preview = ", ".join(f"{v:.3f}" for v in sample[: min(4, len(sample))])
        more = "..." if len(sample) > 4 else ""
        print(f"  sample {got}: [{preview}{more}]  t={ts:.4f}")

    print(f"OK — received {got} samples from {info.name()!r}. LSL is working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
