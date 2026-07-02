#!/usr/bin/env python3
"""Duplicate a clean manifest into named SNR evaluation conditions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from asr_noise_robust.manifest import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+", default=["clean", "snr_20", "snr_10", "snr_5"])
    args = parser.parse_args()

    rows = read_manifest(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "utt_id",
            "audio_path",
            "duration",
            "text",
            "normalized_text",
            "split",
            "condition",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for condition in args.conditions:
                writer.writerow(
                    {
                        "utt_id": row.utt_id,
                        "audio_path": row.audio_path,
                        "duration": f"{row.duration:.6f}",
                        "text": row.text,
                        "normalized_text": row.normalized_text,
                        "split": row.split,
                        "condition": condition,
                    }
                )
    print(f"wrote SNR condition manifest to {args.output}")


if __name__ == "__main__":
    main()

