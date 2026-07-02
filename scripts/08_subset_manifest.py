#!/usr/bin/env python3
"""Write the first N rows of a manifest to a new manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_noise_robust.manifest_ops import write_manifest_subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int)
    args = parser.parse_args()

    count = write_manifest_subset(args.input, args.output, args.limit)
    print(f"wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()

