"""Audit exact DANI row geometry without requesting any image asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_image_detector.dani_geometry import DEFAULT_VIEWER_ROWS_ENDPOINT, scan_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preselection_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_VIEWER_ROWS_ENDPOINT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--min-request-interval", type=float, default=0.2)
    parser.add_argument("--max-viewer-length", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def progress(completed: int, total: int) -> None:
        print(f"geometry rows: {completed}/{total}", flush=True)

    report = scan_geometry(
        args.preselection_dir,
        args.output_dir,
        endpoint=args.endpoint,
        workers=args.workers,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        min_request_interval=args.min_request_interval,
        max_viewer_length=args.max_viewer_length,
        progress=progress,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
