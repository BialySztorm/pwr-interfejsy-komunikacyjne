"""Run the full pipeline: sync_drive -> build_dataset -> classify (all models) -> evaluate.

Each stage is run as its own subprocess (same as invoking the scripts individually),
so a failure in one stage stops the pipeline with that stage's own error message.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, args: list[str]) -> None:
    print(f"\n=== {name} ===")
    result = subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"Step '{name}' failed with exit code {result.returncode}.", file=sys.stderr)
        raise SystemExit(result.returncode)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full sentiment benchmark pipeline.")
    parser.add_argument("--skip-sync", action="store_true", help="Skip the Google Drive sync step.")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device passed to classify.py."
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.skip_sync:
        run_step("sync_drive", ["scripts/sync_drive.py"])
    else:
        print("Skipping sync_drive (--skip-sync).")

    run_step("build_dataset", ["scripts/build_dataset.py"])
    run_step("classify (all models)", ["scripts/classify.py", "--model", "all", "--device", args.device])
    run_step("evaluate", ["scripts/evaluate.py"])

    print("\nPipeline finished. See results/summary.md for the comparison table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
