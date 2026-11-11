"""Anonymize generated audio for blind listening evaluation.

Copies every data/raw/<system>/<person>/<file>.wav into data/anonymized/ under a
random id that reveals nothing about the source system or text, and writes the
de-anonymization key to results/mapping-NNN.csv (gitignored — never hand this to
raters). This fulfils the brief's "nagrania ... zanonimizowane ze względu na TTS
źródłowy, aby ograniczyć wpływ oczekiwań oceniających" requirement.

Run again with --force to re-shuffle (e.g. after generating more systems) — old
anonymized files and their mapping are kept; a new mapping-NNN.csv is written
each time so past listening sessions stay reproducible/traceable.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_corpus import PROJECT_ROOT  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
ANONYMIZED_DIR = PROJECT_ROOT / "data" / "anonymized"
RESULTS_DIR = PROJECT_ROOT / "results"


def next_run_number(results_dir: Path) -> int:
    pattern = re.compile(r"^mapping-(\d{3})\.csv$")
    existing = [
        int(match.group(1))
        for path in results_dir.glob("mapping-*.csv")
        if (match := pattern.match(path.name))
    ]
    return max(existing, default=0) + 1


def find_source_wavs() -> list[tuple[str, str, Path]]:
    """Returns (system, text_id, path) for every data/raw/<system>/<person>/<file>.wav."""
    found = []
    for system_dir in sorted(p for p in DATA_RAW_DIR.glob("*") if p.is_dir()):
        system = system_dir.name
        for wav_path in sorted(system_dir.glob("*/*.wav")):
            text_id = f"{wav_path.parent.name}/{wav_path.stem}"
            found.append((system, text_id, wav_path))
    return found


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy data/raw/**/*.wav into data/anonymized/ under random ids."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-copy even if an anonymized file for this (system, text_id) already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for id generation (mainly for reproducible test runs).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)

    sources = find_source_wavs()
    if not sources:
        print(
            f"No .wav files found under {DATA_RAW_DIR}. Run scripts/generate_speech.py first.",
            file=sys.stderr,
        )
        return 1

    random.shuffle(sources)

    ANONYMIZED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_suffix = f"{next_run_number(RESULTS_DIR):03d}"
    mapping_path = RESULTS_DIR / f"mapping-{run_suffix}.csv"

    rows = []
    for system, text_id, source_path in sources:
        anon_id = uuid.uuid4().hex[:8]
        destination = ANONYMIZED_DIR / f"{anon_id}.wav"
        while destination.exists() and not args.force:
            anon_id = uuid.uuid4().hex[:8]
            destination = ANONYMIZED_DIR / f"{anon_id}.wav"

        shutil.copyfile(source_path, destination)
        rows.append({"anon_id": anon_id, "system": system, "text_id": text_id})
        print(f"{source_path} -> {destination.name}")

    with mapping_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["anon_id", "system", "text_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAnonymized {len(rows)} files.")
    print(f"Mapping saved (keep private from raters!): {mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
