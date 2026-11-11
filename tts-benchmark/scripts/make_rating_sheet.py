"""Build an empty Likert rating sheet for a blind listening session.

Reads the most recent (or --mapping-run) results/mapping-NNN.csv and emits
results/oceny_tts-NNN_template.csv with one row per anonymized file and empty
columns for every criterion that's scored by ear:

  - kryt2_zrozumialosc .. kryt8_przydatnosc: Likert 1-5 (criteria 2-6, 8)
  - bledy_* columns: free-text error log for criterion 1 (poprawność wymowy),
    one column per error subtype named in the brief
  - czas_generowania is NOT here — that's objective and already in latency_tts-NNN.csv

Only `anon_id` is filled in (never `system` or `text_id` — raters must not see
those). Fill this out by listening to data/anonymized/<anon_id>.wav, save it as
results/oceny_tts-NNN.csv (drop the _template suffix), then run
aggregate_ratings.py.

Per the brief, each 2-3 person group picks 10 of their own texts + 20 shared
ones for the full listening evaluation — use --text-ids to restrict the sheet
to that subset (pass text ids as "<person>/<nrindeksu>-<N>-<slug>", matching
the `text_id` column of results/mapping-NNN.csv).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_corpus import PROJECT_ROOT  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

RESULTS_DIR = PROJECT_ROOT / "results"

LIKERT_CRITERIA = [
    "kryt2_zrozumialosc",
    "kryt3_naturalnosc",
    "kryt4_intonacja_prozodia",
    "kryt5_plynnosc",
    "kryt6_stabilnosc",
    "kryt8_przydatnosc_w_dialogu",
]

ERROR_LOG_COLUMNS = [
    "kryt1_bledy_liczby_daty_godziny",
    "kryt1_bledy_skroty",
    "kryt1_bledy_nazwy_wlasne",
    "kryt1_bledy_wyrazy_obce",
    "kryt1_bledy_akcentowanie",
]

FIELDNAMES = ["anon_id", "rater"] + LIKERT_CRITERIA + ERROR_LOG_COLUMNS + ["uwagi"]


def latest_mapping_path() -> Path | None:
    pattern = re.compile(r"^mapping-(\d{3})\.csv$")
    candidates = [
        (int(match.group(1)), path)
        for path in RESULTS_DIR.glob("mapping-*.csv")
        if (match := pattern.match(path.name))
    ]
    if not candidates:
        return None
    return max(candidates)[1]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an empty Likert rating sheet from a mapping-NNN.csv."
    )
    parser.add_argument(
        "--mapping-run",
        type=int,
        default=None,
        help="Run number NNN of results/mapping-NNN.csv to use (default: the latest).",
    )
    parser.add_argument(
        "--text-ids",
        type=Path,
        default=None,
        help="Optional text file with one text_id per line to restrict the sheet to "
        "(e.g. the group's chosen 10 own + 20 shared texts).",
    )
    parser.add_argument(
        "--rater",
        default="",
        help="Pre-fill the 'rater' column (e.g. your name) for a single-rater sheet.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.mapping_run is not None:
        mapping_path = RESULTS_DIR / f"mapping-{args.mapping_run:03d}.csv"
    else:
        mapping_path = latest_mapping_path()

    if mapping_path is None or not mapping_path.exists():
        print(
            f"No mapping file found ({mapping_path or 'results/mapping-*.csv'}). "
            "Run scripts/anonymize_audio.py first.",
            file=sys.stderr,
        )
        return 1

    run_suffix = re.match(r"^mapping-(\d{3})\.csv$", mapping_path.name)[1]

    with mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
        anon_ids = [row["anon_id"] for row in csv.DictReader(f)]

    if args.text_ids is not None:
        wanted_ids = set(args.text_ids.read_text(encoding="utf-8").splitlines())
        with mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
            anon_ids = [row["anon_id"] for row in csv.DictReader(f) if row["text_id"] in wanted_ids]

    template_path = RESULTS_DIR / f"oceny_tts-{run_suffix}_template.csv"
    with template_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for anon_id in anon_ids:
            writer.writerow({"anon_id": anon_id, "rater": args.rater})

    print(f"Saved: {template_path} ({len(anon_ids)} rows)")
    print("Fill it in while listening to data/anonymized/<anon_id>.wav, save without")
    print(f"the _template suffix as results/oceny_tts-{run_suffix}.csv, then run aggregate_ratings.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
