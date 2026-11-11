"""Aggregate a filled rating sheet into per-system/per-criterion/per-category summaries.

Joins results/oceny_tts-NNN.csv (filled in by raters, anon_id-only) with its
results/mapping-NNN.csv (same NNN, de-anonymization key) to recover system and
category, then writes:

  - results/podsumowanie_tts-NNN.csv: mean Likert score (criteria 2-6, 8) per
    system, both overall ("ALL" category row) and per text category
  - results/bledy_wymowy-NNN.csv: tallied pronunciation-error log (criterion 1)
    per system and error subtype — counts non-empty entries in each kryt1_bledy_*
    column, i.e. "rater logged at least one error of this subtype for this clip"

Criterion 7 (speed) is NOT aggregated here — see results/latency_tts-NNN.csv and
scripts/compute_final_score.py, which combines both into the final score.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_rating_sheet import ERROR_LOG_COLUMNS, LIKERT_CRITERIA  # noqa: E402
from text_corpus import PROJECT_ROOT, parse_filename  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

RESULTS_DIR = PROJECT_ROOT / "results"

CATEGORY_BY_SLUG_LOCAL = {
    "info": "Krótki komunikat informacyjny",
    "qna": "Pytanie / odpowiedź",
    "numbers": "Liczby, daty, godziny, skróty",
    "long": "Zdanie złożone / dłuższa wypowiedź",
    "unique-foreign": "Nazwy własne i wyrazy obcojęzyczne",
}


def category_for_text_id(text_id: str) -> str:
    """text_id looks like '<person>/<nrindeksu>-<N>-<slug>'."""
    stem = text_id.split("/", 1)[1]
    _, _, slug = parse_filename(stem)
    return CATEGORY_BY_SLUG_LOCAL[slug]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate a filled oceny_tts-NNN.csv against its mapping-NNN.csv."
    )
    parser.add_argument(
        "ratings_csv",
        type=Path,
        help="Path to the filled results/oceny_tts-NNN.csv.",
    )
    parser.add_argument(
        "mapping_csv",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the matching results/mapping-NNN.csv (default: same NNN as ratings_csv).",
    )
    return parser.parse_args(argv)


def infer_mapping_path(ratings_csv: Path) -> Path:
    # oceny_tts-NNN.csv -> mapping-NNN.csv
    run_suffix = ratings_csv.stem.rsplit("-", 1)[-1]
    return ratings_csv.parent / f"mapping-{run_suffix}.csv"


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    mapping_csv = args.mapping_csv or infer_mapping_path(args.ratings_csv)

    if not args.ratings_csv.exists():
        template_path = args.ratings_csv.with_name(args.ratings_csv.stem + "_template.csv")
        hint = (
            f" Did you forget to fill in and save {template_path.name} as {args.ratings_csv.name}?"
            if template_path.exists()
            else ""
        )
        print(f"Ratings file not found: {args.ratings_csv}.{hint}", file=sys.stderr)
        return 1

    if not mapping_csv.exists():
        print(f"Mapping file not found: {mapping_csv}", file=sys.stderr)
        return 1

    with mapping_csv.open("r", encoding="utf-8-sig", newline="") as f:
        mapping = {row["anon_id"]: row for row in csv.DictReader(f)}

    with args.ratings_csv.open("r", encoding="utf-8-sig", newline="") as f:
        ratings = list(csv.DictReader(f))

    likert_scores = defaultdict(lambda: defaultdict(list))  # (system, category) -> criterion -> [scores]
    error_counts = defaultdict(lambda: defaultdict(int))  # system -> error_column -> count
    rated_counts = defaultdict(int)  # (system, category) -> n

    skipped = 0
    for row in ratings:
        anon_id = row.get("anon_id", "").strip()
        info = mapping.get(anon_id)
        if info is None:
            print(f"WARNING: anon_id {anon_id!r} not found in mapping, skipping row.")
            skipped += 1
            continue

        system = info["system"]
        category = category_for_text_id(info["text_id"])
        rated_counts[(system, category)] += 1
        rated_counts[(system, "ALL")] += 1

        for criterion in LIKERT_CRITERIA:
            raw = row.get(criterion, "").strip()
            if raw == "":
                continue
            likert_scores[(system, category)][criterion].append(float(raw))
            likert_scores[(system, "ALL")][criterion].append(float(raw))

        for error_column in ERROR_LOG_COLUMNS:
            if row.get(error_column, "").strip():
                error_counts[system][error_column] += 1

    systems = sorted({info["system"] for info in mapping.values()})
    categories = ["ALL"] + sorted(set(CATEGORY_BY_SLUG_LOCAL.values()))

    summary_rows = []
    for system in systems:
        for category in categories:
            n = rated_counts.get((system, category), 0)
            if n == 0:
                continue
            row = {"system": system, "category": category, "n_ratings": n}
            criterion_means = []
            for criterion in LIKERT_CRITERIA:
                values = likert_scores[(system, category)].get(criterion, [])
                mean = round(statistics.mean(values), 2) if values else ""
                row[criterion] = mean
                if values:
                    criterion_means.append(mean)
            row["mean_likert_overall"] = round(statistics.mean(criterion_means), 2) if criterion_means else ""
            summary_rows.append(row)

    summary_path = RESULTS_DIR / ("podsumowanie_tts-" + args.ratings_csv.stem.rsplit("-", 1)[-1] + ".csv")
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["system", "category", "n_ratings"] + LIKERT_CRITERIA + ["mean_likert_overall"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {summary_path}")

    errors_path = RESULTS_DIR / ("bledy_wymowy-" + args.ratings_csv.stem.rsplit("-", 1)[-1] + ".csv")
    with errors_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "error_subtype", "count", "pct_of_ratings"])
        for system in systems:
            total_ratings = rated_counts.get((system, "ALL"), 0)
            for error_column in ERROR_LOG_COLUMNS:
                count = error_counts[system].get(error_column, 0)
                pct = round(100 * count / total_ratings, 1) if total_ratings else 0.0
                writer.writerow([system, error_column, count, pct])
    print(f"Saved: {errors_path}")

    if skipped:
        print(f"Skipped {skipped} row(s) with unknown anon_id.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
