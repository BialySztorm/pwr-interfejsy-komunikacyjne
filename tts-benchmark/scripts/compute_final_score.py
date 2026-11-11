"""Compute the weighted final score (criterion synthesis) across the 8 criteria.

Unlike asr-benchmark's compute_final_score.py, this brief gives NO explicit
per-criterion weights — see docs/WYMAGANIA.md. This script therefore groups the
8 criteria into 3 components and defaults to an EQUAL weight across them; treat
the default as a placeholder to be replaced once the group agrees on real
weights (--weights), not as an official value:

  - likert   (0.333): mean of criteria 2-6, 8 (mean_likert_overall, "ALL" row of
               results/podsumowanie_tts-NNN.csv) — already 0-100 on a rescaled
               1-5 -> 0-100 basis (score = (likert - 1) / 4 * 100)
  - speed    (0.333): criterion 7, from results/latency_tts-NNN.csv — mean
               chars/second per system, rescaled relative to the fastest system
               in this run (fastest = 100, others proportionally lower)
  - wymowa   (0.333): criterion 1, from results/bledy_wymowy-NNN.csv — 100 minus
               the mean "% of ratings with a logged error" across the 5 error
               subtypes

All three are clipped to [0, 100] before the weighted sum (weights sum to 1.0,
so the final score is also in [0, 100]).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

DEFAULT_WEIGHTS = {"likert": 1 / 3, "speed": 1 / 3, "wymowa": 1 / 3}


def clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the weighted final score across all 3 components (8 criteria)."
    )
    parser.add_argument("podsumowanie_csv", type=Path, help="results/podsumowanie_tts-NNN.csv")
    parser.add_argument("latency_csv", type=Path, help="results/latency_tts-NNN.csv")
    parser.add_argument("bledy_wymowy_csv", type=Path, help="results/bledy_wymowy-NNN.csv")
    parser.add_argument(
        "--weights",
        default=None,
        metavar="likert=W,speed=W,wymowa=W",
        help="Override the default equal weighting, e.g. 'likert=0.5,speed=0.2,wymowa=0.3'. "
        "Must sum to 1.0.",
    )
    return parser.parse_args(argv)


def parse_weights(raw: str | None) -> dict[str, float]:
    if raw is None:
        return dict(DEFAULT_WEIGHTS)
    weights = {}
    for part in raw.split(","):
        name, value = part.split("=", 1)
        weights[name.strip()] = float(value)
    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError(f"--weights must specify exactly: {', '.join(DEFAULT_WEIGHTS)}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--weights must sum to 1.0 (got {total}).")
    return weights


def load_likert_scores(podsumowanie_csv: Path) -> dict[str, float]:
    with podsumowanie_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["category"] == "ALL"]
    scores = {}
    for row in rows:
        likert_mean = row.get("mean_likert_overall", "")
        if likert_mean == "":
            continue
        scores[row["system"]] = clip((float(likert_mean) - 1) / 4 * 100)
    return scores


def load_speed_scores(latency_csv: Path) -> dict[str, float]:
    with latency_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    chars_per_second_by_system: dict[str, list[float]] = {}
    for row in rows:
        raw = row.get("chars_per_second", "")
        if raw == "":
            continue
        chars_per_second_by_system.setdefault(row["system"], []).append(float(raw))

    means = {system: statistics.mean(values) for system, values in chars_per_second_by_system.items() if values}
    if not means:
        return {}
    fastest = max(means.values())
    return {system: clip(100 * mean / fastest) for system, mean in means.items()}


def load_wymowa_scores(bledy_wymowy_csv: Path) -> dict[str, float]:
    with bledy_wymowy_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    pct_by_system: dict[str, list[float]] = {}
    for row in rows:
        pct_by_system.setdefault(row["system"], []).append(float(row["pct_of_ratings"]))

    return {system: clip(100 - statistics.mean(values)) for system, values in pct_by_system.items()}


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        weights = parse_weights(args.weights)
    except ValueError as exc:
        print(f"Błąd argumentów: {exc}", file=sys.stderr)
        return 1

    for path in (args.podsumowanie_csv, args.latency_csv, args.bledy_wymowy_csv):
        if not path.exists():
            print(f"Plik nie istnieje: {path}", file=sys.stderr)
            return 1

    likert_scores = load_likert_scores(args.podsumowanie_csv)
    speed_scores = load_speed_scores(args.latency_csv)
    wymowa_scores = load_wymowa_scores(args.bledy_wymowy_csv)

    systems = sorted(set(likert_scores) | set(speed_scores) | set(wymowa_scores))
    if not systems:
        print("Brak systemów do oceny we wskazanych plikach wejściowych.", file=sys.stderr)
        return 1

    results = []
    for system in systems:
        available = {"likert": likert_scores, "speed": speed_scores, "wymowa": wymowa_scores}
        components = {name: scores.get(system, 0.0) for name, scores in available.items()}
        for name, scores in available.items():
            if system not in scores:
                print(f"UWAGA: brak danych '{name}' dla systemu '{system}', liczone jako 0.")
        final_score = sum(components[name] * weight for name, weight in weights.items())
        results.append((system, components, final_score))

    print("\n=== OCENA KOŃCOWA (ważona synteza 3 komponentów / 8 kryteriów) ===")
    print(f"Wagi: {weights} (DOMYŚLNIE RÓWNE — nie są to wagi z wymagań, patrz docs/WYMAGANIA.md)")
    for system, components, final_score in results:
        print(f"\n{system}: {final_score:.2f} / 100")
        for name, weight in weights.items():
            print(f"  {name} (waga {weight:.3f}): {components[name]:.2f}")

    out_path = args.podsumowanie_csv.parent / (
        args.podsumowanie_csv.stem.replace("podsumowanie", "ocena_koncowa") + ".csv"
    )
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["system"] + list(weights.keys()) + ["ocena_koncowa_0_100"])
        for system, components, final_score in results:
            writer.writerow([system] + [round(components[name], 2) for name in weights] + [round(final_score, 2)])
    print(f"\nZapisano: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
