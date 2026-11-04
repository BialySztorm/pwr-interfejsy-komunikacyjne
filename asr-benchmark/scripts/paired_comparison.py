"""Paired per-recording comparison between two ASR systems from a wyniki_asr-*.csv.

For each file that both chosen systems transcribed, computes the WER difference
(system B - system A). Positive means B did worse on that file, negative means
B did better. Reports a summary (mean/median diff, win counts) and, if scipy is
installed, a Wilcoxon signed-rank test (paired, non-parametric — appropriate here
since WER differences are not assumed normally distributed).

Pairing is by filename ("Plik") within a single wyniki_asr-*.csv run — this
assumes filenames don't collide across speakers, which holds for this dataset's
<index>-<N> naming convention (indices are per-speaker, see docs/WYMAGANIA.md).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Porównanie sparowane per nagranie między dwoma systemami z wyniki_asr-*.csv."
    )
    parser.add_argument("wyniki_csv", type=Path, help="Ścieżka do results/wyniki_asr-NNN.csv.")
    parser.add_argument("--system-a", default="Whisper (base)", help="Pierwszy system (bazowy).")
    parser.add_argument("--system-b", default="Vosk", help="Drugi system (porównywany do A).")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    with open(args.wyniki_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    by_system = defaultdict(dict)
    for row in rows:
        by_system[row["System"]][row["Plik"]] = row

    if args.system_a not in by_system:
        print(f"Brak systemu '{args.system_a}' w pliku. Dostępne: {sorted(by_system)}", file=sys.stderr)
        return 1
    if args.system_b not in by_system:
        print(f"Brak systemu '{args.system_b}' w pliku. Dostępne: {sorted(by_system)}", file=sys.stderr)
        return 1

    files_a = by_system[args.system_a]
    files_b = by_system[args.system_b]
    common_files = sorted(set(files_a) & set(files_b))

    if not common_files:
        print("Brak wspólnych plików między wybranymi systemami.", file=sys.stderr)
        return 1

    comparison_rows = []
    diffs = []
    for filename in common_files:
        wer_a = float(files_a[filename]["WER [%]"])
        wer_b = float(files_b[filename]["WER [%]"])
        diff = wer_b - wer_a
        diffs.append(diff)
        comparison_rows.append({
            "Plik": filename,
            "Kategoria": files_a[filename]["Kategoria"],
            f"WER {args.system_a} [%]": round(wer_a, 2),
            f"WER {args.system_b} [%]": round(wer_b, 2),
            "Różnica (B - A) [p.p.]": round(diff, 2),
        })

    mean_diff = statistics.mean(diffs)
    median_diff = statistics.median(diffs)
    b_worse = sum(1 for d in diffs if d > 0)
    b_better = sum(1 for d in diffs if d < 0)
    tied = sum(1 for d in diffs if d == 0)

    print(f"\n=== PORÓWNANIE SPAROWANE: {args.system_a} vs {args.system_b} ===")
    print(f"Wspólne nagrania: {len(common_files)}")
    print(f"Średnia różnica (B - A): {mean_diff:+.2f} p.p.")
    print(f"Mediana różnicy (B - A): {median_diff:+.2f} p.p.")
    print(f"{args.system_b} gorszy (wyższy WER) na: {b_worse} plikach")
    print(f"{args.system_b} lepszy (niższy WER) na: {b_better} plikach")
    print(f"Remis: {tied} plikach")

    if wilcoxon is not None and b_worse + b_better > 0:
        stat, p_value = wilcoxon([d for d in diffs if d != 0])
        print(f"Test Wilcoxona (sparowany, dwustronny): statystyka={stat:.2f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("-> różnica statystycznie istotna (p < 0.05)")
        else:
            print("-> różnica NIE jest statystycznie istotna (p >= 0.05)")
    else:
        print("scipy niedostępny — pomijam test Wilcoxona (pip install scipy, jeśli potrzebny).")

    out_path = args.wyniki_csv.parent / (
        args.wyniki_csv.stem + f"_parowane_{args.system_a}_vs_{args.system_b}.csv"
    ).replace(" ", "_").replace("(", "").replace(")", "")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"\n✅ Zapisano: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
