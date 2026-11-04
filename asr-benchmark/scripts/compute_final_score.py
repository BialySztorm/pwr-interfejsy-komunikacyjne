"""Compute the weighted final score (criterion synthesis) from a podsumowanie_asr-*.csv.

Methodology (rescaling to a common 0-100 "higher is better" scale per criterion,
then a weighted sum using the weights from the research brief):

  1. WER korpusowy (0.25):            score = 100 - Globalny WER [%]
  2. Nazwy własne (0.05):             score = Dokładność nazw własnych [%]            (already 0-100, higher better)
  3. Daty/czas/liczby (0.10):         score = Dokładność encji (liczby/daty/czas) [%]  (already 0-100, higher better)
  4. Spontaniczne + pozawerbalne (0.10): score = 100 - WER - spontaniczne z dźwiękiem pozawerbalnym [%]
  5. Odporność na szum (0.10):        score = 100 - Δ WER szum vs pozostałe kategorie [p.p.]
  6. Intencja (0.15):                 score = Zgodność intencji [%]                    (already 0-100, higher better)
  7. Szybkość / RTF (0.10):           score = 100 * (1 - RTF), RTF=1 (real-time) -> 0
  8. Stabilność/kompletność (0.05):   score = Kompletność [%]                          (already 0-100, higher better)
  9. Integracja (0.10):               score = ocena opisowa z docs/INTEGRACJA.md, podana ręcznie
                                       (nie da się wyliczyć automatycznie — patrz --integration)

Wszystkie wyniki są obcinane do [0, 100] przed zsumowaniem (wagi sumują się do 1.0,
więc końcowy wynik jest też w [0, 100]).

Kryterium 9 (integracja) jest jedynym, które wymaga ręcznej oceny — podaj ją przez
--integration "<System>=<0-100>" (można kilka razy). Systemy bez podanej oceny
dostają 50 (neutralnie) i ostrzeżenie w konsoli.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

WEIGHTS = {
    "WER korpusowy": 0.25,
    "Nazwy własne": 0.05,
    "Daty/czas/liczby": 0.10,
    "Spontaniczne + pozawerbalne": 0.10,
    "Odporność na szum": 0.10,
    "Intencja": 0.15,
    "Szybkość (RTF)": 0.10,
    "Stabilność/kompletność": 0.05,
    "Integracja": 0.10,
}


def clip(value, low=0.0, high=100.0):
    return max(low, min(high, value))


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Oblicza ważoną ocenę końcową (synteza 9 kryteriów) z podsumowanie_asr-*.csv."
    )
    parser.add_argument(
        "summary_csv",
        type=Path,
        help="Ścieżka do results/podsumowanie_asr-NNN.csv (ten, który ma być podstawą oceny).",
    )
    parser.add_argument(
        "--integration",
        action="append",
        default=[],
        metavar="SYSTEM=SCORE",
        help='Ręczna ocena integracji (kryterium 9, 0-100), np. --integration "Vosk=85". '
        "Można podać wielokrotnie, raz per system. Bez podania: 50 (neutralnie) + ostrzeżenie.",
    )
    return parser.parse_args(argv)


def parse_integration_scores(raw_values):
    scores = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"Nieprawidłowy format --integration: {raw!r} (oczekiwano SYSTEM=SCORE)")
        name, value = raw.split("=", 1)
        scores[name.strip()] = float(value)
    return scores


def main(argv):
    args = parse_args(argv)
    integration_scores = parse_integration_scores(args.integration)

    with open(args.summary_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Plik podsumowania jest pusty.", file=sys.stderr)
        return 1

    results = []
    for row in rows:
        system = row["System"]
        integration_score = integration_scores.get(system)
        if integration_score is None:
            print(
                f"Brak ręcznej oceny integracji dla '{system}' — używam 50 (neutralnie). "
                f"Podaj --integration \"{system}=<0-100>\" dla realnej wartości."
            )
            integration_score = 50.0

        scores = {
            "WER korpusowy": clip(100 - float(row["Globalny WER [%]"])),
            "Nazwy własne": clip(float(row["Dokładność nazw własnych [%]"])),
            "Daty/czas/liczby": clip(float(row["Dokładność encji (liczby/daty/czas) [%]"])),
            "Spontaniczne + pozawerbalne": clip(
                100 - float(row["WER - spontaniczne z dźwiękiem pozawerbalnym [%]"])
            ),
            "Odporność na szum": clip(100 - float(row["Δ WER szum vs pozostałe kategorie [p.p.]"])),
            "Intencja": clip(float(row["Zgodność intencji [%]"])),
            "Szybkość (RTF)": clip(100 * (1 - float(row["Globalny RTF"]))),
            "Stabilność/kompletność": clip(float(row["Kompletność [%]"])),
            "Integracja": clip(integration_score),
        }
        final_score = sum(scores[name] * weight for name, weight in WEIGHTS.items())
        results.append((system, scores, final_score))

    print("\n=== OCENA KOŃCOWA (ważona synteza 9 kryteriów) ===")
    for system, scores, final_score in results:
        print(f"\n{system}: {final_score:.2f} / 100")
        for name, weight in WEIGHTS.items():
            print(f"  {name} (waga {weight:.2f}): {scores[name]:.2f}")

    out_path = args.summary_csv.parent / (args.summary_csv.stem + "_ocena_koncowa.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["System"] + list(WEIGHTS.keys()) + ["Ocena końcowa [0-100]"])
        for system, scores, final_score in results:
            writer.writerow([system] + [round(scores[name], 2) for name in WEIGHTS] + [round(final_score, 2)])
    print(f"\n✅ Zapisano: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
