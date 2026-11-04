# Status wymagań badania (Whisper vs Vosk)

Legenda: 🟢 zrobione · 🟡 częściowo · 🔴 brak

Stan na podstawie analizy repo i danych w `data/raw/` (38 mówców, 216 `.wav`, 190 kompletnych
par `.wav`/`.txt`, ~45,9 min nagrań, 4374 słowa referencji). Nie wszystko da się potwierdzić
z samego repo — takie przypadki są oznaczone wprost.

## Zbiór danych i warunki

| Wymóg | Status | Gdzie w repo | Czego brakuje |
|---|---|---|---|
| ~200 wypowiedzi, ≥25 mówców, tylko PL, transkrypcje zweryfikowane przez człowieka | 🟡 | `data/raw/`, `scripts/sync_drive.py` | Ufamy studentom nagrywającym że zweryfikowali poprawność |
| Struktura "Imię Nazwisko" + `<indeks>-<N>.wav/.txt` | 🟢 | `data/raw/*` | — |
| Te same teksty nagrane przez różne osoby (opcjonalnie) | 🟢 (20 transkrypcji, każda nagrana przez 2 osoby) | `src/bench_takes.py`, `results/porownanie_oryginal_wariant-*.csv` | — |
| Raport podaje liczbę nagrań, słów, mówców | 🟢 | `scripts/preprocess_audio.py`, `data/processed/raport_zbioru.md` | — |
| Oba systemy: identyczne pliki, ten sam format | 🟢 | `scripts/preprocess_audio.py` → `data/processed/` (16kHz mono), `bench.py`/`bench_takes.py: resolve_audio_path` | — |
| Ustalony sposób pomiaru czasu | 🟢 | `bench.py`/`bench_takes.py: warmup_whisper_model, warmup_vosk_model` (warm-up przed pomiarem), mediana/maks. opóźnienia per plik w podsumowaniu | — |

## Kryteria oceny (wagi)

| # | Kryterium | Waga | Status | Gdzie w repo | Czego brakuje |
|---|---|---|---|---|---|
| 1 | WER korpusowy + per kategoria | 0.25 | 🟢 | `bench.py: main()`, `results/podsumowanie_asr-*.csv` | Zweryfikowane: normalizacja poprawnie zamienia liczby słowne na cyfry (np. "sześćset pięćdziesiąt"→"650"); wysoki WER w kat. 4 to realne błędy rozpoznania marek/modeli, nie artefakt formatu |
| 2 | Nazwy własne: poprawne/wszystkie | 0.05 | 🟢 | `bench.py: extract_named_entities`, `Dokładność nazw własnych [%]` w `podsumowanie_asr-*.csv` | Heurystyka: wielka litera w środku zdania = nazwa własna/marka (nie prawdziwy NER) |
| 3 | Daty/czas/liczby: poprawne/wszystkie | 0.10 | 🟢 | `bench.py: extract_entities, count_matched_entities`, `Dokładność encji (liczby/daty/czas) [%]` w `podsumowanie_asr-*.csv` | — |
| 4 | Krótkie spontaniczne + dźwięki pozawerbalne | 0.10 | 🟢 | `bench.py: has_non_verbal_sound`, WER podzbioru w `podsumowanie_asr-*.csv`; jakościowa analiza błędów w `docs/main.tex` (sekcja "Przykłady...") | Heurystyka słownikowa (ojej/hmm/aha/yyy...), nie w pełni zautomatyzowana analiza |
| 5 | Odporność na szum (Δ WER cisza vs szum) | 0.10 | 🟢 | `Δ WER szum vs pozostałe kategorie [p.p.]` w `podsumowanie_asr-*.csv`, `docs/main.tex` | — |
| 6 | Rozpoznanie intencji | 0.15 | 🟡 | `bench.py: classify_intent`, `Zgodność intencji [%]` w `podsumowanie_asr-*.csv`, opisane w `docs/main.tex` (sekcja "Rozpoznawanie intencji") | To heurystyka słów kluczowych (zamowienie/rezerwacja/problem/polecenie/pytanie/inne), nie prawdziwy model NLU czy ręczna adnotacja |
| 7 | Szybkość: opóźnienie + RTF | 0.10 | 🟢 | `RTF` per kategoria w `podsumowanie_asr-*.csv`; realny pomiar opóźnienia ogonowego w `scripts/latency_pipeline.py` i `docs/main.tex` (sekcja "Pomiar opóźnienia") | — |
| 8 | Stabilność/kompletność: %, S/D/I, braki | 0.05 | 🟢 | `Kompletność [%]`, `Nagrania bez transkrypcji` w `podsumowanie_asr-*.csv` | — |
| 9 | Integracja: API, format, sprzęt, koszty, dane | 0.10 | 🟢 | `docs/INTEGRACJA.md` | — |

## Analiza i synteza

| Wymóg | Status | Gdzie w repo | Czego brakuje |
|---|---|---|---|
| Ocena końcowa ważona + opisane przeskalowanie do wspólnej skali | 🟢 | `scripts/compute_final_score.py`, `docs/main.tex` (sekcja "Ocena końcowa") — wypełnione realnymi liczbami z pełnego przebiegu | — |
| Porównanie sparowane per nagranie (Whisper vs Vosk) | 🟢 | `scripts/paired_comparison.py`, `docs/main.tex` (sekcja "Porównanie sparowane") — z testem Wilcoxona | — |
| Pomiar czasu w rzeczywistym przepływie (reprezentatywny podzbiór) | 🟡 | `scripts/latency_pipeline.py`, `docs/main.tex` (sekcja "Pomiar opóźnienia") — wypełnione realnymi liczbami | To symulacja (audio "dociera" w czasie realnym z plików), nie pomiar z żywego mikrofonu |
| Wnioski (4 pytania z wymagań) | 🟢 | `docs/main.tex: Podsumowanie i wnioski końcowe` | — |
| Raport + prezentacja | 🟡 | `docs/main.tex` — raport wypełniony realnymi danymi z pełnego benchmarku | Prezentacji nadal nie ma |

## Rzeczy, które nie są kodem, ale są potrzebne

- Weryfikacja ludzka transkrypcji (czy ktoś sprawdził zgodność referencji z nagraniem) — nie do
  potwierdzenia z repo.
- Potwierdzenie finalności zbioru (czy cały rocznik już wgrał dane) — nie do potwierdzenia z repo.
- Prezentacja — jedyny deliverable, którego jeszcze nigdzie nie ma.

## Znane problemy / ryzyka

- **`openai-whisper` vs `faster-whisper`** — specyfikacja i `__init__.py` mówią "faster-whisper",
  `requirements.txt`/`bench.py` używają `openai-whisper`. Decyzja do podjęcia (nie zmieniona).
- **Duplikat treści u Karola Zająca** — `276263-1.txt` do `-4.txt` mają identyczny tekst,
  więc te nagrania nie realizują wymaganej różnorodności kategorii u tego mówcy.
- **Brak testów** (`tests/` ma tylko `.gitkeep`).
- **Heurystyki, nie modele**: rozpoznawanie intencji (`classify_intent`) i nazw własnych
  (`extract_named_entities`) to reguły słów kluczowych / wielkich liter, nie prawdziwy NLU/NER —
  traktować wyniki jako przybliżone, zgodnie z zastrzeżeniem w `docs/main.tex`.
- **Pomiar w rzeczywistym przepływie to symulacja** (`scripts/latency_pipeline.py`) — audio z
  plików podawane w czasie realnym, nie pomiar z żywego mikrofonu.
