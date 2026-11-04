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
| Te same teksty nagrane przez różne osoby (opcjonalnie) | 🟢 (20 transkrypcji, każda nagrana przez 2 osoby) | `src/asr_benchmark/bench_takes.py`, `results/porownanie_oryginal_wariant-*.csv` | — |
| Raport podaje liczbę nagrań, słów, mówców | 🟢 | `scripts/preprocess_audio.py`, `data/processed/raport_zbioru.md` | — |
| Oba systemy: identyczne pliki, ten sam format | 🟢 | `scripts/preprocess_audio.py` → `data/processed/` (16kHz mono), `bench.py`/`bench_takes.py: resolve_audio_path` | — |
| Ustalony sposób pomiaru czasu | 🟢 | `bench.py`/`bench_takes.py: warmup_whisper_model, warmup_vosk_model` (warm-up przed pomiarem), mediana/maks. opóźnienia per plik w podsumowaniu | — |

## Kryteria oceny (wagi)

| # | Kryterium | Waga | Status | Gdzie w repo | Czego brakuje |
|---|---|---|---|---|---|
| 1 | WER korpusowy + per kategoria | 0.25 | 🟡 | `bench.py: main()`, `results/podsumowanie_asr-*.csv` | Liczby są, ale normalizacja tekstu (dodana przez kolegę) wymaga weryfikacji, czy faktycznie usuwa przewagę formatu zapisu liczb/dat |
| 2 | Nazwy własne: poprawne/wszystkie | 0.05 | 🔴 | — | Tylko WER całej kategorii 4 jako proxy, zero metryki na poziomie encji |
| 3 | Daty/czas/liczby: poprawne/wszystkie | 0.10 | 🔴 | — | Jak wyżej — brak dedykowanej metryki na poziomie encji |
| 4 | Krótkie spontaniczne + dźwięki pozawerbalne | 0.10 | 🟡 | WER kategorii 3 w `podsumowanie_asr-*.csv` | Brak analizy błędów, brak oznaczenia, które referencje zawierają "wow/ojej" itp. |
| 5 | Odporność na szum (Δ WER cisza vs szum) | 0.10 | 🟡 | WER kategorii 5 liczony | Różnica (Δ) między kategoriami nie jest jawnie obliczana/eksponowana |
| 6 | Rozpoznanie intencji | 0.15 | 🔴 | — | Zero adnotacji intencji, zero kodu oceniającego czy transkrypcja pozwala wydobyć intencję — **najwyższa waga, zero pokrycia** |
| 7 | Szybkość: opóźnienie + RTF | 0.10 | 🟡 | `RTF` + mediana/maks. opóźnienia per plik w `podsumowanie_asr*-*.csv` | Brak RTF per kategoria; mediana/maks. to pomiar offline-batch, nie pomiar w rzeczywistym przepływie (patrz pkt R) |
| 8 | Stabilność/kompletność: %, S/D/I, braki | 0.05 | 🟡 | S/D/I w CSV | "Brak transkrypcji" tylko na konsoli, nieagregowane do liczby/procentu w CSV |
| 9 | Integracja: API, format, sprzęt, koszty, dane | 0.10 | 🔴 | — | Zero notatek/dokumentu porównującego te aspekty (ocena opisowa) |

## Analiza i synteza

| Wymóg | Status | Gdzie w repo | Czego brakuje |
|---|---|---|---|
| Ocena końcowa ważona + opisane przeskalowanie do wspólnej skali | 🔴 | — | Brak kodu łączącego 9 kryteriów, brak opisanej metody przeskalowania |
| Porównanie sparowane per nagranie (Whisper vs Vosk) | 🔴 | — | `bench_takes.py` ma analogiczny mechanizm, ale dla "wariant vs oryginał **tego samego systemu**" — inna oś, nie Whisper-vs-Vosk per plik |
| Pomiar czasu w rzeczywistym przepływie (reprezentatywny podzbiór) | 🔴 | — | Benchmark jest czysto offline/batch na plikach |
| Wnioski (4 pytania z wymagań) | 🔴 | — | Nie napisane; częściowo odpowiadalne z obecnych danych (1–3), pytanie 4 wymaga sekcji integracyjnej |
| Raport + prezentacja | 🔴 | — | Nie znaleziono żadnego pliku raportu/prezentacji w repo |

## Rzeczy, które nie są kodem, ale są potrzebne

- Weryfikacja ludzka transkrypcji (czy ktoś sprawdził zgodność referencji z nagraniem).
- Potwierdzenie finalności zbioru (czy cały rocznik już wgrał dane).
- Adnotacja encji (nazwy własne, daty, liczby) w referencjach kategorii 4.
- Zdefiniowanie i adnotacja intencji + sposób oceny (reguła/osoba oceniająca).
- Opisowa ocena integracji (kryterium 9) — API, koszty, wymagania sprzętowe, prywatność.
- Decyzja o metodzie przeskalowania metryk do wspólnej skali.
- Decyzja, co znaczy "pomiar w rzeczywistym przepływie" w tym projekcie.
- Sam raport i prezentacja.

## Znane problemy / ryzyka

- **Normalizacja tekstu przed WER** — dodana przez kolegę; wymaga sprawdzenia, czy faktycznie
  wyrównuje różnice typu "Vosk zwraca liczby słowami, referencje i Whisper cyframi".
- **`openai-whisper` vs `faster-whisper`** — specyfikacja i `__init__.py` mówią "faster-whisper",
  `requirements.txt`/`bench.py` używają `openai-whisper`. Decyzja do podjęcia.
- **Niejednolity format audio** w korpusie (sample rate, kanały, bit depth) — patrz tabela wyżej.
- **Duplikat treści u Karola Zająca** — `276263-1.txt` do `-4.txt` mają identyczny tekst,
  więc te nagrania nie realizują wymaganej różnorodności kategorii u tego mówcy.
- **Brak testów** (`tests/` ma tylko `.gitkeep`).
- Możliwy **bias GPU warm-up** w pomiarze RTF na małych próbkach (np. w `bench_takes.py`).
