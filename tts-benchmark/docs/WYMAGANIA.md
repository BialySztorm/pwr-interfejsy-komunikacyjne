# Status wymagań badania (Edge-TTS vs Piper vs ElevenLabs)

Legenda: 🟢 zrobione · 🟡 częściowo · 🔴 brak

Stan na podstawie wstępnego szkieletu repo — żadne konto/model TTS nie jest jeszcze
podłączone, zbiór tekstów nie jest jeszcze zsynchronizowany, nie ma jeszcze żadnych nagrań
ani ocen. To jest scaffolding analogiczny do `asr-benchmark/`, nie wypełnione badanie.

## Wybór i konfiguracja systemów

| Wymóg | Status | Gdzie w repo | Czego brakuje |
|---|---|---|---|
| Wybór systemów TTS generujących mowę polską | 🟢 | Edge-TTS, Piper TTS, ElevenLabs — `scripts/generate_speech.py` | — |
| Porównywalne parametry (język, głos, szybkość, sposób generowania) | 🔴 | `scripts/generate_speech.py: EDGE_TTS_VOICE, ELEVENLABS_VOICE_ID, PIPER_MODEL_PATH` | Głosy/placeholdery do realnego wyboru i udokumentowania (dlaczego akurat te głosy są "porównywalne") |

## Zbiór tekstów testowych

| Wymóg | Status | Gdzie w repo | Czego brakuje |
|---|---|---|---|
| ~200 tekstów, 5 per osoba, konwencja nazw ustalona z grupą | 🟢 | `data/texts/raw/` (sync z Drive), konwencja `nrindeksu-N-slug.txt` z grupowego README.txt | Sam zbiór — folder Drive musi zostać wypełniony przez grupę |
| 5 kategorii tekstów (info/qna/numbers/long/unique-foreign) | 🟢 | `scripts/text_corpus.py: CATEGORY_BY_SLUG` | — |
| Neutralne + wymagające intonacji (?, !) | 🟡 | kategoria `qna` | Zależy od treści wpisanej przez każdą osobę — nie da się zweryfikować z kodu |
| Normalizacja kodowania/formatu tekstu | 🟢 | `scripts/preprocess_texts.py` | — |

## Generowanie mowy

| Wymóg | Status | Gdzie w repo | Czego brakuje |
|---|---|---|---|
| Nagrania dla każdego tekstu, każdym systemem | 🟢 (kod gotowy) | `scripts/generate_speech.py` | Realne wygenerowanie — wymaga kont/modeli skonfigurowanych |
| Wspólny format audio | 🟡 | edge-tts i elevenlabs zapisują .wav/.mp3 natywnie różnie — do ujednolicenia | Sprawdzić/wymusić ten sam format wyjściowy (np. 16-bit PCM WAV, ta sama częstotliwość) przed anonimizacją |
| Anonimizacja względem źródłowego TTS | 🟢 | `scripts/anonymize_audio.py`, mapowanie w `results/mapping-NNN.csv` (gitignored) | — |

## Kryteria oceny

| # | Kryterium | Sposób oceny | Status | Gdzie w repo | Czego brakuje |
|---|---|---|---|---|---|
| 1 | Poprawność wymowy | Zliczanie błędów (obiektywne) | 🟡 | kolumny `kryt1_bledy_*` w `scripts/make_rating_sheet.py`, tally w `scripts/aggregate_ratings.py` → `bledy_wymowy-NNN.csv` | To log błędów wpisywany ręcznie przez słuchającego, nie automatyczna detekcja — nie da się tego zautomatyzować bez ASR+alignmentu |
| 2 | Zrozumiałość wypowiedzi | Likert 1-5 | 🟢 | `make_rating_sheet.py: kryt2_zrozumialosc`, agregacja w `aggregate_ratings.py` | — |
| 3 | Naturalność brzmienia | Likert 1-5 | 🟢 | `kryt3_naturalnosc` | — |
| 4 | Intonacja i prozodia | Likert 1-5 | 🟢 | `kryt4_intonacja_prozodia` | — |
| 5 | Płynność wypowiedzi | Likert 1-5 | 🟢 | `kryt5_plynnosc` | — |
| 6 | Stabilność syntezy | Likert 1-5 | 🟢 | `kryt6_stabilnosc` | — |
| 7 | Szybkość generowania | Pomiar czasu (obiektywne) | 🟢 | `scripts/generate_speech.py` mierzy czas synthesis per wywołanie → `results/latency_tts-NNN.csv` | — |
| 8 | Przydatność w dialogu | Likert 1-5 | 🟢 | `kryt8_przydatnosc_w_dialogu` | — |

**Brak w brief wag per kryterium** (w przeciwieństwie do briefu ASR z Ćwiczenia 1, który
podawał wagi 0.05-0.25). `scripts/compute_final_score.py` grupuje 8 kryteriów w 3 komponenty
(likert / speed / wymowa) z domyślną wagą równą (1/3 każdy) — **to placeholder, nie ustalona
wartość**, do zastąpienia przez `--weights` po decyzji grupy/prowadzącego.

## Analiza wyników

| Wymóg | Status | Gdzie w repo | Czego brakuje |
|---|---|---|---|
| Wybór 15 własnych + 30 wspólnych tekstów do analizy (grupa 3-osobowa: liczby z brief'u ×1.5, proporcjonalnie do rozmiaru grupy) | 🟡 | `scripts/make_rating_sheet.py --text-ids` (filtr po liście) | Sama lista 15+30 tekstów — decyzja grupy projektowej, nie do automatyzacji |
| Średnie oceny per kryterium i system | 🟢 | `scripts/aggregate_ratings.py` → `results/podsumowanie_tts-NNN.csv` | — |
| Porównanie wyników per kategoria wypowiedzi | 🟢 | `podsumowanie_tts-NNN.csv` (kolumna `category`) | — |
| Najczęstsze błędy syntezy | 🟢 | `results/bledy_wymowy-NNN.csv` | — |
| Porównanie czasu generowania | 🟢 | `results/latency_tts-NNN.csv` | — |
| Wskazanie najlepszego systemu do dialogu | 🟡 | `scripts/compute_final_score.py` | Zależy od realnych danych + ustalonych wag |

## Rzeczy, które nie są kodem, ale są potrzebne

- Konta/klucze dla Edge-TTS (brak klucza, ale wymaga internetu), Piper (model .onnx do
  pobrania ręcznie), ElevenLabs (`ELEVENLABS_API_KEY` + wybór `voice_id`).
- Wypełnienie wspólnego folderu Drive przez całą grupę (200 tekstów, 5 per osoba).
- Decyzja grupy/prowadzącego o wagach kryteriów (brak w briefie).
- Decyzja, które 15 własnych + 30 wspólnych tekstów (45 łącznie, dla grupy 3-osobowej) analizować w pełni.
- Realna sesja odsłuchowa (ten scaffold tylko przygotowuje i agreguje arkusze ocen).
- Prezentacja i raport (`docs/main.tex` to tylko szkielet sekcji, bez danych).

## Znane problemy / ryzyka

- **Brak testów** (`tests/` ma tylko `.gitkeep`), tak jak w `asr-benchmark/`.
- **Format wyjściowy audio nieujednolicony między systemami** — do sprawdzenia przed
  anonimizacją, inaczej różnice w formacie mogłyby (teoretycznie) zdradzić źródłowy system.
- **Kryterium 1 (błędy wymowy) jest manualnym logiem**, nie automatyczną detekcją — jakość
  tej oceny zależy całkowicie od uwagi słuchającego.
- **Domyślne wagi w `compute_final_score.py` są placeholderem** (1/3 każdy) — nie traktować
  jako wynik badania, dopóki grupa/prowadzący nie ustali realnych wag.
