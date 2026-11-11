# TTS Benchmark

Comparison of three Polish TTS systems: **Edge-TTS** (free, cloud), **Piper TTS** (local,
open-source), and **ElevenLabs** (paid/limited free tier, cloud).

Status: preliminary scaffold, mirroring `asr-benchmark/`'s structure. No accounts/models are
configured yet, the shared text corpus hasn't been synced, and no recordings/ratings exist —
see [`docs/WYMAGANIA.md`](docs/WYMAGANIA.md) for exactly what's done vs. missing.

## Project structure

```
tts-benchmark/
  data/texts/raw/        # synced from Google Drive, not in git
  data/texts/processed/  # normalized text (encoding/whitespace), not in git
  data/raw/               # generated audio per system, not in git
  data/anonymized/        # shuffled/renamed copies for blind listening, not in git
  models/                 # Piper voice models, not in git
  results/                # outputs (raw ratings/mapping gitignored, summaries tracked)
  docs/GDRIVE_API_KEY.md
  docs/WYMAGANIA.md       # requirement-by-requirement status vs. the research brief
  docs/main.tex           # the written report (skeleton)
  scripts/text_corpus.py        # shared helpers: filename convention, corpus listing
  scripts/sync_drive.py         # syncs the shared text corpus from Google Drive
  scripts/preprocess_texts.py   # normalizes text encoding/whitespace, writes raport_zbioru.md
  scripts/generate_speech.py    # synthesizes speech with Edge-TTS/Piper/ElevenLabs, measures latency
  scripts/anonymize_audio.py    # shuffles/renames generated audio for blind listening
  scripts/make_rating_sheet.py  # builds an empty Likert rating template
  scripts/aggregate_ratings.py  # de-anonymizes + aggregates filled ratings
  scripts/compute_final_score.py # weighted final score across all criteria
  tests/
```

## Text corpus convention

Shared with the group via a Google Drive folder, per its own `README.txt`: each person
creates a folder named `"Imię Nazwisko"` and puts 5 UTF-8 `.txt` files in it (no spaces in
filenames):

```
nrindeksu-1-info.txt             krótkie komunikaty informacyjne
nrindeksu-2-qna.txt              pytania i odpowiedzi
nrindeksu-3-numbers.txt          liczby, daty, godziny, skróty
nrindeksu-4-long.txt             zdania złożone i dłuższe wypowiedzi
nrindeksu-5-unique-foreign.txt   nazwy własne i wyrazy obcojęzyczne
```

`scripts/text_corpus.py` parses this convention; an unrecognized filename is skipped with a
warning rather than crashing the pipeline.

## Setup

### 1. Create a virtual environment

```powershell
py -3.15 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure `.env`

Copy `.env.example` to `.env` and fill in:

```
GDRIVE_FOLDER_URL=https://drive.google.com/drive/folders/<your-folder-id>
GDRIVE_API_KEY=<your-google-api-key>
ELEVENLABS_API_KEY=<your-elevenlabs-api-key>
```

If you don't have a Google Drive API key yet (or reuse the one from `asr-benchmark/`), see
[docs/GDRIVE_API_KEY.md](docs/GDRIVE_API_KEY.md). Edge-TTS needs no API key (just internet);
Piper needs no key either, but needs a downloaded voice model (step 5).

### 3. Sync the shared text corpus

```powershell
python scripts\sync_drive.py --dry-run
python scripts\sync_drive.py
```

### 4. Normalize the texts

```powershell
python scripts\preprocess_texts.py
```

Writes `data/texts/processed/` (UTF-8, LF line endings, no trailing whitespace, NFC-normalized)
and `data/texts/processed/raport_zbioru.md` with dataset-wide counts.

### 5. Set up each TTS system

- **Edge-TTS**: nothing to install besides `pip install edge-tts` (already in
  `requirements.txt`); needs internet. No setup needed — `EDGE_TTS_VOICE` is already set.
- **Piper**: `pip install piper-tts`. The Polish voice model is already downloaded —
  `models/pl_PL-darkman-medium.onnx` (+ its `.onnx.json`), from
  https://huggingface.co/rhasspy/piper-voices (22kHz, medium quality). No setup needed.
- **ElevenLabs**: `pip install elevenlabs`, create an account, put your API key in `.env`. The
  default `ELEVENLABS_VOICE_ID` points at ElevenLabs' premade "Adam" voice — **verify it still
  resolves to Adam in your account's Voice Library before a real run**: ElevenLabs
  occasionally retires/remaps premade voice IDs (e.g. Antoni and Arnold were remapped to Adam
  in Feb 2026). See [docs/ELEVENLABS_API_KEY.md](docs/ELEVENLABS_API_KEY.md) for which key
  permissions to grant (only Text to Speech is needed) and how to limit the key's blast
  radius.

#### Voice selection

All three systems are set to a **neutral adult male voice**, so differences in the listening
evaluation reflect synthesis quality rather than gender/timbre:

| System | Voice | Notes |
|---|---|---|
| Edge-TTS | `pl-PL-MarekNeural` | Microsoft's only standard male Polish neural voice |
| Piper | `pl_PL-darkman-medium` | the only actively-maintained male Polish Piper voice (the other male option, `mc_speech`, hasn't been updated in over a year) |
| ElevenLabs | "Adam" (`pNInz6obpgDQGcFmaJgB`) | deep adult male, multilingual-capable — pick a different voice from your account if Adam doesn't sound right for Polish, but keep it male to match the other two |

### 6. Generate speech

```powershell
python scripts\generate_speech.py --system edge-tts piper elevenlabs
```

Run all systems you want compared in **one** invocation, same convention as
`asr-benchmark`'s `bench.py --whisper-model`. Produces `data/raw/<system>/<person>/*.wav` and
`results/latency_tts-NNN.csv` (criterion 7: generation time, chars/second per text).

### 7. Anonymize for blind listening

```powershell
python scripts\anonymize_audio.py
```

Produces `data/anonymized/<random_id>.wav` and the (gitignored) de-anonymization key
`results/mapping-NNN.csv` — never share this file with raters.

### 8. Build a rating sheet and collect Likert scores

```powershell
python scripts\make_rating_sheet.py --rater "Your Name"
```

Produces `results/oceny_tts-NNN_template.csv`. Listen to each
`data/anonymized/<anon_id>.wav`, fill in the Likert columns (criteria 2--6, 8) and the
pronunciation-error log columns (criterion 1), save it as `results/oceny_tts-NNN.csv` (without
`_template`).

Per the brief, each 2-3 person group picks 10 of their own texts + 20 shared ones for the full
evaluation — use `--text-ids path/to/list.txt` to restrict the sheet to that subset (one
`text_id` per line, matching the `text_id` column in `results/mapping-NNN.csv`).

### 9. Aggregate ratings

```powershell
python scripts\aggregate_ratings.py results\oceny_tts-NNN.csv
```

Produces `results/podsumowanie_tts-NNN.csv` (mean Likert per system/criterion/category) and
`results/bledy_wymowy-NNN.csv` (tallied pronunciation errors per system/subtype).

### 10. Weighted final score

```powershell
python scripts\compute_final_score.py results\podsumowanie_tts-NNN.csv results\latency_tts-NNN.csv results\bledy_wymowy-NNN.csv
```

**The brief gives no official per-criterion weights** (unlike the ASR brief). This script
defaults to an equal weight across 3 components (Likert / speed / pronunciation) — a
placeholder, not a decided value. Override with `--weights "likert=0.5,speed=0.2,wymowa=0.3"`
once the group/instructor settles on real weights.

## Status and results

- [`docs/WYMAGANIA.md`](docs/WYMAGANIA.md) — requirement-by-requirement status against the
  research brief (🟢/🟡/🔴), including open decisions (criterion weights, which 30 texts to
  deep-dive on) that aren't code problems.
- [`docs/main.tex`](docs/main.tex) — report skeleton, not yet filled with real results.
