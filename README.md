# Conversational Interfaces — Course Project

This repository collects the project work for the **"Interfejsy konwersacyjne" (Conversational
Interfaces)** course at Wrocław University of Science and Technology (PWr), academic year
2026/27. The overarching goal is to design a voice assistant for an electronics retail store
(laptops, phones, cameras, accessories), built up across four stages that each compare
candidate technologies before they are combined into a working interface.

## Stages

| Stage | Topic | Status | Directory |
|---|---|---|---|
| 1 | ASR/STT comparison (Whisper vs Vosk) | Done | [`asr-benchmark/`](asr-benchmark/) |
| 2 | TTS comparison (Edge-TTS vs Piper vs ElevenLabs) | In progress | [`tts-benchmark/`](tts-benchmark/) |
| 3 | Sentiment recognition comparison | In progress | [`sentiment-benchmark/`](sentiment-benchmark/) |
| 4 | Conversational interface design & implementation | Planned | TBD |

## Stage 1: ASR benchmark (done)

Full details, setup, and usage are in [`asr-benchmark/README.md`](asr-benchmark/README.md).
Summary:

- **Systems compared:** Whisper (`openai-whisper`, run locally) vs Vosk (`vosk-model-small-pl-0.22`).
  Vosk was chosen over CMU Sphinx because Sphinx has no Polish acoustic/language model.
- **Shared test set (cohort-wide):** 38 speakers, 216 recordings (190 complete recording/transcript
  pairs, ~46 minutes total), in Polish, hosted in a public Google Drive folder (link kept only in
  `.env`, never in the repo).
- **Dataset convention:** one folder per speaker named `"Firstname Lastname"`, with paired
  `<student_id>-<N>.wav` / `<student_id>-<N>.txt` files, `N = 1..5`:

  | N | Content |
  |---|---|
  | 1–2 | Commands / questions |
  | 3 | Spontaneous speech |
  | 4 | Named entities, numbers, dates |
  | 5 | Recording with background noise |

  Audio is mono WAV, 16 kHz, 16-bit; transcripts are verbatim UTF-8 text.

- **Evaluation criteria and weights:**

  | Criterion | Weight |
  |---|---|
  | WER (Word Error Rate) | 0.25 |
  | Named entity accuracy | 0.05 |
  | Dates / time / numbers accuracy | 0.10 |
  | Short spontaneous utterances | 0.10 |
  | Noise robustness | 0.10 |
  | Intent recognition | 0.15 |
  | Speed (latency, RTF) | 0.10 |
  | Stability and completeness | 0.05 |
  | Ease of integration | 0.10 |

## Stage 2: TTS benchmark (in progress)

Full details, setup, and usage are in [`tts-benchmark/README.md`](tts-benchmark/README.md).
This is a preliminary scaffold (no accounts configured, no corpus synced, no recordings/ratings
yet) — see [`tts-benchmark/docs/WYMAGANIA.md`](tts-benchmark/docs/WYMAGANIA.md) for exact status.
Summary:

- **Systems compared:** Edge-TTS (free, cloud) vs Piper TTS (local, open-source) vs ElevenLabs
  (paid/limited free tier, cloud).
- **Shared test set (cohort-wide):** ~200 texts (5 per student), in Polish, hosted in a public
  Google Drive folder (link kept only in `.env`, never in the repo).
- **Dataset convention:** one folder per person named `"Imię Nazwisko"`, with 5 `.txt` files
  named `<student_id>-<N>-<slug>.txt`:

  | N | Slug | Content |
  |---|---|---|
  | 1 | `info` | Short informational messages |
  | 2 | `qna` | Questions and answers |
  | 3 | `numbers` | Numbers, dates, times, abbreviations |
  | 4 | `long` | Complex / longer sentences |
  | 5 | `unique-foreign` | Proper nouns and foreign words |

- **Evaluation criteria:** pronunciation correctness and generation speed are measured
  objectively; understandability, naturalness, intonation/prosody, fluency, synthesis
  stability, and dialogue suitability are rated on a 5-point Likert scale by human listeners on
  anonymized recordings. The brief gives no official per-criterion weights (unlike Stage 1).

## Stage 3: Sentiment benchmark (in progress)

Full details, setup, and usage are in
[`sentiment-benchmark/README.md`](sentiment-benchmark/README.md). This is a scaffold
(sync, dataset building, classification, and evaluation implemented and smoke-tested on
fixtures) — no real Drive sync or model run has been done yet; see
[`sentiment-benchmark/docs/WYMAGANIA.md`](sentiment-benchmark/docs/WYMAGANIA.md) for exact
status. Summary:

- **Systems compared:** 3 Polish sentiment models (`Voicelab/herbert-base-cased-sentiment`,
  `bardsai/twitter-sentiment-pl-base`, `Horizon-Labs/multilingual-sentiment-small`) and 3
  emotion models (`visegradmedia-emotion/Emotion_RoBERTa_polish6`,
  `nie3e/go-emotions-polish-gpt2-small-v0.0.1`,
  `tabularisai/multilingual-emotion-classification`).
- **Shared test set:** min. 50 texts (5 per student), in Polish, hosted in a public Google
  Drive folder (link kept only in `.env`, never in the repo).
- **Dataset convention:** one folder per person named `"Imię Nazwisko"`, with 5 `.txt`
  files named `<index>-<sentiment>-<emotion>.txt`, where `sentiment` ∈
  {positive, negative, neutral} and `emotion` ∈ {joy, sadness, anger, fear, irony, none}.
- **Evaluation criteria and weights:** proposed (not yet approved/finalized) in
  [`sentiment-benchmark/config.yaml`](sentiment-benchmark/config.yaml) — see that file's
  `evaluation.criteria_weights`.

## Repository layout

```
pwr-interfejsy-konwersacyjne/
  asr-benchmark/         # Stage 1: ASR/STT benchmark (Whisper vs Vosk) — see its own README
  tts-benchmark/         # Stage 2: TTS benchmark (Edge-TTS vs Piper vs ElevenLabs) — see its own README
  sentiment-benchmark/   # Stage 3: sentiment/emotion recognition benchmark — see its own README
  .gitignore       # Root-level ignores: IDE files, Python caches, venvs, .env
  README.md        # This file
```

## Getting started

- **OS:** Windows
- **Python:** 3.15

For stage-specific setup (virtual environment, dependencies, Google Drive sync), see:

- [`asr-benchmark/README.md`](asr-benchmark/README.md) — setup and usage
- [`asr-benchmark/docs/GDRIVE_API_KEY.md`](asr-benchmark/docs/GDRIVE_API_KEY.md) — how to obtain
  a Google Drive API key
- [`tts-benchmark/README.md`](tts-benchmark/README.md) — setup and usage
- [`sentiment-benchmark/README.md`](sentiment-benchmark/README.md) — setup and usage
  (shares the root `.venv`, same as the other stages)

## Deliverables

- **Stages 1–3:** a written report and a presentation comparing the candidate systems/methods
  for that stage.
- **Stage 4:** a working conversational interface, informed by the conclusions from stages 1–3,
  plus a presentation.

## Data & privacy

The shared test set contains voice recordings of course participants, collected solely for
this research/coursework purpose. Recordings, transcripts, and the `.env` file with access
credentials are never committed to this repository.
