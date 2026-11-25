# Sentiment Benchmark

Comparison of three Polish sentiment-analysis models and three emotion-recognition models
(one required, two optional) on a shared set of Polish utterances written and labeled by
course participants.

## Project structure

```
sentiment-benchmark/
  data/raw/          # synced from Google Drive, not in git
  data/dataset.csv   # parsed dataset with reference labels, not in git (see Data & privacy)
  results/           # outputs: predictions, metrics, summary, plots
  docs/GDRIVE_API_KEY.md
  docs/WYMAGANIA.md       # requirement-by-requirement status vs. the research brief
  config.yaml        # model registry, label mappings, evaluation weights
  scripts/sync_drive.py      # syncs the shared text corpus from Google Drive
  scripts/build_dataset.py   # parses data/raw/ into dataset.csv + dataset_stats.json
  scripts/classify.py        # runs one or all models, measuring per-text and batch time
  scripts/evaluate.py        # metrics, subsets, agreement, errors, plots, final score
  scripts/run_all.py         # sync -> build_dataset -> classify (all) -> evaluate
  tests/fixtures/     # small sample set used to sanity-check build_dataset.py
```

## Dataset convention

Shared with the group via a Google Drive folder: each person creates a folder named
`"Imie Nazwisko"` with 5 UTF-8 `.txt` files (one utterance per file, emoji allowed), named:

```
<index>-<sentiment>-<emotion>.txt
```

where `sentiment` is one of `positive`, `negative`, `neutral`, and `emotion` is one of
`joy`, `sadness`, `anger`, `fear`, `irony`, `none`. `scripts/build_dataset.py` parses this
convention; an unrecognized label or malformed filename is skipped with a warning rather
than crashing the pipeline.

## Label mapping decisions

Every model's raw output is mapped into a shared label space before evaluation (see
`config.yaml`, `label_mapping` per model):

- **Sentiment:** `negative` / `neutral` / `positive`. All three candidate sentiment models
  already return these exact names, so the mapping is effectively an identity (kept
  explicit in `config.yaml` for robustness in case a model's raw labels ever change).
- **Emotion:** `joy` / `sadness` / `anger` / `fear` / `none`. `disgust` is merged into
  `anger` (no model we use treats disgust as independent from anger-adjacent hostility in
  a way useful for a conversational interface, and keeping a 6th low-support class would
  thin out the confusion matrix). Multi-label models (`goemotions_pl_gpt2`,
  `tabularisai_multilingual`) pick the highest-scoring raw label, map it, and fall back to
  `none` if that top score is below `multilabel_threshold`.
- **Irony:** no candidate model recognizes irony as a category. It is evaluated as a
  **sentiment-only** subset (reference sentiment is expected to be `negative`); emotion
  metrics exclude the irony subset entirely. `evaluate.py` reports how often each model is
  "fooled" into predicting `positive` on ironic text.

## Setup

### 1. Create a virtual environment

```powershell
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Why 3.14, not 3.15:** as of November 2026, `transformers`' PyPI classifiers only declare
support up to Python 3.14, and its `tokenizers` dependency only recently gained Windows
wheels for 3.15 (still very fresh). `torch` itself has stable 3.15 wheels, but the
transformers/tokenizers side is the riskier link on Windows, so this stage targets 3.14
instead of the 3.15 used elsewhere in this repo. Revisit once `transformers` officially
lists 3.15 support.

Runs on CPU by default; GPU (`--device cuda`) is optional and only changes speed, not
output labels.

### 2. Configure `.env`

Copy `.env.example` to `.env` and fill in:

```
GDRIVE_FOLDER_URL=https://drive.google.com/drive/folders/<your-folder-id>
GDRIVE_API_KEY=<your-google-api-key>
```

If you don't have a Google Drive API key yet (or want to reuse the one from
`asr-benchmark/`), see [docs/GDRIVE_API_KEY.md](docs/GDRIVE_API_KEY.md).

### 3. Sync the shared text corpus

```powershell
python scripts\sync_drive.py --dry-run
python scripts\sync_drive.py
```

Downloads only new or changed files (compared by MD5) into `data\raw\`.

### 4. Build the dataset

```powershell
python scripts\build_dataset.py
```

Parses `data/raw/` into `data/dataset.csv` (id, speaker, index, text, ref_sentiment,
ref_emotion, length_words, has_emoji, has_negation) and `results/dataset_stats.json`
(text count, label distribution, speaker count). Prints a warning if the dataset has
fewer than 50 texts, and for any unknown label, malformed filename, empty file, or
duplicate text — none of these stop the run.

`has_negation` is a simple heuristic: a standalone "nie", "nigdy", "żaden"/"żadna"/
"żadne"/"żadnych", or "bez" anywhere in the text.

### 5. Classify

```powershell
python scripts\classify.py --model all
python scripts\classify.py --model herbert_sentiment   # or any single model name
```

Loads each model once, runs an untimed warmup pass, then measures per-text latency
(one forward pass per utterance) and batch latency (the whole dataset in one batched
pass). Writes `results/<model>/predictions.csv` (id, raw label, raw score distribution,
mapped label, per-text time) and `results/<model>/timing.json` (hardware info + aggregate
timing).

### 6. Evaluate

```powershell
python scripts\evaluate.py
```

Writes:
- `results/summary.json` / `results/summary.md` — comparison table, per-model metrics,
  agreement, and the weighted final score
- `results/errors.csv` — every text where a model disagrees with the reference or with
  another model, for qualitative review
- `results/plots/` — confusion matrices, F1 per class, subset accuracy, and timing charts

### 7. Run everything at once

```powershell
python scripts\run_all.py
```

Runs steps 3-6 in order (`--skip-sync` to reuse an already-synced `data/raw/`,
`--device cuda` to classify on GPU).

## Data & privacy

The shared test set contains written utterances from course participants, collected
solely for this coursework. `data/raw/`, `data/dataset.csv`, and `.env` are never
committed to this repository — only aggregate statistics (`results/dataset_stats.json`)
and model outputs/metrics are tracked.

## Status and results

See [docs/WYMAGANIA.md](docs/WYMAGANIA.md) for requirement-by-requirement status against
the research brief.

## Conclusions (to fill in after a full run)

Questions from the research brief, to answer once `results/summary.md` has real numbers:

- Should detected anger change how the system phrases its response?
- Should detected frustration trigger a handoff to a human consultant?
- Should detected uncertainty trigger additional explanations?
- Should the system factor in emotions detected in the user's previous messages, not just
  the current one?
- What are the consequences of misrecognizing emotion (false positive vs. false negative,
  and for which emotion)?
