# ASR Benchmark

Comparison of two ASR systems for Polish speech: Whisper (`openai-whisper`, local) and Vosk
(`vosk-model-small-pl-0.22`).

## Project structure

```
asr-benchmark/
  data/raw/          # synced from Google Drive, not in git
  data/processed/    # normalized audio (16kHz mono), not in git
  models/            # downloaded ASR models, not in git
  results/           # outputs (detailed dumps gitignored, summaries tracked)
  docs/GDRIVE_API_KEY.md
  docs/WYMAGANIA.md       # requirement-by-requirement status vs the research brief
  docs/INTEGRACJA.md      # descriptive integration comparison (criterion 9)
  docs/main.tex           # the written report
  src/asr_benchmark/bench.py        # main Whisper vs Vosk benchmark
  src/asr_benchmark/bench_takes.py  # same-text-different-speaker / retake comparison
  scripts/sync_drive.py
  scripts/preprocess_audio.py
  scripts/paired_comparison.py      # per-recording WER diff between two systems
  scripts/compute_final_score.py    # weighted final score across all 9 criteria
  scripts/latency_pipeline.py       # simulated real-time pipeline latency
  tests/
```

## Setup

### 1. Create a virtual environment

```powershell
py -3.15 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

By default this installs a CPU-only `torch` (pulled in by `openai-whisper`). If you have an NVIDIA
GPU and want Whisper to run on it, install a CUDA build of `torch` matching your driver afterwards,
e.g.:

```powershell
pip install --index-url https://download.pytorch.org/whl/cu134 torch --upgrade
```

`bench.py` auto-detects CUDA at runtime and falls back to CPU if it's not available. Vosk always
runs on CPU regardless (it has no GPU support).

### 2. Configure `.env`

Copy `.env.example` to `.env` and fill in the two values:

```
GDRIVE_FOLDER_URL=https://drive.google.com/drive/folders/<your-folder-id>
GDRIVE_API_KEY=<your-google-api-key>
```

If you don't have a Google Drive API key yet, see [docs/GDRIVE_API_KEY.md](docs/GDRIVE_API_KEY.md)
for step-by-step instructions.

### 3. Sync the dataset

```powershell
python scripts\sync_drive.py --dry-run   # preview what would be downloaded
python scripts\sync_drive.py             # actually download into data\raw
```

The source Drive folder is expected to contain one subfolder per speaker ("Imię Nazwisko"),
each with paired `<index>-<N>.wav` / `<index>-<N>.txt` files. The script only downloads new
or changed files (compared by MD5) and never deletes local files removed from Drive — it only
reports them.

Options:
- `--dry-run` — list what would be downloaded without downloading
- `--dest <path>` — destination directory (default: `data/raw`)

### 4. Normalize audio to a common format (optional, recommended)

`data/raw/` is a mix of whatever each speaker's phone/mic produced (different sample rates,
mono/stereo, bit depth). To give both ASR systems byte-for-byte identical, uniform input,
convert everything to 16kHz mono PCM WAV under `data/processed/`:

```powershell
python scripts\preprocess_audio.py --dry-run   # preview
python scripts\preprocess_audio.py             # convert (requires ffmpeg on PATH)
```

`bench.py` and `bench_takes.py` automatically use the `data/processed/` version of a recording
when it exists, and fall back to `data/raw/` otherwise — transcripts (`.txt`) always come from
`data/raw/` only. The script also (re)writes `data/processed/raport_zbioru.md` with dataset-wide
counts (speakers, recordings, reference words, total duration).

### 5. Run the main benchmark

```powershell
python src\asr_benchmark\bench.py --whisper-model base large-v3
```

Run all Whisper sizes you want compared in **one** invocation (one `--whisper-model` with
multiple values) — the report tables assume all systems come from the same
`wyniki_asr-NNN.csv` / `podsumowanie_asr-NNN.csv` run. `large-v3` is much slower than `base`.

Produces (in `results/`, `NNN` = run number):
- `wyniki_asr-NNN.csv` — per-recording detail (gitignored, local only)
- `podsumowanie_asr-NNN.csv` — per-system aggregates: global/per-category WER and RTF, noise
  robustness (Δ WER), non-verbal-sound subset WER, named-entity and date/number/time accuracy,
  intent-match accuracy (heuristic), completeness

### 6. (optional) Same-text-different-speaker / retake comparison

```powershell
python src\asr_benchmark\bench_takes.py --whisper-model base large-v3
```

Compares recordings where the same reference text was read by a different speaker, or re-read
by the same speaker as a second take — see `docs/WYMAGANIA.md` for how these are detected.

### 7. Paired per-recording comparison between two systems

```powershell
python scripts\paired_comparison.py results\wyniki_asr-NNN.csv --system-a "Whisper (base)" --system-b "Vosk"
python scripts\paired_comparison.py results\wyniki_asr-NNN.csv --system-a "Whisper (large-v3)" --system-b "Vosk"
```

Use the `NNN` from step 5. Run once per pair of systems you want compared. Reports win/loss
counts and mean/median WER difference per recording; runs a Wilcoxon signed-rank test if `scipy`
is installed.

### 8. Weighted final score (synthesis of all 9 criteria)

```powershell
python scripts\compute_final_score.py results\podsumowanie_asr-NNN.csv --integration "Whisper (base)=60" --integration "Whisper (large-v3)=55" --integration "Vosk=85"
```

Rescales each criterion to a common 0–100 scale (documented in the script's docstring) and
combines them with the weights from the research brief. `--integration` is your own 0–100
judgment call per system (criterion 9 is descriptive, not computable) — see `docs/INTEGRACJA.md`.

### 9. Latency in a simulated real-time pipeline

```powershell
python scripts\latency_pipeline.py --whisper-model base large-v3 --n-per-category 6
```

RTF (from step 5) measures throughput relative to recording length, not the practical "how long
do I wait after I stop talking" latency. This feeds audio in real time to a representative
subset (a few recordings per category) and measures tail latency after the last chunk arrives —
Vosk streams natively (small, length-independent tail latency), Whisper only starts once the
whole recording has "arrived" (tail latency ≈ full `transcribe()` call).
