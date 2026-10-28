# ASR Benchmark

Comparison of two ASR systems for Polish speech: Whisper (via `faster-whisper`) and Vosk.

This stage only sets up the project skeleton and data syncing. Transcription and evaluation
are not implemented yet.

## Project structure

```
asr-benchmark/
  data/raw/          # synced from Google Drive, not in git
  data/processed/    # converted audio, not in git
  models/            # downloaded ASR models, not in git
  results/           # outputs, not in git
  docs/GDRIVE_API_KEY.md
  src/asr_benchmark/
  scripts/sync_drive.py
  tests/
```

## Setup

### 1. Create a virtual environment

```powershell
py -3.15 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

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
