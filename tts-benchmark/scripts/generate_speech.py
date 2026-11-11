"""Generate speech for every text in the corpus with one or more TTS systems.

For each (system, text) pair:
  - synthesizes data/raw/<system>/<person>/<nrindeksu>-<N>-<slug>.wav
  - measures wall-clock generation time (criterion 7: "Szybkość generowania") and
    appends one row to results/latency_tts-NNN.csv

Reads texts via scripts/text_corpus.py (data/texts/processed/, falling back to
data/texts/raw/). Run all systems you want compared in ONE invocation (one
--system flag with multiple values), same convention as asr-benchmark's
`bench.py --whisper-model`, so results/latency_tts-NNN.csv covers all of them.

Each system needs its own setup — see README.md:
  - edge-tts:    pip install edge-tts (no API key, needs internet)
  - piper:       pip install piper-tts + a downloaded Polish .onnx voice model under models/
  - elevenlabs:  pip install elevenlabs + ELEVENLABS_API_KEY in .env
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_corpus import PROJECT_ROOT, TextItem, iter_corpus  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

SUPPORTED_SYSTEMS = ("edge-tts", "piper", "elevenlabs")

# All three picked as neutral adult MALE voices, so the comparison isn't confounded by
# gender/timbre — see README.md "Voice selection" for the rationale and how to verify them.
EDGE_TTS_VOICE = "pl-PL-MarekNeural"  # MS male PL voice
PIPER_MODEL_PATH = MODELS_DIR / "pl_PL-darkman-medium.onnx"  # male PL voice, 22kHz, medium quality
# "Adam" — ElevenLabs premade voice, deep adult male, multilingual-capable. Premade voice IDs
# occasionally get retired/remapped by ElevenLabs — verify it still points to Adam in your
# account's Voice Library before a real run (see README.md).
ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"


class GenerationError(RuntimeError):
    """Raised when a TTS system fails to synthesize a given text."""


def synth_edge_tts(text: str, destination: Path) -> None:
    import asyncio

    import edge_tts

    async def _run():
        communicator = edge_tts.Communicate(text, EDGE_TTS_VOICE)
        await communicator.save(str(destination))

    try:
        asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - surface any edge-tts/network failure uniformly
        raise GenerationError(f"edge-tts failed: {exc}") from exc


def synth_piper(text: str, destination: Path) -> None:
    import subprocess

    if not PIPER_MODEL_PATH.exists():
        raise GenerationError(
            f"Piper model not found at {PIPER_MODEL_PATH}. Download a Polish voice "
            "model (see README.md) and place it under models/."
        )

    try:
        # Piper's own stdin reader expects UTF-8 regardless of the OS locale, but
        # subprocess.run(text=True) on Windows encodes `input` using the console's
        # codepage (cp1250 for Polish Windows) unless told otherwise — so encode to
        # UTF-8 bytes ourselves instead of relying on text mode for stdin.
        result = subprocess.run(
            ["piper", "--model", str(PIPER_MODEL_PATH), "--output_file", str(destination)],
            input=text.encode("utf-8"),
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise GenerationError(
            "piper binary not found on PATH. Install it with 'pip install piper-tts'."
        ) from exc

    if result.returncode != 0:
        raise GenerationError(f"piper failed: {result.stderr.decode('utf-8', errors='replace')}")


def synth_elevenlabs(text: str, destination: Path) -> None:
    import os

    from elevenlabs.client import ElevenLabs

    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise GenerationError(
            "ELEVENLABS_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    client = ElevenLabs(api_key=api_key)
    try:
        audio = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL_ID,
            text=text,
        )
        destination.write_bytes(b"".join(audio))
    except Exception as exc:  # noqa: BLE001 - surface any ElevenLabs API failure uniformly
        raise GenerationError(f"elevenlabs failed: {exc}") from exc


SYNTH_FUNCS = {
    "edge-tts": synth_edge_tts,
    "piper": synth_piper,
    "elevenlabs": synth_elevenlabs,
}


def next_run_number(results_dir: Path) -> int:
    """Find the next free 3-digit run number based on existing latency_tts-NNN.csv files."""
    pattern = re.compile(r"^latency_tts-(\d{3})\.csv$")
    existing = [
        int(match.group(1))
        for path in results_dir.glob("latency_tts-*.csv")
        if (match := pattern.match(path.name))
    ]
    return max(existing, default=0) + 1


def generate_one(system: str, item: TextItem, force: bool) -> dict:
    destination = DATA_RAW_DIR / system / item.person / f"{item.nrindeksu}-{item.n}-{item.slug}.wav"
    text = item.path.read_text(encoding="utf-8")

    if destination.exists() and not force:
        return {
            "system": system,
            "text_id": item.text_id,
            "category": item.category,
            "chars": len(text),
            "seconds": "",
            "chars_per_second": "",
            "status": "skipped (already exists)",
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    synth = SYNTH_FUNCS[system]

    start = time.time()
    try:
        synth(text, destination)
        elapsed = time.time() - start
        return {
            "system": system,
            "text_id": item.text_id,
            "category": item.category,
            "chars": len(text),
            "seconds": round(elapsed, 3),
            "chars_per_second": round(len(text) / elapsed, 1) if elapsed > 0 else "",
            "status": "ok",
        }
    except GenerationError as exc:
        elapsed = time.time() - start
        print(f"  ERROR ({system}, {item.text_id}): {exc}", file=sys.stderr)
        return {
            "system": system,
            "text_id": item.text_id,
            "category": item.category,
            "chars": len(text),
            "seconds": round(elapsed, 3),
            "chars_per_second": "",
            "status": f"error: {exc}",
        }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate speech for the text corpus with one or more TTS systems."
    )
    parser.add_argument(
        "--system",
        nargs="+",
        choices=SUPPORTED_SYSTEMS,
        required=True,
        help="One or more TTS systems to run in this invocation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if the destination .wav already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    load_dotenv()
    args = parse_args(argv)

    items = iter_corpus()
    if not items:
        print(
            "No texts found. Run scripts/sync_drive.py and scripts/preprocess_texts.py first.",
            file=sys.stderr,
        )
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_suffix = f"{next_run_number(RESULTS_DIR):03d}"
    latency_path = RESULTS_DIR / f"latency_tts-{run_suffix}.csv"

    rows = []
    errors = 0
    for system in args.system:
        print(f"\n=== {system} ===")
        for item in items:
            row = generate_one(system, item, args.force)
            rows.append(row)
            if row["status"].startswith("error"):
                errors += 1
            print(f"  {item.text_id}: {row['status']}")

    with latency_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["system", "text_id", "category", "chars", "seconds", "chars_per_second", "status"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {latency_path}")
    print(f"Generated/skipped: {len(rows)}, errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
