"""Normalize all recordings in data/raw/ to one common audio format in data/processed/.

The dataset in data/raw/ is a mix of sample rates (16/44.1/48 kHz), mono and
stereo, PCM16/24 — because it's whatever each speaker's phone/mic produced.
This script converts every <speaker>/<file>.wav under data/raw/ to 16kHz mono
PCM WAV and writes it to the same relative path under data/processed/, so
both ASR systems can be fed byte-for-byte identical, uniform input.

Transcripts (.txt) are NOT copied — they always stay in data/raw/ only.
Conversion requires the `ffmpeg` binary to be installed and available on PATH.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

# Some filenames from Drive carry invisible Unicode (e.g. zero-width space)
# that would otherwise break exact .wav/.txt name matching.
INVISIBLE_CHARS_PATTERN = re.compile(r"[​‌‍﻿]")

# Some filenames from Drive contain invisible Unicode (e.g. zero-width space)
# that the Windows console's codepage can't encode — make stdout tolerant.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Target format: 16kHz mono PCM (required by Vosk, fine for Whisper too).
WAV_SAMPLE_RATE_HZ = 16000
WAV_CHANNELS = 1


class FfmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg binary is not available on PATH."""


def convert_to_wav(source: Path, destination: Path, samplerate: int, channels: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-ar",
                str(samplerate),
                "-ac",
                str(channels),
                str(destination),
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FfmpegNotFoundError(
            "ffmpeg not found on PATH. Install it (e.g. 'winget install ffmpeg' or "
            "download from https://ffmpeg.org/download.html) to enable audio preprocessing."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed converting {source} to {destination}:\n{result.stderr}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all data/raw/*/*.wav to a common format under data/processed/."
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=WAV_SAMPLE_RATE_HZ,
        help=f"Target sample rate in Hz (default: {WAV_SAMPLE_RATE_HZ}).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=WAV_CHANNELS,
        help=f"Target channel count (default: {WAV_CHANNELS} = mono).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconvert even if the destination file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be converted without actually converting.",
    )
    return parser.parse_args(argv)


def compute_dataset_stats(raw_dir: Path) -> dict:
    """Liczy liczbę mówców, nagrań, par kompletnych i słów w referencjach, z data/raw/."""
    speaker_dirs = sorted(p for p in raw_dir.glob("*") if p.is_dir())
    wav_files = sorted(raw_dir.glob("*/*.wav"))
    txt_files = sorted(raw_dir.glob("*/*.txt"))

    total_duration_s = 0.0
    for wav_path in wav_files:
        info = sf.info(str(wav_path))
        total_duration_s += info.frames / info.samplerate

    total_words = 0
    for txt_path in txt_files:
        text = txt_path.read_text(encoding="utf-8").strip()
        total_words += len(text.split())

    txt_keys = {(p.parent, INVISIBLE_CHARS_PATTERN.sub("", p.stem)) for p in txt_files}
    complete_pairs = sum(
        1 for wav_path in wav_files
        if (wav_path.parent, INVISIBLE_CHARS_PATTERN.sub("", wav_path.stem)) in txt_keys
    )

    return {
        "speakers": len(speaker_dirs),
        "recordings": len(wav_files),
        "transcripts": len(txt_files),
        "complete_pairs": complete_pairs,
        "total_words": total_words,
        "total_duration_s": total_duration_s,
    }


def write_dataset_report(stats: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    duration_min = stats["total_duration_s"] / 60

    lines = [
        "# Raport zbioru danych (data/raw/)",
        "",
        f"Wygenerowano: {generated_at} (przez scripts/preprocess_audio.py)",
        "",
        f"- Liczba mówców: {stats['speakers']}",
        f"- Liczba nagrań (.wav): {stats['recordings']}",
        f"- Liczba transkrypcji (.txt): {stats['transcripts']}",
        f"- Kompletne pary .wav/.txt: {stats['complete_pairs']}",
        f"- Łączna liczba słów w referencjach: {stats['total_words']}",
        f"- Łączny czas nagrań: {stats['total_duration_s']:.1f} s (~{duration_min:.1f} min)",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Zapisano raport zbioru do pliku: {destination}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    source_files = sorted(RAW_DIR.glob("*/*.wav"))
    converted = skipped = errors = 0

    for source_path in source_files:
        rel_path = source_path.relative_to(RAW_DIR)
        destination_path = PROCESSED_DIR / rel_path

        if destination_path.exists() and not args.force:
            skipped += 1
            continue

        if args.dry_run:
            print(f"Would convert: {rel_path}")
            converted += 1
            continue

        print(f"Converting: {rel_path}")
        try:
            convert_to_wav(source_path, destination_path, args.samplerate, args.channels)
            converted += 1
        except FfmpegNotFoundError as exc:
            print(f"Błąd konfiguracji: {exc}", file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(f"Błąd konwersji: {exc}", file=sys.stderr)
            errors += 1

    print("\n--- Podsumowanie ---")
    print(f"Przekonwertowane: {converted}")
    print(f"Pominięte (już istnieją w data/processed): {skipped}")
    print(f"Błędy: {errors}")

    if not args.dry_run:
        stats = compute_dataset_stats(RAW_DIR)
        write_dataset_report(stats, PROCESSED_DIR / "raport_zbioru.md")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
