"""Latency measurement on a representative subset, simulating a real pipeline.

The offline batch RTF computed by bench.py answers "how fast is processing
relative to recording length", but not the practical UX question: "how long
does the user wait AFTER they stop talking before they get a transcription?"

This script simulates that by feeding audio in real time (sleeping to match
playback speed) and measuring the TAIL LATENCY — the time from when the last
bit of audio "arrives" to when the final transcription is ready:

  - Vosk: audio is fed in small chunks via AcceptWaveform() as it "arrives"
    (native streaming) — tail latency is just the final AcceptWaveform/
    FinalResult() call after the last chunk, which is typically small.
  - Whisper: has no incremental API in this setup, so it can only start once
    the full utterance has "arrived" — tail latency equals the full
    transcribe() call, same as the batch processing time.

This highlights the real architectural difference between the two: Vosk can
start transcribing before the user finishes speaking, Whisper (as used here)
cannot. Run on a small representative subset (a few recordings per category)
since each run takes real wall-clock time equal to the recordings' duration.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import whisper
from vosk import KaldiRecognizer, Model

# Some filenames from Drive contain invisible Unicode (e.g. zero-width space)
# that the Windows console's codepage can't encode — make stdout tolerant.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VOSK_MODEL_PATH = str(PROJECT_ROOT / "models" / "vosk-model-small-pl-0.22")
RESULTS_DIR = PROJECT_ROOT / "results"
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORIES = {
    "1": "Polecenia/pytania (1)",
    "2": "Polecenia/pytania (2)",
    "3": "Wypowiedź spontaniczna",
    "4": "Nazwy własne, liczby, daty",
    "5": "Trudniejsze warunki akustyczne (szum)",
}


def resolve_audio_path(raw_wav_path):
    rel_path = Path(raw_wav_path).relative_to(RAW_DIR)
    processed_path = PROCESSED_DIR / rel_path
    return str(processed_path) if processed_path.exists() else raw_wav_path


def pick_representative_subset(n_per_category):
    wav_files = sorted(glob.glob(os.path.join(RAW_DIR, "*", "*.wav")))
    by_category = {cat: [] for cat in CATEGORIES.values()}
    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        if not os.path.exists(wav_path.replace(".wav", ".txt")):
            continue
        file_idx = filename.split("-")[-1].replace(".wav", "")
        category = CATEGORIES.get(file_idx)
        if category is None or len(by_category[category]) >= n_per_category:
            continue
        by_category[category].append(wav_path)
    return [wav for group in by_category.values() for wav in group]


def measure_vosk_tail_latency(model, audio_path, chunk_duration_s):
    data, samplerate = sf.read(audio_path, dtype="int16")
    if data.ndim > 1:
        data = data.mean(axis=1).astype("int16")
    chunk_size = max(1, int(chunk_duration_s * samplerate))

    rec = KaldiRecognizer(model, samplerate)
    rec.SetWords(True)

    pos = 0
    while pos < len(data):
        chunk = data[pos : pos + chunk_size]
        rec.AcceptWaveform(chunk.tobytes())
        time.sleep(len(chunk) / samplerate)  # simulate audio arriving in real time
        pos += chunk_size

    tail_start = time.time()
    result = json.loads(rec.FinalResult())
    tail_latency = time.time() - tail_start
    return tail_latency, result.get("text", "").strip()


def measure_whisper_tail_latency(model, audio_path, duration_s):
    time.sleep(duration_s)  # simulate waiting for the recording to finish arriving
    tail_start = time.time()
    result = model.transcribe(audio_path, language="pl")
    tail_latency = time.time() - tail_start
    return tail_latency, result["text"].strip()


def next_run_number(results_dir):
    import re as _re
    pattern = _re.compile(r"^latency_pipeline-(\d{3})\.csv$")
    existing = [
        int(match.group(1))
        for path in results_dir.glob("latency_pipeline-*.csv")
        if (match := pattern.match(path.name))
    ]
    return max(existing, default=0) + 1


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Pomiar opóźnienia (od końca wypowiedzi do transkrypcji) w symulowanym rzeczywistym przepływie."
    )
    parser.add_argument(
        "--whisper-model",
        nargs="+",
        choices=whisper.available_models(),
        default=["base"],
        help="Jeden lub więcej rozmiarów modeli Whisper do zmierzenia w jednym przebiegu (domyślnie: base).",
    )
    parser.add_argument(
        "--n-per-category", type=int, default=4,
        help="Ile nagrań na kategorię wziąć do podzbioru reprezentatywnego (domyślnie: 4).",
    )
    parser.add_argument(
        "--chunk-ms", type=int, default=200,
        help="Rozmiar kawałka audio podawanego do Vosk przy symulacji streamingu, w ms (domyślnie: 200).",
    )
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    subset = pick_representative_subset(args.n_per_category)
    if not subset:
        print("Brak nagrań z transkrypcją do testu.", file=sys.stderr)
        return 1

    print(f"Podzbiór reprezentatywny: {len(subset)} nagrań ({args.n_per_category} per kategoria).")

    whisper_system_names = {size: f"Whisper ({size})" for size in args.whisper_model}
    whisper_models = {}
    for size in args.whisper_model:
        print(f"Ładowanie modelu Whisper '{size}' (device={WHISPER_DEVICE})...")
        whisper_models[size] = whisper.load_model(size, device=WHISPER_DEVICE)
    print("Ładowanie modelu Vosk...")
    vosk_model = Model(VOSK_MODEL_PATH)

    rows = []
    for wav_path in subset:
        filename = os.path.basename(wav_path)
        file_idx = filename.split("-")[-1].replace(".wav", "")
        category = CATEGORIES.get(file_idx, "Inne")
        audio_path = resolve_audio_path(wav_path)
        info = sf.info(audio_path)
        duration = info.frames / info.samplerate

        latencies = {}
        for size, model in whisper_models.items():
            latencies[whisper_system_names[size]], _ = measure_whisper_tail_latency(
                model, audio_path, duration
            )
        latencies["Vosk"], _ = measure_vosk_tail_latency(vosk_model, audio_path, args.chunk_ms / 1000)

        print(
            f"{filename} ({category}, {duration:.1f}s): "
            + ", ".join(f"{name}={value:.3f}s" for name, value in latencies.items())
        )
        row = {"Plik": filename, "Kategoria": category, "Czas nagrania [s]": round(duration, 2)}
        for name, value in latencies.items():
            row[f"Opóźnienie {name} [s]"] = round(value, 4)
        rows.append(row)

    system_names = list(whisper_system_names.values()) + ["Vosk"]
    print("\n=== PODSUMOWANIE (opóźnienie od końca wypowiedzi do transkrypcji) ===")
    for name in system_names:
        values = [row[f"Opóźnienie {name} [s]"] for row in rows]
        print(f"{name}: mediana={statistics.median(values):.3f}s, maks={max(values):.3f}s")
    print(
        "Vosk zaczyna przetwarzać audio, zanim użytkownik skończy mówić (streaming) — "
        "jego opóźnienie 'ogonowe' jest więc niezależne od długości nagrania, w przeciwieństwie do Whispera."
    )

    import csv

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_suffix = f"{next_run_number(RESULTS_DIR):03d}"
    out_path = RESULTS_DIR / f"latency_pipeline-{run_suffix}.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ Zapisano: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
