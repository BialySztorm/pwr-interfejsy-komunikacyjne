#!/bin/env python3
import argparse
import os
import re
import statistics
import sys
import time
import glob
import json
import csv
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper
from tqdm import tqdm
from vosk import Model, KaldiRecognizer
from jiwer import process_words

# Some filenames from Drive contain invisible Unicode (e.g. zero-width space)
# that the Windows console's codepage can't encode — make stdout tolerant.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

# Vosk działa tylko na CPU, Whisper skorzysta z GPU, jeśli jest dostępne.
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Konfiguracja ścieżek — zakotwiczone względem katalogu projektu (asr-benchmark/),
# niezależnie od katalogu roboczego, z którego skrypt jest odpalany.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BASE_DIR = str(RAW_DIR)
VOSK_MODEL_PATH = str(PROJECT_ROOT / "models" / "vosk-model-small-pl-0.22")
RESULTS_DIR = PROJECT_ROOT / "results"

# Mapowanie kategorii nagrań na podstawie wytycznych
CATEGORIES = {
    "1": "Polecenia/pytania (1)",
    "2": "Polecenia/pytania (2)",
    "3": "Wypowiedź spontaniczna",
    "4": "Nazwy własne, liczby, daty",
    "5": "Trudniejsze warunki akustyczne (szum)"
}


NUMBER_WORDS = {
    "zero": 0, "jeden": 1, "jedna": 1, "jedno": 1, "dwa": 2, "dwie": 2,
    "trzy": 3, "cztery": 4, "pięć": 5, "sześć": 6, "siedem": 7,
    "osiem": 8, "dziewięć": 9, "dziesięć": 10, "jedenaście": 11,
    "dwanaście": 12, "trzynaście": 13, "czternaście": 14, "piętnaście": 15,
    "szesnaście": 16, "siedemnaście": 17, "osiemnaście": 18, "dziewiętnaście": 19,
    "dwadzieścia": 20, "trzydzieści": 30, "czterdzieści": 40,
    "pięćdziesiąt": 50, "sześćdziesiąt": 60, "siedemdziesiąt": 70,
    "osiemdziesiąt": 80, "dziewięćdziesiąt": 90,
    "sto": 100, "dwieście": 200, "trzysta": 300, "czterysta": 400,
    "pięćset": 500, "sześćset": 600, "siedemset": 700,
    "osiemset": 800, "dziewięćset": 900,
}
NUMBER_SCALES = {
    "tysiąc": 10**3, "tysiące": 10**3, "tysięcy": 10**3,
    "milion": 10**6, "miliony": 10**6, "milionów": 10**6,
    "miliard": 10**9, "miliardy": 10**9, "miliardów": 10**9,
}
UNIT_WORDS = {
    "gigabajt": "gb", "gigabajta": "gb", "gigabajty": "gb", "gigabajtów": "gb",
    "terabajt": "tb", "terabajta": "tb", "terabajty": "tb", "terabajtów": "tb",
    "wat": "w", "wata": "w", "waty": "w", "watów": "w",
    "milimetr": "mm", "milimetra": "mm", "milimetry": "mm", "milimetrów": "mm",
    "herc": "hz", "herca": "hz", "herce": "hz", "herców": "hz",
    "złoty": "zł", "złotego": "zł", "złote": "zł", "złotych": "zł",
    "procenta": "procent", "procenty": "procent", "procentów": "procent",
}
ATTACHED_UNIT = re.compile(r"(?<!\w)(-?\d+(?:[,.]\d+)?)(gb|tb|mm|hz|w|zł)\b")
NUMBER_TOKEN = re.compile(r"-?\d+(?:[:,.]\d+)*")

MONTHS = (
    "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca",
    "lipca", "sierpnia", "września", "października", "listopada", "grudnia",
)
DAY_ORDINALS = {
    1: ("pierwszy", "pierwszego", "pierwszym"),
    2: ("drugi", "drugiego", "drugim"),
    3: ("trzeci", "trzeciego", "trzecim"),
}
DAY_ORDINALS.update({
    day: (stem + "y", stem + "ego", stem + "ym")
    for day, stem in enumerate((
        "czwart", "piąt", "szóst", "siódm", "ósm", "dziewiąt", "dziesiąt",
        "jedenast", "dwunast", "trzynast", "czternast", "piętnast",
        "szesnast", "siedemnast", "osiemnast", "dziewiętnast", "dwudziest",
    ), start=4)
})
DAY_ORDINALS[30] = ("trzydziesty", "trzydziestego", "trzydziestym")
for tens, last_unit in ((20, 9), (30, 1)):
    for unit in range(1, last_unit + 1):
        DAY_ORDINALS[tens + unit] = tuple(
            f"{prefix} {suffix}"
            for prefix, suffix in zip(DAY_ORDINALS[tens], DAY_ORDINALS[unit])
        )

DATE_DAYS = {form: str(day) for day, forms in DAY_ORDINALS.items() for form in forms}
DATE_DAYS.update({f"{day:02d}": str(day) for day in range(1, 10)})
DATE_DAYS.update({str(day): str(day) for day in range(1, 32)})
DATE_DAY_PATTERN = "|".join(form.replace(" ", r"\s+") for form in DATE_DAYS)
DATE_PATTERN = re.compile(rf"(?<![\w.,:+-])({DATE_DAY_PATTERN})(?:\.)?\s+({'|'.join(MONTHS)})\b")

HOUR_WORDS = {"godzina", "godzinę", "godzinie", "godziny"}
FEMININE_ORDINALS = {
    "drugi": ("druga", "drugiej", "drugą"),
    "trzeci": ("trzecia", "trzeciej", "trzecią"),
}
HOUR_ORDINALS = {}
for hour in range(1, 24):
    words = DAY_ORDINALS[hour][0].split()
    forms = [FEMININE_ORDINALS.get(word, (word[:-1] + "a", word[:-1] + "ej", word[:-1] + "ą"))
             for word in words]
    HOUR_ORDINALS.update({tuple(parts): hour for parts in zip(*forms)})
NUMERIC_CLOCK = re.compile(r"([01]?\d|2[0-3])([:.])([0-5]\d)")
ORDER_NUMBER = re.compile(r"\b((?:numer|nr\.?)\s+zamówienia\s+)(\d+(?:[ \t]+\d+)+)(?![\w.,:]\d|\w)")


def normalize_date(match):
    day, month = match.groups()
    return f"{DATE_DAYS[' '.join(day.split())]} {month}"


def read_cardinal(tokens, start):
    """Read a Polish cardinal with descending scales and hundreds/tens/units."""
    total = group = 0
    next_limit = 1000
    previous_scale = float("inf")
    end = start

    for index in range(start, len(tokens)):
        token = tokens[index]
        if token in NUMBER_SCALES:
            scale = NUMBER_SCALES[token]
            if scale >= previous_scale:
                break
            total += (group or 1) * scale
            previous_scale = scale
            group = 0
            next_limit = 1000
        elif token in NUMBER_WORDS:
            value = NUMBER_WORDS[token]
            if value == 0:
                if end == start:
                    return 0, start + 1
                break
            if value >= next_limit:
                break
            group += value
            next_limit = 100 if value >= 100 else 10 if value >= 20 else 0
        else:
            break
        end += 1

    return total + group, end


def read_clock(tokens, start, explicit_hour):
    if start >= len(tokens):
        return None, start

    token = tokens[start]
    numeric_clock = NUMERIC_CLOCK.fullmatch(token)
    if numeric_clock and (explicit_hour or numeric_clock[2] == ":"):
        return f"{int(numeric_clock[1]):02d}:{numeric_clock[3]}", start + 1

    for size in (2, 1):
        end = start + size
        if end > len(tokens):
            continue
        hour = HOUR_ORDINALS.get(tuple(tokens[start:end]))
        if hour is not None:
            break
    else:
        if not explicit_hour:
            return None, start
        hour, end = read_cardinal(tokens, start)
        if token.isdecimal() and len(token) <= 2:
            hour, end = int(token), start + 1
        if end == start or hour > 23:
            return None, start

    minute, minute_end = read_cardinal(tokens, end)
    if tokens[end:end + 2] == ["zero", "zero"]:
        minute_end = end + 2
    elif end < len(tokens) and tokens[end].isdecimal() and len(tokens[end]) == 2:
        minute, minute_end = int(tokens[end]), end + 1
    if minute > 59 or (not explicit_hour and minute_end == end):
        return None, start
    return f"{hour:02d}:{minute:02d}", minute_end


def normalize_for_wer(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = DATE_PATTERN.sub(normalize_date, text)
    text = ATTACHED_UNIT.sub(r"\1 \2", text)
    text = re.sub(r"\bram-u\b", "ramu", text)
    text = ORDER_NUMBER.sub(lambda match: match[1] + "".join(match[2].split()), text)
    # Keep numeric separators and use punctuation as a boundary for spoken numbers.
    tokens = re.findall(r"(?<!\w)-?\d+(?:[:,.]\d+)*(?!\w)|\w+|[^\s\w]", text)
    normalized = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in HOUR_WORDS or token == "o":
            clock, end = read_clock(tokens, index + 1, token in HOUR_WORDS)
            if clock is not None:
                normalized.extend((token, clock))
                index = end
                continue
        if token == "%" and normalized and NUMBER_TOKEN.fullmatch(normalized[-1]):
            normalized.append("procent")
            index += 1
            continue
        start = index + (token == "minus")
        value, end = read_cardinal(tokens, start)
        if end > start:
            normalized.append(str(-value if token == "minus" else value))
            index = end
        else:
            if re.fullmatch(r"-?\d+(?:[:,.]\d+)*|\w+", token):
                if token == "nr":
                    token = "numer"
                if normalized and NUMBER_TOKEN.fullmatch(normalized[-1]):
                    token = UNIT_WORDS.get(token, token)
                normalized.append(token)
            index += 1
    return " ".join(normalized)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Whisper vs Vosk na polskich nagraniach.")
    parser.add_argument(
        "--whisper-model",
        nargs="+",
        choices=whisper.available_models(),
        default=["base"],
        help="Jeden lub więcej rozmiarów modeli Whisper do przetestowania w jednym przebiegu (domyślnie: base).",
    )
    return parser.parse_args()

def warmup_whisper_model(model):
    """Jedna pusta transkrypcja, żeby zainicjalizować kernele CUDA/alokacje przed
    pomiarem czasu — bez tego pierwszy plik w benchmarku miałby nieuczciwie
    wysoki czas przetwarzania (koszt inicjalizacji, nie samego rozpoznawania)."""
    silence = np.zeros(16000, dtype=np.float32)
    model.transcribe(silence, language="pl")


def warmup_vosk_model(model):
    rec = KaldiRecognizer(model, 16000)
    rec.AcceptWaveform(b"\x00\x00" * 1600)
    rec.FinalResult()


def load_models(whisper_model_sizes):
    whisper_models = {}
    for size in whisper_model_sizes:
        print(f"Ładowanie modelu Whisper '{size}' (device={WHISPER_DEVICE})...")
        whisper_models[size] = whisper.load_model(size, device=WHISPER_DEVICE)
        warmup_whisper_model(whisper_models[size])

    print("Ładowanie modelu Vosk (CPU, Vosk nie wspiera GPU)...")
    vosk_model = Model(VOSK_MODEL_PATH)
    warmup_vosk_model(vosk_model)

    return whisper_models, vosk_model

def get_audio_info(file_path):
    data, samplerate = sf.read(file_path)
    duration = len(data) / samplerate
    return duration, samplerate


def resolve_audio_path(raw_wav_path):
    """Zwraca ścieżkę do pliku audio, który faktycznie trafi do ASR.

    Jeśli dla danego nagrania istnieje znormalizowana wersja w data/processed/
    (ten sam względny folder/nazwa), używamy jej (jednolity format dla obu
    systemów) — w przeciwym razie wracamy do oryginału z data/raw/.
    """
    rel_path = Path(raw_wav_path).relative_to(RAW_DIR)
    processed_path = PROCESSED_DIR / rel_path
    return str(processed_path) if processed_path.exists() else raw_wav_path

def transcribe_whisper(model, file_path):
    start_time = time.time()
    result = model.transcribe(file_path, language="pl")
    processing_time = time.time() - start_time
    return result["text"].strip(), processing_time

def transcribe_vosk(model, file_path, samplerate):
    rec = KaldiRecognizer(model, samplerate)
    rec.SetWords(True)
    
    data, _ = sf.read(file_path, dtype='int16')
    if len(data.shape) > 1:
        data = data.mean(axis=1).astype('int16')
        
    start_time = time.time()
    rec.AcceptWaveform(data.tobytes())
    res = json.loads(rec.FinalResult())
    processing_time = time.time() - start_time
    
    return res.get("text", "").strip(), processing_time

def next_run_number(results_dir):
    """Find the next free 3-digit run number based on existing wyniki_asr-NNN.csv files."""
    pattern = re.compile(r"^wyniki_asr-(\d{3})\.csv$")
    existing = [
        int(match.group(1))
        for path in results_dir.glob("wyniki_asr-*.csv")
        if (match := pattern.match(path.name))
    ]
    return max(existing, default=0) + 1

def main():
    args = parse_args()
    whisper_system_names = {size: f"Whisper ({size})" for size in args.whisper_model}
    whisper_models, vosk_model = load_models(args.whisper_model)

    csv_data = [] # Lista na wyniki szczegółowe
    summary_data = [] # Lista na wyniki zagregowane
    
    wav_files = glob.glob(os.path.join(BASE_DIR, "*", "*.wav"))

    progress = tqdm(wav_files, desc="Transkrypcja", unit="plik")
    for wav_path in progress:
        progress.set_postfix_str(os.path.basename(wav_path))
        txt_path = wav_path.replace(".wav", ".txt")
        if not os.path.exists(txt_path):
            tqdm.write(f"Brak transkrypcji dla {wav_path}, pomijanie.")
            continue
            
        with open(txt_path, 'r', encoding='utf-8-sig') as f:
            reference_text = f.read().strip()
            
        normalized_reference = normalize_for_wer(reference_text)
        ref_words_count = len(normalized_reference.split())
        if ref_words_count == 0:
            continue
            
        filename = os.path.basename(wav_path)
        file_idx = filename.split('-')[-1].replace('.wav', '')
        category = CATEGORIES.get(file_idx, "Inne")
        
        audio_path = resolve_audio_path(wav_path)
        duration, samplerate = get_audio_info(audio_path)

        # Test Whisper (dla każdego z podanych rozmiarów modelu)
        for size, model in whisper_models.items():
            hyp_whisper, time_whisper = transcribe_whisper(model, audio_path)
            rtf_whisper = time_whisper / duration
            normalized_whisper = normalize_for_wer(hyp_whisper)
            wer_whisper = process_words(normalized_reference, normalized_whisper)

            csv_data.append({
                "Plik": filename,
                "System": whisper_system_names[size],
                "Kategoria": category,
                "Czas nagrania [s]": duration,
                "Czas przetwarzania [s]": time_whisper,
                "RTF": rtf_whisper,
                "WER [%]": wer_whisper.wer * 100,
                "S (Zamiany)": wer_whisper.substitutions,
                "D (Pominięcia)": wer_whisper.deletions,
                "I (Wstawienia)": wer_whisper.insertions,
                "N (Słowa)": ref_words_count,
                "Transkrypcja systemu": hyp_whisper,
                "Transkrypcja referencyjna": reference_text,
                "Transkrypcja systemu do WER": normalized_whisper,
                "Transkrypcja referencyjna do WER": normalized_reference,
            })

        # Test Vosk
        hyp_vosk, time_vosk = transcribe_vosk(vosk_model, audio_path, samplerate)
        rtf_vosk = time_vosk / duration
        normalized_vosk = normalize_for_wer(hyp_vosk)
        wer_vosk = process_words(normalized_reference, normalized_vosk)
        
        csv_data.append({
            "Plik": filename,
            "System": "Vosk",
            "Kategoria": category,
            "Czas nagrania [s]": duration,
            "Czas przetwarzania [s]": time_vosk,
            "RTF": rtf_vosk,
            "WER [%]": wer_vosk.wer * 100,
            "S (Zamiany)": wer_vosk.substitutions,
            "D (Pominięcia)": wer_vosk.deletions,
            "I (Wstawienia)": wer_vosk.insertions,
            "N (Słowa)": ref_words_count,
            "Transkrypcja systemu": hyp_vosk,
            "Transkrypcja referencyjna": reference_text,
            "Transkrypcja systemu do WER": normalized_vosk,
            "Transkrypcja referencyjna do WER": normalized_reference,
        })

    # Zapis szczegółowych wyników do pliku CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_suffix = f"{next_run_number(RESULTS_DIR):03d}"
    csv_filename = RESULTS_DIR / f"wyniki_asr-{run_suffix}.csv"
    if csv_data:
        keys = csv_data[0].keys()
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(csv_data)
        print(f"\n✅ Zapisano szczegółowe wyniki do pliku: {csv_filename}")

    # Agregacja, raportowanie w konsoli i tworzenie danych do podsumowania
    for system_name in list(whisper_system_names.values()) + ["Vosk"]:
        sys_data = [row for row in csv_data if row["System"] == system_name]
        if not sys_data:
            continue
            
        print(f"\n=== WYNIKI DLA {system_name.upper()} ===")
        total_s = sum(row["S (Zamiany)"] for row in sys_data)
        total_d = sum(row["D (Pominięcia)"] for row in sys_data)
        total_i = sum(row["I (Wstawienia)"] for row in sys_data)
        total_n = sum(row["N (Słowa)"] for row in sys_data)
        
        total_dur = sum(row["Czas nagrania [s]"] for row in sys_data)
        total_proc = sum(row["Czas przetwarzania [s]"] for row in sys_data)
        
        global_wer = (total_s + total_d + total_i) / total_n * 100 if total_n > 0 else 0
        global_rtf = total_proc / total_dur if total_dur > 0 else 0

        # RTF to zagregowany wskaźnik (cały czas / całe nagranie) — opóźnienie to
        # czas przetwarzania POJEDYNCZEGO pliku, liczone tu jako mediana i maksimum
        # (RTF ukrywa rozrzut, np. jeden bardzo długi plik dominujący w sumie).
        per_file_times = sorted(row["Czas przetwarzania [s]"] for row in sys_data)
        median_latency = statistics.median(per_file_times)
        max_latency = per_file_times[-1]

        print(f"Globalny WER: {global_wer:.2f}% (Substytucje: {total_s}, Pominięcia: {total_d}, Wstawienia: {total_i}, Słowa referencyjne: {total_n})")
        print(f"Globalny RTF: {global_rtf:.4f} (Czas nagrań: {total_dur:.2f}s, Czas przetwarzania: {total_proc:.2f}s)")
        print(f"Opóźnienie per plik: mediana {median_latency:.3f}s, maks {max_latency:.3f}s")

        summary_row = {
            "System": system_name,
            "Globalny WER [%]": round(global_wer, 2),
            "Globalny RTF": round(global_rtf, 4),
            "Mediana opóźnienia per plik [s]": round(median_latency, 4),
            "Maks. opóźnienie per plik [s]": round(max_latency, 4),
            "Czas nagrań [s]": round(total_dur, 2),
            "Czas przetwarzania [s]": round(total_proc, 2),
            "Substytucje": total_s,
            "Pominięcia": total_d,
            "Wstawienia": total_i,
            "Słowa referencyjne": total_n
        }
        
        print("Średni WER w kategoriach:")
        for cat in CATEGORIES.values():
            cat_data = [row["WER [%]"] for row in sys_data if row["Kategoria"] == cat]
            if cat_data:
                avg_wer = sum(cat_data) / len(cat_data)
                print(f" - {cat}: {avg_wer:.2f}%")
                summary_row[f"WER - {cat} [%]"] = round(avg_wer, 2)
            else:
                summary_row[f"WER - {cat} [%]"] = "Brak danych"
                
        summary_data.append(summary_row)

    # Zapis zagregowanego podsumowania do pliku CSV
    summary_filename = RESULTS_DIR / f"podsumowanie_asr-{run_suffix}.csv"
    if summary_data:
        summary_keys = summary_data[0].keys()
        with open(summary_filename, 'w', newline='', encoding='utf-8-sig') as sum_file:
            sum_writer = csv.DictWriter(sum_file, summary_keys)
            sum_writer.writeheader()
            sum_writer.writerows(summary_data)
        print(f"✅ Zapisano podsumowanie wyników do pliku: {summary_filename}")

if __name__ == "__main__":
    main()
