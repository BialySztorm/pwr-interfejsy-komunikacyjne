#!/bin/env python3
"""Benchmark Whisper vs Vosk dla nagrań, które mają po kilka wariantów (takes)
współdzielących jedną transkrypcję.

Plik bazowy `<index>-<N>.wav` z własnym `<index>-<N>.txt` (w tym samym
folderze) to kompletna, samodzielna para obsługiwana przez `bench.py` — ten
skrypt ją ignoruje. Interesują go tylko warianty, które NIE mają własnej
transkrypcji w swoim folderze — to w KAŻDYM przypadku ta sama transkrypcja
nagrana przez INNĄ osobę niż właściciel pliku z tekstem (wymóg badania "ten
sam tekst nagrany przez różną osobę"). Różnica jest tylko w tym, JAK to
zostało zapisane na Drive, bo nie było ustalonej konwencji nazewnictwa:
  - litera-wariant przyklejona do indeksu i/lub numeru kategorii, np.
    `272664b-1.wav` albo `272644-1b.wav` — druga osoba wrzuciła swoje nagranie
    do folderu właściciela tekstu, dopisując literę do nazwy (jej tożsamość
    nie jest znana z samej nazwy pliku);
  - ten sam `<index>-<N>` nagrany przez inną osobę, ale zapisany w JEJ
    WŁASNYM folderze pod oryginalnym (czyimś) indeksem, bez żadnej litery —
    tu tożsamość drugiej osoby jest znana (to właściciel tego folderu).
  W obu przypadkach referencja (`<index>-<N>.txt`) jest szukana w CAŁYM
  korpusie, nie tylko w folderze danego pliku audio.

Dla każdej takiej grupy skrypt transkrybuje też plik oryginalny (ten z własną
transkrypcją), żeby porównać WER wariantu z WER oryginału w tym samym
przebiegu — wynikiem jest dodatkowy plik `porownanie_oryginal_wariant-NNN.csv`
oraz podsumowanie w konsoli (średnia różnica WER per system).
"""
import argparse
import os
import re
import statistics
import sys
import time
import glob
import json
import csv
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

# Litera-wariant może być przyklejona do indeksu ("272664b-1"), do numeru
# kategorii ("272644-1b") albo do obu naraz ("272664b-1b").
RECORDING_NAME_PATTERN = re.compile(
    r"^(?P<index>\d+)(?P<index_take>[A-Za-z]*)-(?P<n>\d+)(?P<n_take>[A-Za-z]*)$"
)

# Niektóre pliki z Drive mają doklejone niewidoczne znaki Unicode (np. zero-width
# space) w nazwie, co psuje parowanie z transkrypcją — usuwamy je przed parsowaniem.
INVISIBLE_CHARS_PATTERN = re.compile(r"[​‌‍﻿]")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Porównanie wariantów nagrań współdzielących jedną transkrypcję (Whisper vs Vosk)."
    )
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


def parse_recording_name(stem):
    """Rozbija nazwę pliku (bez rozszerzenia) na (index, n, take).

    `index` i `n` to wyłącznie cyfrowe części nazwy (np. "272664" i "1") —
    to po nich odnajdowana jest wspólna transkrypcja `<index>-<n>.txt`.
    `take` to litera(y) wariantu zebrane z obu możliwych miejsc (przy indeksie
    i/lub przy numerze kategorii), albo "base" dla pliku bez żadnej litery.
    Zwraca None, jeśli nazwa nie odpowiada oczekiwanemu wzorcowi.
    """
    clean_stem = INVISIBLE_CHARS_PATTERN.sub("", stem)
    match = RECORDING_NAME_PATTERN.match(clean_stem)
    if not match:
        return None
    take = (match.group("index_take") + match.group("n_take")) or "base"
    return match.group("index"), match.group("n"), take


def parse_txt_name(stem):
    """Rozbija nazwę transkrypcji (bez rozszerzenia) na (index, n), albo None."""
    clean_stem = INVISIBLE_CHARS_PATTERN.sub("", stem)
    match = re.match(r"^(?P<index>\d+)-(?P<n>\d+)$", clean_stem)
    if not match:
        return None
    return match.group("index"), match.group("n")


def build_global_txt_index(base_dir):
    """Mapuje (index, n) -> ścieżka .txt, przeszukując CAŁY korpus (wszystkie foldery).

    Transkrypcje są czyste (bez liter wariantu) i jednoznaczne per (index, n) —
    dzięki temu plik audio z innego folderu niż jego własna transkrypcja
    (bo nagrała go inna osoba) i tak trafia na właściwy tekst.
    """
    txt_index = {}
    for txt_path in Path(base_dir).glob("*/*.txt"):
        parsed = parse_txt_name(txt_path.stem)
        if parsed is None:
            continue
        if parsed in txt_index:
            tqdm.write(
                f"Niejednoznaczna transkrypcja dla {parsed[0]}-{parsed[1]}: "
                f"{txt_index[parsed]} oraz {txt_path}, zostaje pierwsza."
            )
            continue
        txt_index[parsed] = txt_path
    return txt_index


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
    return processed_path if processed_path.exists() else Path(raw_wav_path)


def transcribe_whisper(model, file_path):
    start_time = time.time()
    result = model.transcribe(file_path, language="pl")
    processing_time = time.time() - start_time
    return result["text"].strip().lower(), processing_time


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

    return res.get("text", "").strip().lower(), processing_time


def build_recording_plan(base_dir):
    """Grupuje pliki audio po wspólnej transkrypcji i zwraca tylko grupy, które
    mają przynajmniej jeden wariant (litera i/lub inny indeks w tym samym
    folderze) — razem z plikiem oryginalnym tej grupy, jeśli istnieje, żeby
    można było porównać wariant z oryginałem.

    Zwraca (groups, unknown_names, missing_ref), gdzie `groups` to
    {reference_txt_path: {"original": Path|None, "variants": [(Path, etykieta)]}}.
    """
    wav_files = sorted(glob.glob(os.path.join(base_dir, "*", "*.wav")))
    global_txt_index = build_global_txt_index(base_dir)
    groups = {}
    unknown_names = []
    missing_ref = []

    for wav_path in wav_files:
        wav_path_obj = Path(wav_path)
        parsed = parse_recording_name(wav_path_obj.stem)
        if parsed is None:
            unknown_names.append(wav_path)
            continue
        index, n, take = parsed

        own_reference_path = wav_path_obj.with_name(f"{index}-{n}.txt")
        if take == "base" and own_reference_path.exists():
            group = groups.setdefault(own_reference_path, {"original": None, "variants": []})
            group["original"] = wav_path_obj
            continue

        reference_path = global_txt_index.get((index, n))
        if reference_path is None:
            missing_ref.append(wav_path)
            continue

        # W obu przypadkach to inna osoba niż właściciel transkrypcji — litera to
        # nagranie wrzucone do folderu właściciela tekstu (tożsamość nieznana z nazwy),
        # inny folder to nagranie zapisane we własnym folderze pod czyimś indeksem.
        label_parts = []
        if take != "base":
            label_parts.append(take)
        if reference_path.parent != wav_path_obj.parent:
            label_parts.append(f"inna_osoba:{wav_path_obj.parent.name}")
        variant_label = "+".join(label_parts) if label_parts else "base"

        group = groups.setdefault(reference_path, {"original": None, "variants": []})
        group["variants"].append((wav_path_obj, variant_label))

    groups = {ref: group for ref, group in groups.items() if group["variants"]}
    return groups, unknown_names, missing_ref


def next_run_number(results_dir):
    """Znajduje kolejny wolny 3-cyfrowy numer na podstawie istniejących wyniki_asr_takes-NNN.csv."""
    pattern = re.compile(r"^wyniki_asr_takes-(\d{3})\.csv$")
    existing = [
        int(match.group(1))
        for path in results_dir.glob("wyniki_asr_takes-*.csv")
        if (match := pattern.match(path.name))
    ]
    return max(existing, default=0) + 1


def main():
    args = parse_args()
    whisper_system_names = {size: f"Whisper ({size})" for size in args.whisper_model}
    whisper_models, vosk_model = load_models(args.whisper_model)

    csv_data = []        # Lista na wyniki szczegółowe (w tym oryginały potrzebne do porównania)
    summary_data = []    # Lista na wyniki zagregowane (per System)
    variant_data = []    # Lista na wyniki zagregowane per (System, Wariant)
    comparison_data = []  # Lista na porównanie wariant vs oryginał

    groups, unknown_names, missing_ref = build_recording_plan(BASE_DIR)
    for name in unknown_names:
        print(f"Nieznany format nazwy pliku: {name}, pomijanie.")
    for path in missing_ref:
        print(f"Brak transkrypcji dla pliku {path}, pomijanie.")

    # Płaska lista zadań do transkrypcji: (plik, etykieta wariantu, referencja, etykieta grupy, N kategorii)
    items = []
    for reference_path, group in groups.items():
        ref_index, ref_n = parse_txt_name(reference_path.stem)
        group_label = f"{ref_index}-{ref_n}"
        if group["original"] is not None:
            items.append((group["original"], "oryginał", reference_path, group_label, ref_n))
        else:
            print(
                f"Brak pliku oryginalnego dla grupy {group_label} w {reference_path.parent}, "
                "warianty będą ocenione bez porównania do oryginału."
            )
        for wav_path_obj, variant_label in group["variants"]:
            items.append((wav_path_obj, variant_label, reference_path, group_label, ref_n))
    items.sort(key=lambda it: (it[3], it[1]))

    reference_text_cache = {}
    progress = tqdm(items, desc="Transkrypcja (wariantowa)", unit="plik")
    for wav_path_obj, variant_label, reference_path, group_label, ref_n in progress:
        progress.set_postfix_str(wav_path_obj.name)
        wav_path = str(wav_path_obj)

        if reference_path not in reference_text_cache:
            with open(reference_path, 'r', encoding='utf-8') as f:
                reference_text_cache[reference_path] = f.read().strip().lower()
        reference_text = reference_text_cache[reference_path]

        ref_words_count = len(reference_text.split())
        if ref_words_count == 0:
            continue

        filename = wav_path_obj.name
        category = CATEGORIES.get(ref_n, "Inne")
        speaker = wav_path_obj.parent.name

        audio_path = str(resolve_audio_path(wav_path))
        duration, samplerate = get_audio_info(audio_path)

        # Test Whisper (dla każdego z podanych rozmiarów modelu)
        for size, model in whisper_models.items():
            hyp_whisper, time_whisper = transcribe_whisper(model, audio_path)
            rtf_whisper = time_whisper / duration
            wer_whisper = process_words(reference_text, hyp_whisper)

            csv_data.append({
                "Mówca": speaker,
                "Grupa": group_label,
                "Wariant": variant_label,
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
                "Transkrypcja referencyjna": reference_text
            })

        # Test Vosk
        hyp_vosk, time_vosk = transcribe_vosk(vosk_model, audio_path, samplerate)
        rtf_vosk = time_vosk / duration
        wer_vosk = process_words(reference_text, hyp_vosk)

        csv_data.append({
            "Mówca": speaker,
            "Grupa": group_label,
            "Wariant": variant_label,
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
            "Transkrypcja referencyjna": reference_text
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_suffix = f"{next_run_number(RESULTS_DIR):03d}"

    # Zapis szczegółowych wyników do pliku CSV
    csv_filename = RESULTS_DIR / f"wyniki_asr_takes-{run_suffix}.csv"
    if csv_data:
        keys = csv_data[0].keys()
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(csv_data)
        print(f"\n✅ Zapisano szczegółowe wyniki do pliku: {csv_filename}")

    system_names = list(whisper_system_names.values()) + ["Vosk"]
    all_variants = sorted({row["Wariant"] for row in csv_data})

    # Agregacja per System (globalnie i po kategoriach) — jak w bench.py
    for system_name in system_names:
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
        # czas przetwarzania POJEDYNCZEGO pliku, liczone tu jako mediana i maksimum.
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

        # Porównanie wariantów (base, b, c, ...) dla tego systemu
        print("Średni WER w wariantach nagrań:")
        variant_row = {"System": system_name}
        for variant in all_variants:
            variant_wer_data = [row["WER [%]"] for row in sys_data if row["Wariant"] == variant]
            if variant_wer_data:
                avg_wer = sum(variant_wer_data) / len(variant_wer_data)
                print(f" - {variant}: {avg_wer:.2f}% (n={len(variant_wer_data)})")
                variant_row[f"WER - wariant {variant} [%]"] = round(avg_wer, 2)
                variant_row[f"N - wariant {variant}"] = len(variant_wer_data)
            else:
                variant_row[f"WER - wariant {variant} [%]"] = "Brak danych"
                variant_row[f"N - wariant {variant}"] = 0
        variant_data.append(variant_row)

    # Zapis zagregowanego podsumowania (per System) do pliku CSV
    summary_filename = RESULTS_DIR / f"podsumowanie_asr_takes-{run_suffix}.csv"
    if summary_data:
        summary_keys = summary_data[0].keys()
        with open(summary_filename, 'w', newline='', encoding='utf-8-sig') as sum_file:
            sum_writer = csv.DictWriter(sum_file, summary_keys)
            sum_writer.writeheader()
            sum_writer.writerows(summary_data)
        print(f"✅ Zapisano podsumowanie wyników do pliku: {summary_filename}")

    # Zapis porównania wariantów (per System) do pliku CSV
    variant_filename = RESULTS_DIR / f"wariant_asr_takes-{run_suffix}.csv"
    if variant_data:
        variant_keys = variant_data[0].keys()
        with open(variant_filename, 'w', newline='', encoding='utf-8-sig') as var_file:
            var_writer = csv.DictWriter(var_file, variant_keys)
            var_writer.writeheader()
            var_writer.writerows(variant_data)
        print(f"✅ Zapisano porównanie wariantów nagrań do pliku: {variant_filename}")

    # Porównanie każdego wariantu z oryginałem tej samej grupy (ten sam System)
    rows_by_group_system = {}
    for row in csv_data:
        rows_by_group_system.setdefault((row["Grupa"], row["System"]), []).append(row)

    for (grupa, system_name), rows in rows_by_group_system.items():
        original_row = next((r for r in rows if r["Wariant"] == "oryginał"), None)
        if original_row is None:
            continue
        for row in rows:
            if row["Wariant"] == "oryginał":
                continue
            comparison_data.append({
                "Mówca": row["Mówca"],
                "Grupa": grupa,
                "System": system_name,
                "Wariant": row["Wariant"],
                "WER oryginału [%]": round(original_row["WER [%]"], 2),
                "WER wariantu [%]": round(row["WER [%]"], 2),
                "Różnica WER (wariant - oryginał) [p.p.]": round(row["WER [%]"] - original_row["WER [%]"], 2),
                "RTF oryginału": round(original_row["RTF"], 4),
                "RTF wariantu": round(row["RTF"], 4),
            })

    comparison_data.sort(key=lambda row: (row["Grupa"], row["System"], row["Wariant"]))

    if comparison_data:
        print("\n=== PORÓWNANIE WARIANT vs ORYGINAŁ ===")
        for system_name in system_names:
            deltas = [
                row["Różnica WER (wariant - oryginał) [p.p.]"]
                for row in comparison_data
                if row["System"] == system_name
            ]
            if not deltas:
                continue
            avg_delta = sum(deltas) / len(deltas)
            trend = "gorzej" if avg_delta > 0 else ("lepiej" if avg_delta < 0 else "bez zmian")
            print(f"{system_name}: średnia różnica WER = {avg_delta:+.2f} p.p. ({trend} niż oryginał, n={len(deltas)})")

        comparison_filename = RESULTS_DIR / f"porownanie_oryginal_wariant-{run_suffix}.csv"
        comparison_keys = comparison_data[0].keys()
        with open(comparison_filename, 'w', newline='', encoding='utf-8-sig') as cmp_file:
            cmp_writer = csv.DictWriter(cmp_file, comparison_keys)
            cmp_writer.writeheader()
            cmp_writer.writerows(comparison_data)
        print(f"✅ Zapisano porównanie wariant vs oryginał do pliku: {comparison_filename}")


if __name__ == "__main__":
    main()
