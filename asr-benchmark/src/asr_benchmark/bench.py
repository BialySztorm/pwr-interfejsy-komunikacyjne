#!/bin/env python3
import os
import time
import glob
import json
import csv
import soundfile as sf
import whisper
from vosk import Model, KaldiRecognizer
from jiwer import process_words

# Konfiguracja ścieżek
BASE_DIR = "./Nagrania testowe/"
VOSK_MODEL_PATH = "./vosk-model-small-pl-0.22/"

# Mapowanie kategorii nagrań na podstawie wytycznych
CATEGORIES = {
    "1": "Polecenia/pytania (1)",
    "2": "Polecenia/pytania (2)",
    "3": "Wypowiedź spontaniczna",
    "4": "Nazwy własne, liczby, daty",
    "5": "Trudniejsze warunki akustyczne (szum)"
}

def load_models():
    print("Ładowanie modelu Whisper...")
    whisper_model = whisper.load_model("base")
    
    print("Ładowanie modelu Vosk...")
    vosk_model = Model(VOSK_MODEL_PATH)
    
    return whisper_model, vosk_model

def get_audio_info(file_path):
    data, samplerate = sf.read(file_path)
    duration = len(data) / samplerate
    return duration, samplerate

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

def main():
    whisper_model, vosk_model = load_models()
    
    csv_data = [] # Lista na wyniki szczegółowe
    summary_data = [] # Lista na wyniki zagregowane
    
    wav_files = glob.glob(os.path.join(BASE_DIR, "*", "*.wav"))
    
    for wav_path in wav_files:
        txt_path = wav_path.replace(".wav", ".txt")
        if not os.path.exists(txt_path):
            print(f"Brak transkrypcji dla {wav_path}, pomijanie.")
            continue
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            reference_text = f.read().strip().lower()
            
        ref_words_count = len(reference_text.split())
        if ref_words_count == 0:
            continue
            
        filename = os.path.basename(wav_path)
        file_idx = filename.split('-')[-1].replace('.wav', '')
        category = CATEGORIES.get(file_idx, "Inne")
        
        duration, samplerate = get_audio_info(wav_path)
        
        # Test Whisper
        hyp_whisper, time_whisper = transcribe_whisper(whisper_model, wav_path)
        rtf_whisper = time_whisper / duration
        wer_whisper = process_words(reference_text, hyp_whisper)
        
        csv_data.append({
            "Plik": filename,
            "System": "Whisper",
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
        hyp_vosk, time_vosk = transcribe_vosk(vosk_model, wav_path, samplerate)
        rtf_vosk = time_vosk / duration
        wer_vosk = process_words(reference_text, hyp_vosk)
        
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
            "Transkrypcja referencyjna": reference_text
        })

    # Zapis szczegółowych wyników do pliku CSV
    csv_filename = "wyniki_asr.csv"
    if csv_data:
        keys = csv_data[0].keys()
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(csv_data)
        print(f"\n✅ Zapisano szczegółowe wyniki do pliku: {csv_filename}")

    # Agregacja, raportowanie w konsoli i tworzenie danych do podsumowania
    for system_name in ["Whisper", "Vosk"]:
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
        
        print(f"Globalny WER: {global_wer:.2f}% (Substytucje: {total_s}, Pominięcia: {total_d}, Wstawienia: {total_i}, Słowa referencyjne: {total_n})")
        print(f"Globalny RTF: {global_rtf:.4f} (Czas nagrań: {total_dur:.2f}s, Czas przetwarzania: {total_proc:.2f}s)")
        
        summary_row = {
            "System": system_name,
            "Globalny WER [%]": round(global_wer, 2),
            "Globalny RTF": round(global_rtf, 4),
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
    summary_filename = "podsumowanie_asr.csv"
    if summary_data:
        summary_keys = summary_data[0].keys()
        with open(summary_filename, 'w', newline='', encoding='utf-8-sig') as sum_file:
            sum_writer = csv.DictWriter(sum_file, summary_keys)
            sum_writer.writeheader()
            sum_writer.writerows(summary_data)
        print(f"✅ Zapisano podsumowanie wyników do pliku: {summary_filename}")

if __name__ == "__main__":
    main()
