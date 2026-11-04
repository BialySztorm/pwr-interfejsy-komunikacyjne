# Kryterium 9: Możliwość integracji (ocena opisowa, waga 0.10)

Porównanie Whisper (`openai-whisper`, lokalnie) vs Vosk (`vosk-model-small-pl-0.22`) pod kątem
wdrożenia w asystencie głosowym elektromarketu. Oparte na obserwacjach z tego projektu (patrz
`results/`) oraz dokumentacji obu bibliotek.

## API i sposób integracji

| | Whisper | Vosk |
|---|---|---|
| Interfejs | Python: `model.transcribe(audio)` — jedna funkcja, cały plik na raz | Python: `KaldiRecognizer` + `AcceptWaveform()` na kawałkach bajtów — od razu gotowe pod streaming |
| Tryb pracy | Batch (cały plik/bufor na wejściu) | Natywnie strumieniowy (można podawać audio kawałkami w czasie rzeczywistym, dostać wynik częściowy przed końcem wypowiedzi) |
| Alternatywa w chmurze | Tak — OpenAI API (`gpt-4o-transcribe`/`whisper-1`), jeśli nie chcemy hostować lokalnie | Nie — Vosk jest wyłącznie lokalny, brak oficjalnego API w chmurze |
| Język/SDK | Python, C++, dostępne też porty community (whisper.cpp, faster-whisper, whisper-rs) | Python, Java, C#, Node.js, Android/iOS — oficjalne bindingi dla wielu platform |

Dla asystenta głosowego w sklepie (reakcja na żywo, mikrofon) natywny streaming Vosk jest
wygodniejszy architektonicznie — Whisper wymaga buforowania wypowiedzi (np. do wykrycia końca
mówienia / VAD) przed przekazaniem do modelu, co dodaje opóźnienie.

## Format wyników

- **Whisper**: słownik z `text` (cały tekst), `segments` (podział na fragmenty z `start`/`end`
  w sekundach), opcjonalnie `language` wykryty automatycznie. Brak pewności na poziomie słowa
  (word-level timestamps trzeba włączyć dodatkowo, kosztem czasu).
- **Vosk**: JSON z `text`; przy `SetWords(True)` (używane w tym projekcie) dodatkowo lista
  słów z `start`/`end`/`conf` (confidence) per słowo — łatwiej wyciągnąć pojedyncze encje
  (np. numer zamówienia) z dokładnym timestampem i pewnością rozpoznania.

Dla ekstrakcji intencji/encji (patrz kryteria 2, 3, 6) format Vosk z confidence per słowo jest
bardziej przydatny — można np. odrzucić słowo z niską pewnością, zamiast ufać całej transkrypcji.

## Wymagania sprzętowe

Zmierzone w tym projekcie (`results/podsumowanie_asr-*.csv`, RTF = czas przetwarzania / czas nagrania):

| Model | Rozmiar na dysku | RTF (CPU) | RTF (GPU, RTX 5080) | Pamięć |
|---|---|---|---|---|
| Vosk small-pl-0.22 | ~50 MB | ~0.07–0.08 | — (Vosk nie wspiera GPU) | kilkaset MB RAM |
| Whisper base | ~140 MB | zdecydowanie wolniej niż Vosk na CPU (nie testowane wprost w tym repo) | ~0.03–0.05 | ~1 GB VRAM |
| Whisper large-v3 | ~3 GB | nierealne na CPU w czasie zbliżonym do rzeczywistego | ~0.15 | ~10 GB VRAM |

Vosk działa komfortowo na słabym CPU (np. mini-PC w kiosku sklepowym, bez GPU) — to jego
największa przewaga integracyjna. Whisper poniżej modelu `small`/`medium` wymaga GPU, żeby
nadążyć z przetwarzaniem w czasie bliskim rzeczywistemu; większe modele (`large-v3`) praktycznie
wymuszają dedykowaną kartę graficzną.

## Koszty

- **Vosk**: w pełni darmowy (Apache 2.0), model lokalny, zero kosztów licencji ani
  zapytań/API. Koszt to tylko sprzęt (minimalny, patrz wyżej).
- **Whisper lokalnie** (`openai-whisper`, MIT): też darmowy, ale koszt przesuwa się na sprzęt —
  GPU potrzebne dla większych modeli to realny koszt inwestycyjny per stanowisko/kiosk.
- **Whisper via OpenAI API**: alternatywa bez własnego GPU, ale płatna per minuta audio i wymaga
  stałego połączenia internetowego — nie testowane w tym projekcie (używamy lokalnego
  `openai-whisper`), wspomniane tylko jako opcja.

## Przetwarzanie danych / prywatność

Obie opcje testowane w tym projekcie (`openai-whisper` lokalnie, Vosk lokalnie) przetwarzają
audio **w całości na urządzeniu**, bez wysyłania nagrań do zewnętrznego serwisu — istotne dla
sklepu, gdzie nagrania mogą zawierać dane klientów (zamówienia, adresy, numery). Jedyny
scenariusz z wysyłką danych poza lokalną infrastrukturę to Whisper via OpenAI API (opcja, nie
testowana tutaj) — wybór lokalnego uruchomienia obu modeli eliminuje to ryzyko.

## Podsumowanie pod kątem wdrożenia

- **Jeśli priorytet to niski koszt sprzętu i prostota wdrożenia na słabym urządzeniu** (kiosk,
  mini-PC bez GPU) → Vosk: lżejszy, szybszy na CPU, natywny streaming, ale niższa dokładność
  (patrz `podsumowanie_asr-*.csv`) i słabsze rozpoznawanie nazw marek/modeli.
- **Jeśli priorytet to dokładność i akceptowalny jest koszt GPU** → Whisper (`medium`/`large-v3`):
  wyraźnie niższy WER, lepsze radzenie sobie z szumem i nazwami własnymi, ale wymaga
  buforowania wypowiedzi (brak natywnego streamingu) i droższego sprzętu.
- Oba działają w pełni offline, więc prywatność danych klienta nie jest czynnikiem
  różnicującym — zależy to od wyboru self-hosted Whisper, nie od wyboru Whisper vs Vosk per se.
