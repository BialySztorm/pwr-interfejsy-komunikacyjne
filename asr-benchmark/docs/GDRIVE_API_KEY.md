# Jak zdobyć klucz API do Google Drive

1. Wejdź na https://console.cloud.google.com i zaloguj się kontem Google.
2. Utwórz nowy projekt (selektor projektu u góry → **New Project**, np. nazwa `asr-benchmark`).
3. **APIs & Services → Library** → wyszukaj "Google Drive API" → **Enable**.
4. **APIs & Services → Credentials → Create credentials → API key**; skopiuj wygenerowany klucz.
5. Zalecane: **Edit API key → API restrictions → Restrict key** → zaznacz tylko "Google Drive API" → **Save**
   (klucz nie zadziała wtedy z innymi usługami, jeśli wycieknie).
6. Wklej klucz do `.env` jako `GDRIVE_API_KEY=...`, a link do folderu jako `GDRIVE_FOLDER_URL=...`.
7. Uruchom: `python scripts/sync_drive.py --dry-run`, żeby sprawdzić połączenie.

## Uwagi

- Nie trzeba podpinać karty ani płatności — odczyt publicznego folderu mieści się w darmowym limicie.
- Aktywacja API (krok 3) może potrwać kilka minut. Jeśli dostajesz błąd 403 od razu po włączeniu API, poczekaj
  chwilę i spróbuj ponownie.
- Klucza nie commituj do repozytorium i nie wklejaj na grupę/czat zespołu — każdy członek zespołu generuje
  własny klucz w swoim projekcie GCP.
