# Jak zdobyć i ograniczyć klucz API do ElevenLabs

1. Zarejestruj się na https://elevenlabs.io i zaloguj.
2. **Profil → API Keys → Create API Key** (lub bezpośrednio w kreatorze klucza, jeśli
   platforma pyta o uprawnienia przy tworzeniu — patrz niżej).
3. Nazwa: cokolwiek rozpoznawalnego, np. `InterfejsyKonwersacyjne`.
4. **Ogranicz klucz** — ustaw uprawnienia minimalne, potrzebne tylko do tego projektu:

   | Endpoint / sekcja | Uprawnienie |
   |---|---|
   | **Text to Speech** | **Dostęp** — jedyny endpoint, którego używa `scripts/generate_speech.py` (`client.text_to_speech.convert`) |
   | **Głosy** | **Przeczytane** (opcjonalnie) — żeby wylistować dostępne głosy i wybrać `voice_id` bez szukania go w panelu WWW |
   | Wszystko inne (Speech to Speech, Mowa na tekst, Sound Effects, Izolacja audio, Generowanie muzyki/obrazów/wideo, Dubbing, ElevenAgents, Architect, Projects, Audio Native, Kolekcje głosów, Generowanie głosu, Wymuszone dopasowanie, Silnik reklam, Znak wodny, Historia, Modele, Słowniki wymowy, Użytkownik, Przestrzeń robocza, Analiza miejsca pracy, Webhooks, Konta serwisowe, Członkowie grupy, zarządzanie miejscem pracy) | **Brak dostępu** — niepotrzebne temu projektowi, a każde dodatkowe uprawnienie to większa szkoda przy wycieku klucza |

5. **Limity użycia (kredyty)** — zamiast "Nieograniczone", ustaw konkretny limit kredytów na
   okres odświeżania. To chroni przed przypadkowym wyczerpaniem całego planu przez
   pętlę/błąd w skrypcie generującym.
6. **Automatyczne wyłączanie po wycieku** — zostaw włączone (nie zaznaczaj opcji wyłączenia
   tej ochrony). Jeśli klucz przypadkiem trafi do publicznego repo, ElevenLabs sam go wyłączy.
7. **Wygasa po** — 90 dni jest rozsądnym domyślnym wyborem; krótszy okres życia klucza = mniejsze
   ryzyko, jeśli wycieknie niezauważony.
8. **Ogranicz według adresu IP** — opcjonalnie, jeśli generujesz zawsze z tej samej sieci.
9. Wklej klucz do `.env` jako `ELEVENLABS_API_KEY=...`.
10. Wybierz głos w swoim koncie (**Głosy**) i wpisz jego `voice_id` w
    `scripts/generate_speech.py: ELEVENLABS_VOICE_ID` (domyślna wartość to placeholder, który
    nie zadziała).

## Uwagi

- Klucza nie commituj do repozytorium i nie wklejaj na grupę/czat zespołu — każdy członek
  zespołu, który generuje mowę przez ElevenLabs, korzysta z własnego klucza/konta.
- Darmowy/najniższy płatny tier ma limit znaków miesięcznie — przy generowaniu ~200 tekstów
  dla wielu osób w grupie warto to policzyć z wyprzedzeniem (patrz
  `data/texts/processed/raport_zbioru.md` dla łącznej liczby znaków w zbiorze).
