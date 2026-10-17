# BPL Niezależny Asystent Anty-Cheat

Moduł dedykowany do bezpiecznej filtracji materiałów wideo z mapy Balistic R (Fortnite) przed startem ligi.

### Założenia architektury:
1. **0% Ingerencji w system:** Bot nie posiada dostępu do bazy danych Neon PostgreSQL ani kodu strony WWW.
2. **Asysta, nie automatyzm:** System nie banuje automatycznie i nie wysyła fałszywych werdyktów. Generuje alerty i logi czasowe dla Administratora w celu ręcznej oceny sytuacji.
3. **Optymalizacja zasobów:** Pobiera pliki wideo strumieniowo (w kawałkach po 1024 bajty) i przetwarza je asynchronicznie za pomocą `ThreadPoolExecutor`.
4. **Automatyczne kanały:** Po połączeniu bot provisionuje w każdej gildii kategorię `🏆 BPL ANTI-CHEAT`, publiczny `#ac-magazyn-plikow` i ograniczony do administratorów/bota `#ac-alerty-sedziowskie`. ID kanałów są przechowywane osobno dla każdej gildii w pamięci procesu.

### Testy i plan weryfikacji przed startem

Testy jednostkowe uruchom z katalogu tego repozytorium:

```powershell
python -m unittest discover -s tests -v
```

Zestaw sprawdza strumieniowe pobieranie i twardy limit rozmiaru, obsługę uszkodzonych klipów, zwalnianie dekodera, czarny materiał i przerwy w znacznikach czasu, pracę analizatora poza pętlą zdarzeń, czyszczenie pliku tymczasowego po wyjątku, ograniczenie liczby równoległych zgłoszeń, odmowę dla nie-administratora oraz przywrócenie trwałego widoku po ponownym połączeniu. Testy używają mocków — nie łączą się z Discordem ani Railway.

Przed produkcyjnym startem wykonaj osobny test integracyjny na testowym serwerze i testowym bocie:

1. Zaproś bota na testowy serwer z włączonym Message Content Intent i uprawnieniami Manage Channels, View Channels, Send Messages, Read Message History, Attach Files oraz Manage Messages. Nie używaj kanału `#magazyn-tabel` z istniejącego OCR.
2. Uruchom usługę i sprawdź Railway Deploy Logs. Oczekiwane wpisy to `Created anti-cheat category '🏆 BPL ANTI-CHEAT' in guild <ID>`, `Created #ac-magazyn-plikow in guild <ID>`, `Created restricted #ac-alerty-sedziowskie in guild <ID>` oraz `Anti-cheat channels ready in guild <ID> (archive=<ID>, alerts=<ID>)`. Przy kolejnym reconnectcie powinien pojawić się wpis `Anti-Cheat BPL is ready for manual review assistance in <N> guild(s).`; kanały mają być ponownie użyte, bez tworzenia duplikatów. Zweryfikuj w kliencie Discord, że użytkownicy widzą i mogą pisać w magazynie, zwykli gracze nie widzą alertów, a bot ma dostęp do obu kanałów.
3. W razie braku uprawnień sprawdź wpis `Discord rejected anti-cheat channel provisioning in guild <ID>` oraz błąd Discord. Bot powinien pozostać online; po nadaniu uprawnień wymuś reconnect/restart i potwierdź wpis `Anti-cheat channels ready...`.
4. Wyślij prawidłowy, krótki klip, uszkodzony `.mp4`, pusty klip, plik z rozszerzeniem innym niż `.mp4`/`.mov` i załącznik ponad limit. Sprawdź logi, brak automatycznego werdyktu/bana i brak pozostawionych `verify_*.mp4`.
5. Zewnętrzny URL jest obsługiwany tylko wtedy, gdy prowadzi bezpośrednio do ścieżki `.mp4` lub `.mov`. YouTube/Streamable page URL-e nie są dekodowane jako klipy. Zweryfikuj limit również dla odpowiedzi bez `Content-Length`.
6. Kliknij alert zwykłym kontem: akcja musi zostać odrzucona ephemeral, a wiadomość źródłowa musi pozostać. Kliknij jako administrator: źródłowa wiadomość może zostać usunięta dopiero po potwierdzeniu Discorda. Uruchom bota ponownie przed kliknięciem przycisku w starym alercie i powtórz testy.
7. Test obciążeniowy wykonuj stopniowo na testowym serwerze: zwiększ liczbę równoległych zgłoszeń od 1 do ponad 2. Maksymalnie dwa klipy są pobierane/analizowane jednocześnie; kolejne podczas pełnego obciążenia są logowane jako odrzucone i pozostają w kanale do ponownego zgłoszenia. Monitoruj pamięć RAM, wolne miejsce na dysku, czasy odpowiedzi Discorda i logi wyjątków. Przerwij test przy narastającym użyciu pamięci lub dysku.

Testy jednostkowe nie dowodzą braku wycieków w długim procesie ani niezawodności usług Discord/Railway; obciążenie, uprawnienia i restarty trzeba zweryfikować integracyjnie. Optical flow jest jedynie sygnałem do ręcznej weryfikacji, nie rozpoznaje samodzielnie cheatów i nie powinien być używany do automatycznych kar.

### Wdrożenie na Railway:
1. Utwórz nowe, odizolowane repozytorium na GitHubie i wrzuć tam pliki: `anticheat_bot.py`, `Dockerfile`, `requirements.txt`.
2. Dodaj nową usługę (New Service) na Railway z tego repozytorium.
3. W sekcji **Variables** na Railway dodaj:
   - `DISCORD_BOT_TOKEN` (całkowicie nowy, dedykowany token bota z Developer Portal).

Nie ustawiaj `ANTYCHEAT_MAGAZYN_ID` ani `ANTYCHEAT_ALERTY_ID`; kanały i ich ID są konfigurowane automatycznie po połączeniu z serwerem.
