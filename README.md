# BPL Niezależny Asystent Anty-Cheat

Moduł dedykowany do bezpiecznej filtracji materiałów wideo z mapy Balistic R (Fortnite) przed startem ligi.

### Założenia architektury:
1. **Trwałe raporty:** Bot zapisuje metryki i werdykty w Neon PostgreSQL przez `asyncpg`. Nie wykonuje automatycznych kar ani nie modyfikuje danych strony WWW.
2. **Asysta, nie automatyzm:** System nie banuje automatycznie. Raporty są sygnałem dla sędziów, którzy zatwierdzają profil sterowania albo oznaczają potwierdzone oszustwo.
3. **Optymalizacja zasobów:** Pobiera pliki wideo strumieniowo (w kawałkach po 1024 bajty) i przetwarza je asynchronicznie za pomocą `ThreadPoolExecutor`.
4. **Automatyczne kanały:** Po połączeniu bot provisionuje w każdej gildii kategorię `🏆 BPL ANTI-CHEAT`, `#ac-magazyn-plikow` przyjmujący zgłoszenia bez dostępu `@everyone` do historii oraz ograniczony do administratorów/bota `#ac-alerty-sedziowskie`. ID kanałów są przechowywane osobno dla każdej gildii w pamięci procesu.
5. **Uczenie z weryfikacji:** Profile KBM i PAD aktualizują średnią oraz wariancję wyłącznie po werdykcie `VERIFIED_CLEAN`. Początkowe odchylenie standardowe wyliczane jest jako `(max_legal_velocity - base_avg_speed) / 3`, więc początkowy próg 3σ odpowiada podanemu limitowi legalnej prędkości. Krótkie skoki lub przerwy czasu klatek są opisywane jako możliwy lag; anomalia wymaga co najmniej czterech kolejnych klatek oraz przekroczenia progu.

### Testy i plan weryfikacji przed startem

Testy jednostkowe uruchom z katalogu tego repozytorium:

```powershell
python -m unittest discover -s tests -v
```

Zestaw sprawdza strumieniowe pobieranie i twardy limit rozmiaru, obsługę uszkodzonych klipów, zwalnianie dekodera, czarny materiał i przerwy w znacznikach czasu, pracę analizatora poza pętlą zdarzeń, czyszczenie pliku tymczasowego po wyjątku, ograniczenie liczby równoległych zgłoszeń, odmowę dla nie-administratora oraz przywrócenie trwałego widoku po ponownym połączeniu. Testy jednostkowe używają mocków i nie wymagają połączenia z Discordem ani Neon.

Przed produkcyjnym startem wykonaj osobny test integracyjny na testowym serwerze i testowym bocie:

1. Zaproś bota na testowy serwer z włączonym Message Content Intent i uprawnieniami Manage Channels, View Channels, Send Messages, Read Message History, Attach Files oraz Manage Messages. Nie używaj kanału `#magazyn-tabel` z istniejącego OCR.
2. Uruchom usługę i sprawdź Railway Deploy Logs. Oczekiwane wpisy to `Created anti-cheat category '🏆 BPL ANTI-CHEAT' in guild <ID>`, `Created #ac-magazyn-plikow in guild <ID>`, `Created restricted #ac-alerty-sedziowskie in guild <ID>` oraz `Anti-cheat channels ready in guild <ID> (archive=<ID>, alerts=<ID>)`. Przy kolejnym reconnectcie powinien pojawić się wpis `Anti-Cheat BPL is ready for manual review assistance in <N> guild(s).`; kanały mają być ponownie użyte, bez tworzenia duplikatów. Zweryfikuj w kliencie Discord, że użytkownicy mogą pisać w magazynie, ale nie mogą odczytywać historii, zwykli gracze nie widzą alertów, a bot ma dostęp do obu kanałów.
3. W razie braku uprawnień sprawdź wpis `Discord rejected anti-cheat channel provisioning in guild <ID>` oraz błąd Discord. Bot powinien pozostać online; po nadaniu uprawnień wymuś reconnect/restart i potwierdź wpis `Anti-cheat channels ready...`.
4. Wyślij prawidłowy, krótki klip, uszkodzony `.mp4`, pusty klip, plik z rozszerzeniem innym niż `.mp4`/`.mov` i załącznik ponad limit. Potwierdź, że przyjęte zgłoszenie jest usuwane z magazynu, a tymczasowe potwierdzenie znika po 15 sekundach; kanał alertów powinien otrzymać raport zarówno dla czystego klipu, jak i wykrytej anomalii. Sprawdź logi, brak automatycznego werdyktu/bana i brak pozostawionych `verify_*.mp4`.
5. Zewnętrzny URL jest obsługiwany tylko wtedy, gdy prowadzi bezpośrednio do ścieżki `.mp4` lub `.mov`. YouTube/Streamable page URL-e nie są dekodowane jako klipy. Zweryfikuj limit również dla odpowiedzi bez `Content-Length`.
6. Kliknij alert zwykłym kontem: akcja musi zostać odrzucona ephemeral, a raport i przyciski muszą pozostać. Jako administrator sprawdź, że każdy werdykt edytuje alert i usuwa przyciski po zapisaniu do Neon. Uruchom bota ponownie przed kliknięciem przycisku w starym alercie i powtórz testy.
7. Zweryfikuj raport w Neon: powinien mieć status `PENDING`, zapisane metryki oraz `message_id`; przetestuj oba werdykty czyste i potwierdzony cheat. Po restarcie otwórz stary oczekujący alert i sprawdź, czy przyciski nadal działają. Potwierdź, że tylko werdykt czysty aktualizuje właściwy profil, a powtórne kliknięcie nie dodaje kolejnej próbki.
8. Test obciążeniowy wykonuj stopniowo na testowym serwerze: zwiększ liczbę równoległych zgłoszeń od 1 do ponad 2. Maksymalnie dwa klipy są pobierane/analizowane jednocześnie; kolejne podczas pełnego obciążenia są logowane jako odrzucone i pozostają w kanale do ponownego zgłoszenia. Monitoruj pamięć RAM, wolne miejsce na dysku, czasy odpowiedzi Discorda i logi wyjątków. Przerwij test przy narastającym użyciu pamięci lub dysku.

Testy jednostkowe nie dowodzą braku wycieków w długim procesie ani niezawodności usług Discord/Railway; obciążenie, uprawnienia i restarty trzeba zweryfikować integracyjnie. Optical flow jest jedynie sygnałem do ręcznej weryfikacji, nie rozpoznaje samodzielnie cheatów i nie powinien być używany do automatycznych kar.

### Wdrożenie na Railway:
1. Utwórz nowe, odizolowane repozytorium na GitHubie i wrzuć tam pliki: `anticheat_bot.py`, `Dockerfile`, `requirements.txt`.
2. Dodaj nową usługę (New Service) na Railway z tego repozytorium.
3. W sekcji **Variables** na Railway dodaj:
   - `DISCORD_BOT_TOKEN` (całkowicie nowy, dedykowany token bota z Developer Portal).
   - `DATABASE_URL` (connection string Neon PostgreSQL; włącz SSL w konfiguracji Neon).

Nie ustawiaj `ANTYCHEAT_MAGAZYN_ID` ani `ANTYCHEAT_ALERTY_ID`; kanały i ich ID są konfigurowane automatycznie po połączeniu z serwerem.
