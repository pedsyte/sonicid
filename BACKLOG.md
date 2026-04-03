# SonicID — BACKLOG

## 2026-04-03 — Инициализация проекта
- **Задача:** Создать сервис AI-анализа музыки
- **Решение:**
  - Выбран стек: Essentia + TensorFlow + discogs-effnet для эмбеддингов
  - Скачаны 12 предобученных моделей (genre, mood, danceability, voice, gender, tonal)
  - Бэкенд: FastAPI с lazy-загрузкой моделей для экономии RAM
  - Фронтенд: тёмная тема, drag & drop, визуализация жанров/настроения/ритма
  - Деплой: systemd (порт 8090) + nginx + Let's Encrypt SSL
  - Домен: sonicid.gornich.fun
  - GitHub: https://github.com/pedsyte/sonicid
