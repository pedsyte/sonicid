# SonicID — AI Music Analyzer

AI-сервис для анализа аудиофайлов: определение жанра, настроения, BPM, тональности и других характеристик.

**URL:** https://sonicid.gornich.fun

## Возможности

- **Жанр** — 400 поджанров (Discogs taxonomy), агрегация в топ-жанры
- **Настроение** — happy, sad, aggressive, relaxed, electronic, acoustic
- **BPM** — точное определение темпа
- **Тональность** — ключ + мажор/минор
- **Танцевальность** — процент danceability
- **Вокал/Инструментал** — определение наличия вокала + пол вокалиста
- **Энергия, громкость, динамический диапазон**

## Стек

- **Backend:** Python 3.11, FastAPI, Uvicorn
- **ML:** Essentia + TensorFlow (discogs-effnet embeddings)
- **Модели:** genre_discogs400, mood_*, danceability, voice_instrumental, gender, tonal_atonal
- **Frontend:** Vanilla HTML/CSS/JS
- **Сервер:** systemd + nginx + Let's Encrypt

## Как работает

1. Пользователь загружает аудиофайл (MP3, WAV, FLAC, OGG и др.)
2. Аудио загружается на сервер, конвертируется в моно 16kHz
3. Discogs-EffNet извлекает эмбеддинги (audio embeddings)
4. Эмбеддинги подаются в 10+ классификационных моделей
5. Отдельно: RhythmExtractor2013 определяет BPM, KeyExtractor — тональность
6. Результат отображается на фронтенде в реальном времени

## Форматы

MP3, WAV, FLAC, OGG, AAC, M4A, OPUS, WMA, AIFF — до 100 MB

## Структура

```
/opt/sonicid/
├── backend/
│   └── main.py          # FastAPI бэкенд + ML pipeline
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── models/              # Essentia TF модели (.pb)
├── uploads/             # Временные загрузки (автоочистка 30 мин)
├── .gitignore
├── README.md
├── ROADMAP.md
└── BACKLOG.md
```

## Локальный запуск

```bash
pip install fastapi uvicorn essentia-tensorflow numpy
cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8090
```

## Сервер

- systemd: `sonicid.service` (порт 8090)
- nginx: `sonicid.conf` → sonicid.gornich.fun
- SSL: Let's Encrypt (auto-renew)
