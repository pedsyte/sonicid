"""SonicID — AI Music Analyzer: genre, mood, BPM, key, energy."""

import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
MODELS_DIR = ROOT / "models"
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
KEEP_SECONDS = 1800  # 30 min
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".opus", ".wma", ".aiff"}

# ── Genre labels ───────────────────────────────────────────────────
GENRE_DORTMUND_LABELS = ['Alternative', 'Blues', 'Electronic', 'Folk/Country', 'Funk/Soul/R&B', 'Jazz', 'Pop', 'Rap/Hip-Hop', 'Rock']
GENRE_ELECTRONIC_LABELS = ['Ambient', 'Drum & Bass', 'House', 'Techno', 'Trance']
GENRE_ROSAMERICA_LABELS = ['Classical', 'Dance', 'Hip-Hop', 'Jazz', 'Pop', 'R&B', 'Rock', 'Speech']

# ── Lazy model loading ─────────────────────────────────────────────
_models_cache: dict[str, object] = {}


def _get_model(name: str):
    """Load TensorFlow prediction model lazily."""
    if name not in _models_cache:
        from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs
        pb = MODELS_DIR / f"{name}.pb"
        if not pb.exists():
            return None
        if name == "discogs-effnet-bs64":
            _models_cache[name] = TensorflowPredictEffnetDiscogs(
                graphFilename=str(pb), output="PartitionedCall:1"
            )
        else:
            _models_cache[name] = TensorflowPredict2D(
                graphFilename=str(pb), output="model/Softmax"
            )
    return _models_cache[name]


def _load_audio(filepath: Path) -> np.ndarray:
    """Load audio as mono 16kHz for effnet, mono 44.1kHz for rhythm."""
    from essentia.standard import MonoLoader
    audio = MonoLoader(filename=str(filepath), sampleRate=16000)()
    return audio


def _load_audio_44k(filepath: Path) -> np.ndarray:
    from essentia.standard import MonoLoader
    return MonoLoader(filename=str(filepath), sampleRate=44100)()


def _extract_embeddings(audio: np.ndarray) -> np.ndarray:
    model = _get_model("discogs-effnet-bs64")
    if model is None:
        raise RuntimeError("Embedding model not found")
    embeddings = model(audio)
    return embeddings


def _classify_genre(embeddings: np.ndarray) -> tuple:
    genres = []
    sub_genres = []

    # Main genre detection (Dortmund — 9 genres)
    model = _get_model("genre_dortmund")
    if model is not None:
        predictions = model(embeddings)
        avg = np.mean(predictions, axis=0)
        for i in np.argsort(avg)[::-1]:
            if i < len(GENRE_DORTMUND_LABELS):
                genres.append({
                    "genre": GENRE_DORTMUND_LABELS[i],
                    "score": round(float(avg[i]) * 100, 1),
                })

    # Rosamerica (8 genres) — as sub-genres / second opinion
    model2 = _get_model("genre_rosamerica")
    if model2 is not None:
        predictions2 = model2(embeddings)
        avg2 = np.mean(predictions2, axis=0)
        for i in np.argsort(avg2)[::-1][:5]:
            if i < len(GENRE_ROSAMERICA_LABELS):
                sub_genres.append({
                    "genre": GENRE_ROSAMERICA_LABELS[i],
                    "subgenre": GENRE_ROSAMERICA_LABELS[i],
                    "score": round(float(avg2[i]) * 100, 1),
                })

    # Electronic sub-genres (if electronic is top)
    is_electronic = genres and genres[0]["genre"] == "Electronic" and genres[0]["score"] > 20
    model3 = _get_model("genre_electronic")
    if model3 is not None and is_electronic:
        predictions3 = model3(embeddings)
        avg3 = np.mean(predictions3, axis=0)
        electronic_subs = []
        for i in np.argsort(avg3)[::-1]:
            if i < len(GENRE_ELECTRONIC_LABELS):
                electronic_subs.append({
                    "genre": "Electronic",
                    "subgenre": GENRE_ELECTRONIC_LABELS[i],
                    "score": round(float(avg3[i]) * 100, 1),
                })
        sub_genres = electronic_subs + sub_genres

    return genres, sub_genres


def _classify_binary(embeddings: np.ndarray, model_name: str, positive_index: int = 0) -> float:
    model = _get_model(model_name)
    if model is None:
        return 0.0
    predictions = model(embeddings)
    avg = np.mean(predictions, axis=0)
    return float(avg[positive_index]) if len(avg) > positive_index else 0.0


def _detect_bpm_key(audio_44k: np.ndarray) -> dict:
    from essentia.standard import (
        RhythmExtractor2013, KeyExtractor, Energy, DynamicComplexity,
        ZeroCrossingRate, Loudness
    )

    # BPM
    rhythm = RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm(audio_44k)

    # Key / Scale
    key_ext = KeyExtractor()
    key, scale, key_strength = key_ext(audio_44k)

    # Energy
    energy_ext = Energy()
    total_energy = energy_ext(audio_44k)

    # Loudness
    loudness_ext = Loudness()
    loudness = loudness_ext(audio_44k)

    # Dynamic range
    dyn_ext = DynamicComplexity()
    dynamic_complexity, loudness_range = dyn_ext(audio_44k)

    return {
        "bpm": round(float(bpm)),
        "bpm_exact": round(float(bpm), 1),
        "beats_count": len(beats),
        "key": key,
        "scale": scale,
        "key_strength": round(float(key_strength), 3),
        "energy": round(float(total_energy), 2),
        "loudness": round(float(loudness), 2),
        "dynamic_complexity": round(float(dynamic_complexity), 2),
    }


def _get_duration(filepath: Path) -> float:
    from essentia.standard import MetadataReader
    try:
        reader = MetadataReader(filename=str(filepath), failOnError=False)
        _, _, _, _, duration, _, _, _ = reader()[:8]
        return round(float(duration), 2)
    except Exception:
        return 0.0


def _human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n} B"


def _analyze_full(filepath: Path) -> dict:
    """Full analysis pipeline."""
    # Load audio at both sample rates
    audio_16k = _load_audio(filepath)
    audio_44k = _load_audio_44k(filepath)

    # Extract embeddings (discogs-effnet)
    embeddings = _extract_embeddings(audio_16k)

    # Genre classification
    genres, sub_genres = _classify_genre(embeddings)

    # Mood / characteristics (binary classifiers)
    # Classes: happy=[happy,non_happy], sad=[non_sad,sad], aggressive=[aggressive,not],
    #          relaxed=[non_relaxed,relaxed], electronic=[electronic,non], acoustic=[acoustic,non]
    moods = {
        "happy": round(_classify_binary(embeddings, "mood_happy", 0) * 100, 1),
        "sad": round(_classify_binary(embeddings, "mood_sad", 1) * 100, 1),
        "aggressive": round(_classify_binary(embeddings, "mood_aggressive", 0) * 100, 1),
        "relaxed": round(_classify_binary(embeddings, "mood_relaxed", 1) * 100, 1),
        "electronic": round(_classify_binary(embeddings, "mood_electronic", 0) * 100, 1),
        "acoustic": round(_classify_binary(embeddings, "mood_acoustic", 0) * 100, 1),
    }

    # Danceability: [danceable, not_danceable]
    danceability = round(_classify_binary(embeddings, "danceability", 0) * 100, 1)

    # Voice / instrumental: [instrumental, voice] — index 1 = voice
    voice_score = _classify_binary(embeddings, "voice_instrumental", 1)
    is_vocal = voice_score > 0.5
    vocal_confidence = round(abs(voice_score - 0.5) * 200, 1)

    # Gender: [female, male] — index 0 = female
    gender_score = _classify_binary(embeddings, "gender", 0) if is_vocal else 0.5
    gender = "female" if gender_score > 0.5 else "male"
    gender_confidence = round(abs(gender_score - 0.5) * 200, 1)

    # Tonal / atonal: [atonal, tonal] — index 1 = tonal
    tonal_score = _classify_binary(embeddings, "tonal_atonal", 1)
    is_tonal = tonal_score > 0.5

    # BPM, Key, Energy
    rhythm_data = _detect_bpm_key(audio_44k)

    # Duration
    duration = _get_duration(filepath)

    # Build Suno-style prompt
    prompt_parts = []
    # Top genres
    for g in genres[:3]:
        if g["score"] > 10:
            prompt_parts.append(g["genre"])
    # Sub-genres
    for sg in sub_genres[:3]:
        if sg["score"] > 15 and sg["subgenre"] not in prompt_parts:
            prompt_parts.append(sg["subgenre"])
    # Dominant moods (>50%)
    mood_names = {"happy": "Happy", "sad": "Sad", "aggressive": "Aggressive",
                  "relaxed": "Relaxed", "electronic": "Electronic", "acoustic": "Acoustic"}
    for mk, mv in sorted(moods.items(), key=lambda x: x[1], reverse=True):
        if mv > 50:
            prompt_parts.append(mood_names.get(mk, mk))
    # BPM
    prompt_parts.append(f"{rhythm_data['bpm']} BPM")
    # Key
    if rhythm_data["key"]:
        scale_str = "minor" if rhythm_data["scale"] == "minor" else "major"
        prompt_parts.append(f"{rhythm_data['key']} {scale_str}")
    # Vocal/Instrumental
    if is_vocal:
        prompt_parts.append(f"{gender} vocals")
    else:
        prompt_parts.append("instrumental")
    # Danceability
    if danceability > 60:
        prompt_parts.append("danceable")
    # Energy level
    if moods.get("aggressive", 0) > 50:
        prompt_parts.append("high energy")
    elif moods.get("relaxed", 0) > 50:
        prompt_parts.append("chill")

    suno_prompt = ", ".join(prompt_parts)

    return {
        "genres": genres,
        "sub_genres": sub_genres,
        "moods": moods,
        "danceability": danceability,
        "vocal": {
            "has_vocals": is_vocal,
            "confidence": vocal_confidence,
            "gender": gender if is_vocal else None,
            "gender_confidence": gender_confidence if is_vocal else None,
        },
        "tonal": is_tonal,
        "rhythm": rhythm_data,
        "duration": duration,
        "suno_prompt": suno_prompt,
    }


# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(title="SonicID")
app.mount("/css", StaticFiles(directory=str(FRONTEND / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND / "js")), name="js")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (FRONTEND / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in AUDIO_EXTS:
        raise HTTPException(400, "Unsupported format. Audio files only.")
    file_id = uuid.uuid4().hex[:12]
    dest = UPLOADS / f"{file_id}{ext}"
    size = 0
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(400, "File too large (max 100 MB)")
            out.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Empty file")
    return {"fileId": f"{file_id}{ext}", "name": file.filename, "size": size, "sizeHuman": _human_size(size)}


@app.post("/api/analyze/{file_id}")
async def analyze(file_id: str):
    # Find file
    safe = Path(file_id).name
    src = UPLOADS / safe
    if not src.exists() or not src.resolve().is_relative_to(UPLOADS.resolve()):
        raise HTTPException(404, "File not found")

    try:
        result = await asyncio.to_thread(_analyze_full, src)
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}") from exc

    return result


@app.delete("/api/file/{file_id}")
async def delete_file(file_id: str):
    safe = Path(file_id).name
    target = UPLOADS / safe
    if target.exists() and target.resolve().is_relative_to(UPLOADS.resolve()):
        target.unlink(missing_ok=True)
    return JSONResponse({"ok": True})


# ── Cleanup ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_cleanup():
    async def cleanup_loop():
        while True:
            await asyncio.sleep(600)
            now = time.time()
            for f in UPLOADS.iterdir():
                if f.is_file() and (now - f.stat().st_mtime) > KEEP_SECONDS:
                    f.unlink(missing_ok=True)
    asyncio.create_task(cleanup_loop())
