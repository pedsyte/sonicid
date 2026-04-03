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

# ── Genre grouping: sub-genre → top genre ──────────────────────────
GENRE_LABELS: list[str] = []
TOP_GENRES_MAP: dict[str, str] = {}


def _load_genre_labels():
    global GENRE_LABELS, TOP_GENRES_MAP
    meta_path = MODELS_DIR / "genre_discogs400.json"
    if meta_path.exists():
        data = json.loads(meta_path.read_text())
        GENRE_LABELS = data.get("classes", [])
        for label in GENRE_LABELS:
            if "---" in label:
                top, sub = label.split("---", 1)
                TOP_GENRES_MAP[label] = top.strip()
            else:
                TOP_GENRES_MAP[label] = label.strip()


_load_genre_labels()

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


def _classify_genre(embeddings: np.ndarray) -> list[dict]:
    model = _get_model("genre_discogs400")
    if model is None:
        return []
    predictions = model(embeddings)
    avg = np.mean(predictions, axis=0)

    # Top-level genre aggregation
    top_genre_scores: dict[str, float] = {}
    for i, score in enumerate(avg):
        if i < len(GENRE_LABELS):
            label = GENRE_LABELS[i]
            top = TOP_GENRES_MAP.get(label, label)
            top_genre_scores[top] = top_genre_scores.get(top, 0.0) + float(score)

    # Sort by score
    sorted_genres = sorted(top_genre_scores.items(), key=lambda x: x[1], reverse=True)

    # Top sub-genres
    sub_indices = np.argsort(avg)[::-1][:10]
    sub_genres = []
    for idx in sub_indices:
        if idx < len(GENRE_LABELS):
            label = GENRE_LABELS[idx]
            sub = label.split("---", 1)[1].strip() if "---" in label else label
            top = TOP_GENRES_MAP.get(label, "Unknown")
            sub_genres.append({
                "genre": top,
                "subgenre": sub,
                "score": round(float(avg[idx]) * 100, 1),
            })

    result = []
    for genre, score in sorted_genres[:8]:
        result.append({"genre": genre, "score": round(score * 100, 1)})
    return result, sub_genres


def _classify_binary(embeddings: np.ndarray, model_name: str) -> float:
    model = _get_model(model_name)
    if model is None:
        return 0.0
    predictions = model(embeddings)
    avg = np.mean(predictions, axis=0)
    # Index 0 is typically the positive class
    return float(avg[0]) if len(avg) > 0 else 0.0


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
    moods = {
        "happy": round(_classify_binary(embeddings, "mood_happy") * 100, 1),
        "sad": round(_classify_binary(embeddings, "mood_sad") * 100, 1),
        "aggressive": round(_classify_binary(embeddings, "mood_aggressive") * 100, 1),
        "relaxed": round(_classify_binary(embeddings, "mood_relaxed") * 100, 1),
        "electronic": round(_classify_binary(embeddings, "mood_electronic") * 100, 1),
        "acoustic": round(_classify_binary(embeddings, "mood_acoustic") * 100, 1),
    }

    # Danceability
    danceability = round(_classify_binary(embeddings, "danceability") * 100, 1)

    # Voice / instrumental
    voice_score = _classify_binary(embeddings, "voice_instrumental")
    is_vocal = voice_score > 0.5
    vocal_confidence = round(abs(voice_score - 0.5) * 200, 1)

    # Gender (if vocal)
    gender_score = _classify_binary(embeddings, "gender") if is_vocal else 0.5
    gender = "female" if gender_score > 0.5 else "male"
    gender_confidence = round(abs(gender_score - 0.5) * 200, 1)

    # Tonal / atonal
    tonal_score = _classify_binary(embeddings, "tonal_atonal")
    is_tonal = tonal_score > 0.5

    # BPM, Key, Energy
    rhythm_data = _detect_bpm_key(audio_44k)

    # Duration
    duration = _get_duration(filepath)

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
