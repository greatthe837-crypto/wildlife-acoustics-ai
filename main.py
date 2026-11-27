import os, tempfile
from typing import List
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="Wildlife Acoustics Analyzer")
app.mount("/static", StaticFiles(directory="static"), name="static")
MODEL_ID = os.getenv("AUDIO_MODEL", "MIT/ast-finetuned-audioset-10-10-0.4593")
_classifier = None

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline("audio-classification", model=MODEL_ID, top_k=5)
    return _classifier

def classify_file(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[0] < sr // 2:
        raise HTTPException(status_code=400, detail="Please upload at least 0.5 seconds of audio.")
    # Average channels to mono. Resample to 16 kHz using linear interpolation.
    mono = audio.mean(axis=1)
    target_sr = 16000
    if sr != target_sr:
        old_t = np.arange(len(mono), dtype=np.float64) / sr
        new_n = max(1, int(len(mono) * target_sr / sr))
        new_t = np.arange(new_n, dtype=np.float64) / target_sr
        mono = np.interp(new_t, old_t, mono).astype(np.float32)
    chunk = 10 * target_sr
    hop = 8 * target_sr
    segments = []
    for start in range(0, len(mono), hop):
        clip = mono[start:start+chunk]
        if len(clip) < target_sr // 2: break
        preds = get_classifier()({"raw": clip, "sampling_rate": target_sr})
        for p in preds[:5]:
            segments.append({"label": str(p["label"]), "score": float(p["score"]),
                             "start_seconds": start / target_sr,
                             "end_seconds": (start + len(clip)) / target_sr})
    # Aggregate strongest label score and keep best segment timestamp.
    best = {}
    for p in segments:
        if p["label"] not in best or p["score"] > best[p["label"]]["score"]:
            best[p["label"]] = p
    ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:8]
    return {"predictions": ranked,
            "summary": f"Scanned {len(mono)/target_sr:.1f} seconds in overlapping segments. These are the model's top sound-class estimates, not verified species identifications."}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    allowed = {"audio/wav","audio/x-wav","audio/mpeg","audio/mp3","audio/mp4","audio/x-m4a","audio/ogg","audio/flac","audio/aac","application/octet-stream"}
    name = (file.filename or "").lower()
    if not (file.content_type in allowed or name.endswith((".wav",".mp3",".m4a",".ogg",".flac",".aac"))):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please use WAV, MP3, M4A, OGG, FLAC, or AAC.")
    data = await file.read()
    if not data: raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > 25 * 1024 * 1024: raise HTTPException(status_code=413, detail="File exceeds the 25 MB limit.")
    suffix = next((s for s in (".wav",".mp3",".m4a",".ogg",".flac",".aac") if name.endswith(s)), ".audio")
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data); path = tmp.name
        return await run_in_threadpool(classify_file, path)
    except HTTPException: raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not analyze this recording. Check audio encoding and server model setup. ({type(exc).__name__})")
    finally:
        if path and os.path.exists(path): os.remove(path)

@app.get("/health")
def health():
    return {"status":"ok","model":MODEL_ID,"model_loaded":_classifier is not None}
