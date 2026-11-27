# Wildlife Acoustics — AI sound explorer

Responsive website plus FastAPI audio-analysis backend. It accepts audio, scans overlapping 10-second windows, and returns top sound-class labels with confidence estimates and segment times.

## Run locally
Use Python 3.11+. Install dependencies with `pip install -r requirements.txt`, then run `uvicorn main:app --reload`. Open http://127.0.0.1:8000. The first analysis downloads the configured model and can take several minutes; model inference needs substantial RAM/CPU.

## Deploy to Render
This is a Python web service (not a static site). Create a Render Web Service from this repository. Build command: `pip install -r requirements.txt`; start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`. The included `render.yaml` can be used as a Blueprint. Model loading/inference may exceed free-tier memory or timeout limits; use a suitable paid instance or external inference service for reliable production use.

## Model limitations
Default model: `MIT/ast-finetuned-audioset-10-10-0.4593`, a general AudioSet sound-event classifier. It may recognize broad labels (e.g. bird, insect, frog, animal) but is not a specialist species classifier. Labels are model predictions, not confirmed identities; it can miss quiet/overlapping sounds or confuse similar classes. Exact species identification needs a validated model trained for target species, plus regional context and expert verification. Results should not be used as proof of presence/absence.

## Privacy and security
Uploaded audio is temporarily written to a server-side temporary file for inference and deleted after processing. The app does not save recordings intentionally. For production, add authentication, rate limits, request timeouts, explicit retention policy, and HTTPS. Do not upload sensitive recordings without permission.

## Audio support
Backend decoding depends on the installed `soundfile`/libsndfile build. WAV and FLAC are the most dependable. MP3/M4A/AAC/OGG support may vary by deployment; add FFmpeg or an audio decoder such as `torchaudio` if needed for your environment.
