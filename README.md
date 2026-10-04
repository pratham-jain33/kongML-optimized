# kongML-optimized

Flask microservice for low‑memory piano transcription using ONNX.

## Motivation
Kong's high‑resolution piano transcription model requires ~900 MB RAM when run with PyTorch, which exceeds the limits of free‑tier hosting environments (e.g., Render's 512 MB containers). This project rewrites the inference pipeline to use a fp32 ONNX model and chunked processing, eliminating the heavyweight torch dependency and keeping peak memory below 500 MB.

## Tech stack
- Python 3
- Flask (web service)
- ONNX Runtime (CPU inference)
- NumPy & SoundFile (audio handling)
- Docker (containerisation)

## Features
- fp32 ONNX export of the original note‑only model (`export_2s5.py`)
- Fixed‑size 2.5 s non‑overlapping chunks to limit memory usage (`onnx_notes.py`)
- No runtime PyTorch; only lightweight dependencies (`requirements.txt`)
- Vendored NumPy‑only post‑processor (`postproc.py`) identical to upstream
- Simple Flask API (`/health`, `/transcribe`) ready for deployment
- Dockerfile for reproducible container builds

## Installation
```bash
# Clone the repository
git clone https://github.com/pratham-jain33/kongML-optimized.git
cd kongML-optimized

# Install Python dependencies
pip install -r requirements.txt
```

## Usage
Run the service locally:
```bash
python app.py
```
The service listens on port 8000 by default.
Or build and run the Docker image:
```bash
docker build -t kongml-optimized .
docker run -p 8000:8000 kongml-optimized
```

## Build status
No continuous‑integration workflow is defined in this repository. To verify the build locally, build the Docker image (`docker build …`) and start the container; the service should start without errors and respond to `GET /health`.

## Code style
The code follows PEP 8 conventions: 4‑space indentation, snake_case naming, type hints where used, double‑quoted docstrings, and single‑quoted strings elsewhere. No explicit linting or formatting tools are configured in the repository.

## Code example
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@example.wav" \
  -F 'sections=[{"start":10,"end":70}]' \
  -H "Accept: application/json"
```

## API reference
- **GET /health** – Returns HTTP 200 when the ONNX model is loaded.
- **POST /transcribe** – Multipart form upload.
  - `audio` (required): audio file (mp3, wav, m4a, ogg, flac).
  - `sections` (optional): JSON array of `{ "start": float, "end": float }` objects.
  - **Success (200)**: `{"notes":[{"start":...,"end":...,"midi":...,"velocity":...}, ...]}`
  - **Error 400/422/500**: `{"error":"..."}`
- The response format matches the original Kong transcription service.

## Tests
The repository does not contain automated tests.

---

*Created with [repo-doctor](https://prathamjain.com/projects/repo-doctor)*
