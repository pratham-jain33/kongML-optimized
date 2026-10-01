#!/usr/bin/env python3
"""KeySync transcription microservice.

Receives audio + sections, slices with ffmpeg, transcribes with Kong's
high-resolution piano transcription model (PyTorch, CPU), returns note
events as JSON. Runs as a separate Render service so the 512MB RAM is
dedicated to the ML stack (no Next.js competing for memory).

POST /transcribe (multipart/form-data):
  audio: the audio file (mp3/wav/m4a/ogg/flac)
  sections: JSON array like [{"start": 10, "end": 70}] (optional; whole file if omitted)

Response 200: {"notes": [{"start":..,"end":..,"midi":..,"velocity":..}, ...]}
Response 422: {"error": "no notes detected"}  (same contract as the old local script)
Response 400/500: {"error": "..."}
"""

import contextlib
import io
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)
# 25MB upload cap; sections are short slices, not full concerts.
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

# Kong's model expects 16kHz mono.
SAMPLE_RATE = 16000

# Singleton transcriptor, created on first request (lazy so the service
# boots fast and the ~100MB ONNX model only loads when actually needed).
_transcriptor = None
_transcriptor_lock = threading.Lock()


def get_transcriptor():
    global _transcriptor
    if _transcriptor is None:
        with _transcriptor_lock:
            if _transcriptor is None:
                from onnx_notes import OnnxNotes

                model_path = os.environ.get(
                    "KEYSYNC_ONNX_MODEL",
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "kong_notes_2s5_fp32.onnx",
                    ),
                )
                _transcriptor = OnnxNotes(model_path)
    return _transcriptor


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-500:]}")
    return p.stdout


@app.post("/transcribe")
def transcribe():
    if "audio" not in request.files:
        return jsonify(error="missing audio file"), 400
    f = request.files["audio"]
    if not f.filename:
        return jsonify(error="empty audio filename"), 400

    try:
        sections = json.loads(request.form.get("sections", "[]"))
    except json.JSONDecodeError:
        return jsonify(error="sections must be JSON"), 400

    try:
        transcriptor = get_transcriptor()
    except Exception as e:  # noqa: BLE001 - surface as JSON
        return jsonify(error=f"transcription model failed to load: {e}"), 500

    tmpdir = Path(tempfile.mkdtemp(prefix="ks-transcribe-"))
    try:
        src = tmpdir / "input"
        src.write_bytes(f.read())

        # Build the slice list (30s chunks keep peak memory comfortable).
        MAX_CHUNK = 30
        slices: list[tuple[float, float]] = []
        if not sections:
            probe = run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(src),
            ])
            total = float(probe.strip() or 0)
            cur = 0.0
            while cur < total:
                slices.append((cur, min(cur + MAX_CHUNK, total)))
                cur += MAX_CHUNK
            if not slices:
                slices.append((0.0, total))
        else:
            for s in sections:
                cur = float(s["start"])
                end = float(s["end"])
                while cur < end:
                    slices.append((cur, min(cur + MAX_CHUNK, end)))
                    cur += MAX_CHUNK

        import numpy as np
        import soundfile as sf

        notes: list[dict] = []
        for i, (start, end) in enumerate(slices):
            wav = tmpdir / f"chunk-{i}.wav"
            run([
                "ffmpeg", "-y", "-ss", str(start), "-to", str(end),
                "-i", str(src), "-ar", str(SAMPLE_RATE), "-ac", "1", str(wav),
            ])
            audio, sr = sf.read(str(wav), dtype="float32")
            if sr != SAMPLE_RATE:
                raise RuntimeError(f"unexpected sample rate {sr}")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = np.ascontiguousarray(audio, dtype=np.float32)
            # Kong prints per-segment progress; swallow it so logs stay clean.
            with contextlib.redirect_stdout(io.StringIO()):
                result = transcriptor.transcribe(audio)
            for ev in result:
                ns, ne = float(ev["onset_time"]) + start, float(ev["offset_time"]) + start
                pitch = int(ev["midi_note"])
                vel = float(ev["velocity"]) / 127.0
                if ne > ns and 0 <= pitch <= 127:
                    notes.append({
                        "start": round(ns, 3),
                        "end": round(ne, 3),
                        "midi": pitch,
                        "velocity": round(max(0.0, min(1.0, vel)), 3),
                    })
        notes.sort(key=lambda n: n["start"])
        if not notes:
            return jsonify(error="no notes detected in audio"), 422
        return jsonify(notes=notes)
    except subprocess.TimeoutExpired:
        return jsonify(error="transcription timed out"), 500
    except Exception as e:  # noqa: BLE001 - surface as JSON
        return jsonify(error=f"transcription failed: {e}"), 500
    finally:
        # Best-effort cleanup; container disk is ephemeral anyway.
        for p in tmpdir.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass


@app.get("/health")
def health():
    return jsonify(ok=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
