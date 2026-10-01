# kongML-optimized

Kong's high-resolution piano transcription model, optimized to run inside a
512MB container with zero PyTorch.

The stock model peaks near 900MB RSS on CPU, which kills it on free-tier
hosting. This repo is the full pipeline that gets the same model down to a
verified 428MB peak for a live Flask service, with outputs numerically
identical to the PyTorch original.

## What changed vs stock

1. **fp32 ONNX export** (`export_2s5.py`) — the note-only model exported to
   ONNX with external weights. Max diff vs PyTorch: ~1e-6 on all four output
   rolls (onset, offset, frame, velocity).
2. **2.5-second fixed chunks** (`onnx_notes.py`) — inference runs on short
   windows instead of 10s+ segments, which is where most of the memory went.
3. **No torch at runtime** — `requirements.txt` is flask, numpy, soundfile,
   onnxruntime. Nothing else.
4. **Vendored post-processor** (`postproc.py`) — the numpy-only note
   extraction from Kong's package, so the service never imports torch.
5. **Split weights** (`models/`) — the 98MB weight file is split into parts
   under GitHub's per-file limit and reassembled at Docker build time.

## Rejected: int8 quantization

`export_quantize.py` is kept for the record. int8 crushed the onset and frame
outputs (max diff 0.13 vs fp32, peaks fell below detection thresholds) and
detected zero notes on real piano audio. A smaller model that hears nothing
is not an optimization. fp32 stays.

## Verified numbers

- 30s of Bach: 138 notes, 360MB peak, zero torch modules imported.
- Full Flask service on a 129s upload: 648 notes, ~428MB peak.
- Test upload of 30s audio: ~10s end to end.

## Run it

```bash
docker build -t kongml-optimized .
docker run -p 8000:8000 kongml-optimized
```

Or locally:

```bash
pip install -r requirements.txt
python app.py
```

## API

- `GET /health` — 200 when the model is loaded.
- `POST /transcribe` — multipart upload, `file` field. Returns
  `{"notes": [{"pitch": 60, "start": 1.23, "end": 1.87, "velocity": 72}, ...]}`.

## Files

- `app.py` — Flask service, lazy model load.
- `onnx_notes.py` — chunked ONNX inference (2.5s windows).
- `postproc.py` — note event extraction, vendored numpy-only.
- `export_2s5.py` — PyTorch checkpoint to fp32 ONNX export.
- `export_quantize.py` — int8 experiment (rejected, kept for reference).
- `models/` — ONNX graph + split external weights.

## Credits

Model architecture, weights, and post-processing logic come from
[qiuqiangkong/piano_transcription](https://github.com/qiuqiangkong/piano_transcription)
(MIT license). This repo only optimizes how it is packaged and served.
