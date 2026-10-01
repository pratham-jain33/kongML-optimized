# KeySync transcription microservice: Python + ffmpeg + Kong's high-resolution
# piano transcription model as fp32 ONNX (onnxruntime, CPU).
# Runs as its own Render service so the 512MB RAM is dedicated to the ML
# stack. Peak RSS ~360MB on a 30s clip: safe with margin.
# No torch anywhere: post-processing is vendored numpy-only in postproc.py.
FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Model weights ship as split parts (GitHub's per-file limit); reassemble.
# ORT finds kong_notes_2s5_fp32.onnx.data next to the .onnx graph automatically.
COPY models/ ./models/
RUN cat models/kong_notes_2s5_fp32.onnx.data.part-* > kong_notes_2s5_fp32.onnx.data && \
    cp models/kong_notes_2s5_fp32.onnx . && \
    test $(stat -c%s kong_notes_2s5_fp32.onnx.data) -eq 98705408 && \
    rm -rf models
COPY app.py onnx_notes.py postproc.py ./
EXPOSE 8000
CMD ["python", "app.py"]
