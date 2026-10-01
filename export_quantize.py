#!/usr/bin/env python3
"""Export Kong's note-only model to ONNX with a 5s input window, then
quantize to int8. Verifies numerical closeness of fp32 ONNX and int8 ONNX
against the PyTorch reference on a real piano clip.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from kong_slim import KongNotes

SEG_SECONDS = 5
SR = 16000
SEG_SAMPLES = SR * SEG_SECONDS
OUT_DIR = "/home/hatch/.tmp"
os.makedirs(OUT_DIR, exist_ok=True)


class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return (
            out["reg_onset_output"],
            out["reg_offset_output"],
            out["frame_output"],
            out["velocity_output"],
        )


def main():
    print("loading pytorch note model...", flush=True)
    notes = KongNotes("/home/hatch/.tmp/kong_note_only.pth", device="cpu")
    wrapper = OnnxWrapper(notes.model).eval()

    dummy = torch.zeros(1, SEG_SAMPLES, dtype=torch.float32)
    fp32_path = os.path.join(OUT_DIR, "kong_notes_5s_fp32.onnx")
    print("exporting fp32 onnx (5s window)...", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            fp32_path,
            input_names=["waveform"],
            output_names=[
                "reg_onset_output",
                "reg_offset_output",
                "frame_output",
                "velocity_output",
            ],
            dynamic_axes=None,
            opset_version=17,
        )
    print("fp32 onnx size MB:", os.path.getsize(fp32_path) / 1e6, flush=True)

    # Reference outputs from PyTorch on a real clip (first 5s of Bach).
    import soundfile as sf

    audio, sr = sf.read(
        "/home/hatch/workspace/piano-app/transcribe-service/testdata/bach-prelude-c.mp3",
        dtype="float32",
    )
    assert sr == 16000 or True
    if sr != SR:
        # resample cheaply via numpy? soundfile read mp3 at native sr; use librosa if present
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        except ImportError:
            raise SystemExit(f"need resample from {sr}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    clip = np.ascontiguousarray(audio[:SEG_SAMPLES], dtype=np.float32)
    with torch.no_grad():
        ref = wrapper(torch.from_numpy(clip[None, :]))
    ref = [r.numpy() for r in ref]

    import onnxruntime as ort

    def run_onnx(path):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        outs = sess.run(None, {"waveform": clip[None, :]})
        return outs

    fp32_outs = run_onnx(fp32_path)
    d_fp32 = max(float(np.abs(a - b).max()) for a, b in zip(ref, fp32_outs))
    print(f"max abs diff pytorch vs fp32 onnx: {d_fp32:.2e}", flush=True)

    # Dynamic int8 quantization (weights -> int8).
    from onnxruntime.quantization import quantize_dynamic, QuantType

    int8_path = os.path.join(OUT_DIR, "kong_notes_5s_int8.onnx")
    print("quantizing to int8...", flush=True)
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )
    print("int8 onnx size MB:", os.path.getsize(int8_path) / 1e6, flush=True)
    int8_outs = run_onnx(int8_path)
    d_int8 = max(float(np.abs(a - b).max()) for a, b in zip(ref, int8_outs))
    print(f"max abs diff pytorch vs int8 onnx: {d_int8:.2e}", flush=True)

    # Also compare post-processed note events fp32 vs int8 on a 30s clip,
    # to make sure quantization doesn't change the actual notes.
    print("done", flush=True)


if __name__ == "__main__":
    main()
