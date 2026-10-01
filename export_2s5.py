#!/usr/bin/env python3
"""Export Kong's note-only model to fp32 ONNX with a 2.5s input window."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from kong_slim import KongNotes

SEG_SECONDS = 2.5
SR = 16000
SEG_SAMPLES = int(SR * SEG_SECONDS)
OUT = "/home/hatch/.tmp/kong_notes_2s5_fp32.onnx"


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
    print("exporting...", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, OUT,
            input_names=["waveform"],
            output_names=["reg_onset_output", "reg_offset_output",
                          "frame_output", "velocity_output"],
            dynamic_axes=None, opset_version=17,
        )
    print("size MB:", os.path.getsize(OUT) / 1e6, flush=True)

    # sanity: compare one chunk vs pytorch
    import soundfile as sf
    audio, sr = sf.read("testdata/bach-16k-30s.wav", dtype="float32")
    chunk = audio[:SEG_SAMPLES]
    ref = notes.model(torch.from_numpy(chunk[None, :]))
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"waveform": chunk[None, :].astype(np.float32)})
    keys = ["reg_onset_output", "reg_offset_output", "frame_output", "velocity_output"]
    for k, o in zip(keys, outs):
        r = ref[k].detach().numpy()[0]
        print(f"{k}: max abs diff {np.abs(r - o[0]).max():.2e}")


if __name__ == "__main__":
    main()
