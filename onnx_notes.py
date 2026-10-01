#!/usr/bin/env python3
"""Chunked fp32 ONNX inference for Kong's note-only piano transcription model.

Replaces kong_slim.py (PyTorch, ~700MB peak) with a fp32 ONNX model running
2.5-second windows. Peak RSS ~360MB, safe for the 512MB Render free container
with margin to spare. Post-processing is vendored in postproc.py (numpy-only,
MIT-licensed from Kong's piano_transcription_inference), so the service never
imports torch at all.

Interface matches KongNotes: OnnxNotes(model_path).transcribe(audio)
takes mono 16kHz float32 numpy audio and returns est_note_events dicts
with onset_time / offset_time / midi_note / velocity.
"""
import gc
import os

import numpy as np

SEG_SECONDS = 2.5
SR = 16000
SEG_SAMPLES = int(SR * SEG_SECONDS)

# Same thresholds kong_slim.py used with the PyTorch model.
ONSET_THRESHOLD = 0.3
OFFSET_THRESHOLD = 0.3
FRAME_THRESHOLD = 0.1


class OnnxNotes:
    def __init__(self, model_path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2  # free tier has few cores; more threads = more RAM
        # BASIC optimization only: ALL duplicates tensors and costs ~100MB RAM.
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # Shrink the CPU arena: reuse less, return more to the OS.
        opts.add_session_config_entry("arena.extend_strategy", "kSameAsRequested")
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Post-processing is numpy-only and vendored in postproc.py (MIT,
        # from Kong's piano_transcription_inference), so torch is never imported.
        from postproc import (
            FRAMES_PER_SECOND,
            CLASSES_NUM,
            RegressionPostProcessor,
        )

        self.post_processor = RegressionPostProcessor(
            FRAMES_PER_SECOND,
            classes_num=CLASSES_NUM,
            onset_threshold=ONSET_THRESHOLD,
            offset_threshold=OFFSET_THRESHOLD,
            frame_threshold=FRAME_THRESHOLD,
            pedal_offset_threshold=0.2,
        )
        self.frames_per_second = FRAMES_PER_SECOND

    def _run_chunk(self, chunk: np.ndarray) -> dict:
        x = chunk.astype(np.float32)[None, :]
        outs = self.session.run(self.output_names, {self.input_name: x})
        return dict(zip(self.output_names, [o[0] for o in outs]))

    def transcribe(self, audio: np.ndarray) -> list:
        """Transcribe mono 16kHz float32 audio. Returns note event dicts."""
        audio = np.ascontiguousarray(audio, dtype=np.float32).ravel()
        n = len(audio)
        pad = (-n) % SEG_SAMPLES
        if pad:
            audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])

        # Non-overlapping 2.5s chunks; stitch rolls in time order.
        keys = ["reg_onset_output", "reg_offset_output", "frame_output", "velocity_output"]
        acc = {k: [] for k in keys}
        for start in range(0, len(audio), SEG_SAMPLES):
            out = self._run_chunk(audio[start : start + SEG_SAMPLES])
            for k in keys:
                acc[k].append(out[k])
            del out
        output_dict = {k: np.concatenate(acc[k], axis=0) for k in keys}
        del acc
        gc.collect()

        # Trim padding frames: 100 fps.
        keep_frames = int(n / SR * self.frames_per_second)
        for k in output_dict:
            output_dict[k] = output_dict[k][:keep_frames]

        est_note_events, _ = self.post_processor.output_dict_to_midi_events(output_dict)
        return est_note_events


def measure_peak_rss(fn, *args):
    """Run fn and return (result, peak RSS in MB) using resource.getrusage."""
    import resource

    result = fn(*args)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return result, peak_mb


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/.tmp/kong_notes_2s5_fp32.onnx"
    )
    clip = os.path.expanduser(
        "~/workspace/piano-app/transcribe-service/testdata/bach-prelude-c.mp3"
    )
    import soundfile as sf

    audio, sr = sf.read(clip, dtype="float32")
    if sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio[: SR * 30]  # 30s test = 6 chunks

    def run():
        m = OnnxNotes(model_path)
        return m.transcribe(audio)

    notes, peak = measure_peak_rss(run)
    print(f"notes: {len(notes)}, peak RSS: {peak:.0f} MB")
    for ev in notes[:5]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in ev.items()})
