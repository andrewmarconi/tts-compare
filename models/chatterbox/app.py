#!/usr/bin/env python3
"""Chatterbox TTS — reads JSON from stdin, writes WAV to output_path."""

import os
import sys
import time

# Add repo root to path to import shared utilities
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tts_common


def generate(text: str, reference_audio_path: str, output_path: str) -> str:
    from chatterbox.tts import ChatterboxTTS

    t_load = time.perf_counter()
    print("Loading Chatterbox...", file=sys.stderr)
    model = ChatterboxTTS.from_pretrained(device=tts_common.get_device())
    load_ms = (time.perf_counter() - t_load) * 1000

    kwargs: dict = {"text": text, "exaggeration": 0.5}
    if reference_audio_path:
        kwargs["audio_prompt_path"] = reference_audio_path

    t_infer = time.perf_counter()
    print("Generating speech...", file=sys.stderr)
    wav = model.generate(**kwargs)
    audio = wav.squeeze().cpu().numpy()

    infer_ms = (time.perf_counter() - t_infer) * 1000

    print("Saving audio...", file=sys.stderr)
    tts_common.save_audio(audio, model.sr, output_path)
    tts_common.emit_timing(load_ms, infer_ms)
    print("Done.", file=sys.stderr)
    return output_path


if __name__ == "__main__":
    params = tts_common.read_params()
    path = generate(
        text=params["text"],
        reference_audio_path=params.get("reference_audio_path", ""),
        output_path=params["output_path"],
    )
    print(path)
