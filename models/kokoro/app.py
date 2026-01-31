#!/usr/bin/env python3
"""Kokoro-82M TTS — reads JSON from stdin, writes WAV to output_path."""

import os
import sys
import time

# Add repo root to path to import shared utilities
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tts_common
import numpy as np

SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"


def generate(text: str, voice_preset: str, output_path: str) -> str:
    from kokoro import KPipeline

    t_load = time.perf_counter()
    print("Loading Kokoro-82M...", file=sys.stderr)
    pipeline = KPipeline(lang_code="a")
    load_ms = (time.perf_counter() - t_load) * 1000

    voice = voice_preset or DEFAULT_VOICE
    t_infer = time.perf_counter()
    print(f"Generating speech (voice={voice})...", file=sys.stderr)

    chunks = []
    for _gs, _ps, audio in pipeline(text, voice=voice):
        if audio is not None:
            chunks.append(audio)

    if not chunks:
        print("Error: Kokoro produced no audio.", file=sys.stderr)
        sys.exit(1)

    audio = np.concatenate(chunks)
    infer_ms = (time.perf_counter() - t_infer) * 1000

    print("Saving audio...", file=sys.stderr)
    tts_common.save_audio(audio, SAMPLE_RATE, output_path)
    tts_common.emit_timing(load_ms, infer_ms)
    print("Done.", file=sys.stderr)
    return output_path


if __name__ == "__main__":
    params = tts_common.read_params()
    path = generate(
        text=params["text"],
        voice_preset=params.get("voice_preset", ""),
        output_path=params["output_path"],
    )
    print(path)
