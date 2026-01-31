#!/usr/bin/env python3
"""XTTS v2 — reads JSON from stdin, writes WAV to output_path."""

import os
import sys
import time

# Add repo root to path to import shared utilities
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tts_common

SAMPLE_RATE = 24000


def generate(text: str, reference_audio_path: str, output_path: str) -> str:
    import torch
    from TTS.api import TTS
    from TTS.utils import manage

    # Monkey-patch TOS acceptance to work in subprocess context (issue #5)
    # Since stdin is piped for JSON input, input() calls would fail with EOFError
    manage.ask_tos = lambda model_full_path: True

    if not reference_audio_path:
        print("Error: XTTS v2 requires a reference audio file (~6s).", file=sys.stderr)
        sys.exit(1)

    t_load = time.perf_counter()
    print("Loading XTTS v2...", file=sys.stderr)
    device = tts_common.get_device()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    load_ms = (time.perf_counter() - t_load) * 1000

    t_infer = time.perf_counter()
    print("Generating speech...", file=sys.stderr)
    wav = tts.tts(text=text, speaker_wav=reference_audio_path, language="en")

    audio = wav
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
        reference_audio_path=params.get("reference_audio_path", ""),
        output_path=params["output_path"],
    )
    print(path)
