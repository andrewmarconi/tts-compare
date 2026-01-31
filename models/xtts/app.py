#!/usr/bin/env python3
"""XTTS v2 — reads JSON from stdin, writes WAV to output_path."""

import json
import sys
import time

import numpy as np
import soundfile as sf

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
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    load_ms = (time.perf_counter() - t_load) * 1000

    t_infer = time.perf_counter()
    print("Generating speech...", file=sys.stderr)
    wav = tts.tts(text=text, speaker_wav=reference_audio_path, language="en")

    audio = np.array(wav, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    infer_ms = (time.perf_counter() - t_infer) * 1000

    print("Saving audio...", file=sys.stderr)
    sf.write(output_path, audio, SAMPLE_RATE)
    print(f"TIMING:{json.dumps({'model_load_ms': round(load_ms, 1), 'model_inference_ms': round(infer_ms, 1)})}", file=sys.stderr)
    print("Done.", file=sys.stderr)
    return output_path


if __name__ == "__main__":
    params = json.loads(sys.stdin.read())
    path = generate(
        text=params["text"],
        reference_audio_path=params.get("reference_audio_path", ""),
        output_path=params["output_path"],
    )
    print(path)
