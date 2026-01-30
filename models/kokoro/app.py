#!/usr/bin/env python3
"""Kokoro-82M TTS — reads JSON from stdin, writes WAV to output_path."""

import json
import sys
import time

import numpy as np
import soundfile as sf

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
        voice_preset=params.get("voice_preset", ""),
        output_path=params["output_path"],
    )
    print(path)
