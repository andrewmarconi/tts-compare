#!/usr/bin/env python3
"""Chatterbox TTS — reads JSON from stdin, writes WAV to output_path."""

import json
import sys
import time

import numpy as np
import soundfile as sf


def generate(text: str, reference_audio_path: str, output_path: str) -> str:
    from chatterbox.tts import ChatterboxTTS

    t_load = time.perf_counter()
    print("Loading Chatterbox...", file=sys.stderr)
    model = ChatterboxTTS.from_pretrained(device="cuda")
    load_ms = (time.perf_counter() - t_load) * 1000

    kwargs: dict = {"text": text, "exaggeration": 0.5}
    if reference_audio_path:
        kwargs["audio_prompt_path"] = reference_audio_path

    t_infer = time.perf_counter()
    print("Generating speech...", file=sys.stderr)
    wav = model.generate(**kwargs)
    audio = wav.squeeze().cpu().numpy()

    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    infer_ms = (time.perf_counter() - t_infer) * 1000

    print("Saving audio...", file=sys.stderr)
    sf.write(output_path, audio, model.sr)
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
