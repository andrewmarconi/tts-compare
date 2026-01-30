#!/usr/bin/env python3
"""Qwen3-TTS (VoiceDesign) — reads JSON from stdin, writes WAV to output_path."""

import json
import sys
import time

import numpy as np
import soundfile as sf


def _get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate(text: str, description: str, output_path: str) -> str:
    import torch
    from qwen_tts import Qwen3TTSModel

    t_load = time.perf_counter()
    print("Loading Qwen3-TTS...", file=sys.stderr)
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map=_get_device(),
        dtype=torch.bfloat16,
    )
    load_ms = (time.perf_counter() - t_load) * 1000

    instruct = description or "A natural, clear voice"
    t_infer = time.perf_counter()
    print("Generating speech...", file=sys.stderr)
    wavs, sr = model.generate_voice_design(text=text, instruct=instruct)

    audio = np.array(wavs[0], dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    infer_ms = (time.perf_counter() - t_infer) * 1000

    print("Saving audio...", file=sys.stderr)
    sf.write(output_path, audio, sr)
    print(f"TIMING:{json.dumps({'model_load_ms': round(load_ms, 1), 'model_inference_ms': round(infer_ms, 1)})}", file=sys.stderr)
    print("Done.", file=sys.stderr)
    return output_path


if __name__ == "__main__":
    params = json.loads(sys.stdin.read())
    path = generate(
        text=params["text"],
        description=params.get("description", ""),
        output_path=params["output_path"],
    )
    print(path)
