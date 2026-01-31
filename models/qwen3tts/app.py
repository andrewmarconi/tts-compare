#!/usr/bin/env python3
"""Qwen3-TTS (VoiceDesign) — reads JSON from stdin, writes WAV to output_path."""

import os
import sys
import time

# Add repo root to path to import shared utilities
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tts_common


def generate(text: str, description: str, output_path: str) -> str:
    import torch
    from qwen_tts import Qwen3TTSModel

    t_load = time.perf_counter()
    print("Loading Qwen3-TTS...", file=sys.stderr)
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map=tts_common.get_device(),
        dtype=torch.bfloat16,
    )
    load_ms = (time.perf_counter() - t_load) * 1000

    instruct = description or "A natural, clear voice"
    t_infer = time.perf_counter()
    print("Generating speech...", file=sys.stderr)
    wavs, sr = model.generate_voice_design(text=text, instruct=instruct)

    audio = wavs[0]
    infer_ms = (time.perf_counter() - t_infer) * 1000

    print("Saving audio...", file=sys.stderr)
    tts_common.save_audio(audio, sr, output_path)
    tts_common.emit_timing(load_ms, infer_ms)
    print("Done.", file=sys.stderr)
    return output_path


if __name__ == "__main__":
    params = tts_common.read_params()
    path = generate(
        text=params["text"],
        description=params.get("description", ""),
        output_path=params["output_path"],
    )
    print(path)
