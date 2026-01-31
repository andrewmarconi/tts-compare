#!/usr/bin/env python3
"""CosyVoice 2 — reads JSON from stdin, writes WAV to output_path.

NOTE: CosyVoice requires cloning https://github.com/FunAudioLLM/CosyVoice
and installing its dependencies manually. This is a stub implementation.
"""

import json
import os
import sys
import time

# Add cloned CosyVoice repo and its Matcha-TTS submodule to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_COSYVOICE_ROOT = os.path.join(_HERE, "CosyVoice")
_MATCHA_TTS = os.path.join(_COSYVOICE_ROOT, "third_party", "Matcha-TTS")
for _p in (_REPO_ROOT, _COSYVOICE_ROOT, _MATCHA_TTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tts_common
import torch


def generate(text: str, description: str, reference_audio_path: str, output_path: str) -> str:
    from cosyvoice.cli.cosyvoice import CosyVoice2

    device = tts_common.get_device()
    print(f"Using device: {device}", file=sys.stderr)

    t_load = time.perf_counter()
    print("Loading CosyVoice 2...", file=sys.stderr)
    try:
        model = CosyVoice2(
            "iic/CosyVoice2-0.5B",
            load_jit=False, load_trt=False, fp16=False,
        )
        # Move model to the selected device
        if hasattr(model, 'to'):
            model = model.to(device)
    except Exception as e:
        if "Separator is not found" in str(e) or "chunk exceed the limit" in str(e):
            print("Error: Model files appear corrupted. Try re-downloading:", file=sys.stderr)
            print(f"  rm -rf ~/.cache/modelscope/hub/iic___CosyVoice2-0.5B", file=sys.stderr)
            print("Then run again to re-download.", file=sys.stderr)
        raise
    load_ms = (time.perf_counter() - t_load) * 1000

    if not reference_audio_path:
        print("Error: CosyVoice 2 requires a reference audio file.", file=sys.stderr)
        sys.exit(1)

    print("Loading reference audio...", file=sys.stderr)

    t_infer = time.perf_counter()
    print("Generating speech...", file=sys.stderr)
    # CosyVoice expects a file path, not a pre-loaded tensor
    if description:
        results = list(model.inference_instruct2(text, description, reference_audio_path, stream=False))
    else:
        results = list(model.inference_zero_shot(text, "", reference_audio_path, stream=False))

    audio = results[0]["tts_speech"].squeeze().cpu().numpy()
    sr = model.sample_rate

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
        reference_audio_path=params.get("reference_audio_path", ""),
        output_path=params["output_path"],
    )
    print(path)
