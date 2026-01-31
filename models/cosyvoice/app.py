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
_COSYVOICE_ROOT = os.path.join(_HERE, "CosyVoice")
_MATCHA_TTS = os.path.join(_COSYVOICE_ROOT, "third_party", "Matcha-TTS")
for _p in (_COSYVOICE_ROOT, _MATCHA_TTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import soundfile as sf
import torch


def _get_device() -> str:
    """Detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate(text: str, description: str, reference_audio_path: str, output_path: str) -> str:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    import torchaudio

    device = _get_device()
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

    prompt_speech = None
    if reference_audio_path:
        print("Loading reference audio...", file=sys.stderr)
        prompt_speech, _sr = torchaudio.load(reference_audio_path)
        prompt_speech = torchaudio.functional.resample(prompt_speech, _sr, 16000)
        # Move reference audio to the same device as the model
        prompt_speech = prompt_speech.to(device)

    t_infer = time.perf_counter()
    print("Generating speech...", file=sys.stderr)
    if description and prompt_speech is not None:
        results = list(model.inference_instruct2(text, description, prompt_speech, stream=False))
    elif prompt_speech is not None:
        results = list(model.inference_zero_shot(text, "", prompt_speech, stream=False))
    else:
        print("Error: CosyVoice 2 requires a reference audio file.", file=sys.stderr)
        sys.exit(1)

    audio = results[0]["tts_speech"].squeeze().cpu().numpy()
    sr = model.sample_rate

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
        reference_audio_path=params.get("reference_audio_path", ""),
        output_path=params["output_path"],
    )
    print(path)
