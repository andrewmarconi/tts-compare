#!/usr/bin/env python3
"""Orpheus-3B TTS — reads JSON from stdin, writes WAV to output_path."""

import io
import os
import sys
import time
import wave

# Add repo root to path to import shared utilities
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tts_common

DEFAULT_VOICE = "tara"


def generate(text: str, voice_preset: str, output_path: str) -> str:
    from orpheus_tts import OrpheusModel
    from vllm.engine.arg_utils import AsyncEngineArgs

    # Patch to lower GPU memory utilization for cards with <12GB VRAM
    _orig_setup = OrpheusModel._setup_engine
    def _patched_setup(self):
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
            gpu_memory_utilization=0.6,
        )
        return AsyncLLMEngine.from_engine_args(engine_args)
    OrpheusModel._setup_engine = _patched_setup

    t_load = time.perf_counter()
    print("Loading Orpheus-3B...", file=sys.stderr)
    model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
    load_ms = (time.perf_counter() - t_load) * 1000

    voice = voice_preset or DEFAULT_VOICE
    t_infer = time.perf_counter()
    print(f"Generating speech (voice={voice})...", file=sys.stderr)
    wav_bytes = model.generate_speech(text, voice=voice)

    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)

    infer_ms = (time.perf_counter() - t_infer) * 1000

    print("Saving audio...", file=sys.stderr)
    with open(output_path, "wb") as f:
        f.write(wav_bytes)
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
