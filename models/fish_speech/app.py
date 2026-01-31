#!/usr/bin/env python3
"""Fish Speech 1.5 — reads JSON from stdin, writes WAV to output_path.

Uses the fish_speech Python API directly (no HTTP server needed).
Requires checkpoint at checkpoints/openaudio-s1-mini/ (or set FISH_CHECKPOINT env var).
"""

import os
import sys
import time

# Add repo root to path to import shared utilities
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tts_common
import torch

CHECKPOINT_PATH = os.environ.get(
    "FISH_CHECKPOINT", "checkpoints/openaudio-s1-mini"
)
CODEC_PATH = os.path.join(CHECKPOINT_PATH, "codec.pth")
DEVICE = tts_common.get_device()
PRECISION = torch.bfloat16


def generate(text: str, reference_audio_path: str, output_path: str) -> str:
    # Check if checkpoint exists before attempting to load
    if not os.path.isdir(CHECKPOINT_PATH):
        print(f"Error: Fish Speech checkpoint not found at: {CHECKPOINT_PATH}", file=sys.stderr)
        print("Download the checkpoint and place it in the expected location, or set FISH_CHECKPOINT env var.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(CODEC_PATH):
        print(f"Error: Fish Speech codec not found at: {CODEC_PATH}", file=sys.stderr)
        sys.exit(1)

    from fish_speech.models.text2semantic.inference import (
        launch_thread_safe_queue,
    )
    from fish_speech.models.dac.inference import load_model as load_decoder
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

    t_load = time.perf_counter()
    print("Loading Fish Speech LLM...", file=sys.stderr)
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        precision=PRECISION,
        compile=False,
    )

    print("Loading Fish Speech DAC decoder...", file=sys.stderr)
    decoder_model = load_decoder(
        config_name="modded_dac_vq",
        checkpoint_path=CODEC_PATH,
        device=DEVICE,
    )

    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=PRECISION,
        compile=False,
    )
    load_ms = (time.perf_counter() - t_load) * 1000

    # Build request
    references = []
    if reference_audio_path and os.path.isfile(reference_audio_path):
        print("Loading reference audio...", file=sys.stderr)
        with open(reference_audio_path, "rb") as f:
            audio_bytes = f.read()
        references.append(ServeReferenceAudio(audio=audio_bytes, text=""))

    req = ServeTTSRequest(
        text=text,
        references=references,
        streaming=False,
        format="wav",
    )

    t_infer = time.perf_counter()
    print("Generating speech...", file=sys.stderr)

    final_audio = None
    sample_rate = None
    for result in engine.inference(req):
        if result.code == "error":
            print(f"Error during inference: {result.error}", file=sys.stderr)
            sys.exit(1)
        if result.code == "final" and result.audio is not None:
            sample_rate, final_audio = result.audio

    if final_audio is None:
        print("Error: No audio generated.", file=sys.stderr)
        sys.exit(1)

    infer_ms = (time.perf_counter() - t_infer) * 1000

    print("Saving audio...", file=sys.stderr)
    tts_common.save_audio(final_audio, sample_rate, output_path)
    tts_common.emit_timing(load_ms, infer_ms)
    print("Done.", file=sys.stderr)

    # Signal the LLM worker thread to stop
    llama_queue.put(None)

    return output_path


if __name__ == "__main__":
    params = tts_common.read_params()
    path = generate(
        text=params["text"],
        reference_audio_path=params.get("reference_audio_path", ""),
        output_path=params["output_path"],
    )
    print(path)
