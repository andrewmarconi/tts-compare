#!/usr/bin/env python3
"""Shared utilities for all TTS model wrappers.

This module provides common functionality to reduce code duplication across
model wrappers. Each model can import this by adding the repo root to sys.path.

Compatible with Python 3.10-3.12.
"""

import json
import sys
from typing import Any

import numpy as np
import soundfile as sf


def get_device() -> str:
    """Detect the best available device for PyTorch.

    Returns:
        Device string: "cuda", "mps", or "cpu" (in order of preference)
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize float32 audio to int16 for WAV output.

    Args:
        audio: Float32 audio array (values typically in -1.0 to 1.0 range)

    Returns:
        Int16 audio array suitable for WAV file writing
    """
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def save_audio(audio: np.ndarray, sample_rate: int, output_path: str) -> None:
    """Normalize and save audio to a WAV file.

    Args:
        audio: Float32 audio array
        sample_rate: Sample rate in Hz
        output_path: Path to output WAV file
    """
    normalized = normalize_audio(audio)
    sf.write(output_path, normalized, sample_rate)


def emit_timing(load_ms: float, infer_ms: float) -> None:
    """Emit timing information in the standardized format.

    Writes a JSON line to stderr with model load and inference times.

    Args:
        load_ms: Model load time in milliseconds
        infer_ms: Inference time in milliseconds
    """
    timing = {
        "model_load_ms": round(load_ms, 1),
        "model_inference_ms": round(infer_ms, 1),
    }
    print(f"TIMING:{json.dumps(timing)}", file=sys.stderr)


def read_params() -> dict[str, Any]:
    """Read and parse JSON parameters from stdin.

    Returns:
        Dictionary containing the input parameters
    """
    return json.loads(sys.stdin.read())
