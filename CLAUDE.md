# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tts-compare (tts-compare) is a TTS model comparison framework with a Textual TUI for side-by-side evaluation of 8 open-source text-to-speech models. Each model runs in an isolated Python environment managed by `uv`.

## Commands

```bash
# Setup all model environments (creates isolated .venv per model)
./setup_models.sh

# Pre-download model weights (optional, ~15-20 GB, 10-30 min)
./download_models.sh

# Run the interactive TUI
python main.py

# Test a single model directly (example: kokoro)
cd models/kokoro && uv run python app.py <<< '{"text":"Hello","voice_preset":"af_heart","output_path":"out.wav"}'

# Sync a single model's dependencies after pyproject.toml changes
cd models/<model_name> && uv sync

# Fix corrupted model cache files (Maya1, CosyVoice)
./fix_safetensors.sh
```

## Architecture

**Main app** (`main.py`): Textual TUI with 3 screens — ParamsScreen (input text/voice config), ModelSelectScreen (pick models + per-model settings), ExecutionScreen (subprocess execution with real-time log streaming).

**Model wrapper pattern**: Each model lives in `models/<name>/` with:
- `manifest.json` — metadata (voice presets, emotion tags, capabilities like `supports_description`, `supports_reference_audio`)
- `app.py` — wrapper that reads JSON from stdin, writes WAV file, emits `TIMING:` JSON lines on stderr
- `pyproject.toml` + `.venv/` — isolated dependencies via `uv`

**Shared utilities** (`tts_common.py`): Common module at repo root providing:
- `get_device()` — PyTorch device detection (CUDA > MPS > CPU)
- `normalize_audio(audio)` — Convert float32 audio to int16
- `save_audio(audio, sr, path)` — Normalize and save WAV file
- `emit_timing(load_ms, infer_ms)` — Emit standardized TIMING JSON to stderr
- `read_params()` — Read and parse JSON from stdin

Model wrappers add the repo root to `sys.path` and import `tts_common` to avoid code duplication.

**Model discovery**: `discover_models()` in `main.py` scans `models/` for directories containing both `manifest.json` and `app.py`.

**Subprocess execution**: Models run via `uv run python app.py` in their own directory. Input is JSON on stdin (`text`, `description`, `voice_preset`, `reference_audio_path`, `output_path`). Output is the file path on stdout, timing/logs on stderr. Uses asyncio with 1 MiB buffer limit (increased from 64 KiB default) to handle long debug output lines from model loading.

**Output structure**: Each run creates a numbered folder (`output/001/`, `output/002/`, etc.) containing WAV files, `log.txt`, and `results.json` with timing data.

**Benchmarking**: Results (load time, inference time per model) are collected into each run's `results.json`.

## Adding a New Model

Create `models/<name>/` with `app.py`, `manifest.json`, and `pyproject.toml`. The app must:
- Accept JSON on stdin using `tts_common.read_params()`
- Write a WAV file using `tts_common.save_audio()` or similar
- Emit `TIMING:` lines on stderr using `tts_common.emit_timing()`
- Add repo root to `sys.path` to import `tts_common` (see existing wrappers for the pattern)

Run `uv sync` in the model directory to create its venv.

## Notable Constraints & Quirks

**Python versions**:
- Root project requires Python 3.12
- Chatterbox and CosyVoice require Python 3.10–3.11 (managed via their `pyproject.toml`)

**System dependencies**:
- Qwen3-TTS requires SoX: `brew install sox` (macOS) or `sudo apt install sox libsox-fmt-all` (Linux)

**PyTorch 2.9+ compatibility**:
- Requires `torchcodec` for audio I/O (auto-installed via `coqui-tts[codec]` for XTTS, explicitly added for Fish Speech)
- Fish Speech uses torchaudio monkey-patch for `list_audio_backends()` removed in 2.1+

**Device compatibility**:
- CosyVoice forced to CPU (MPS causes Conv1d dimension errors)
- Most models support CUDA > MPS > CPU (via `tts_common.get_device()`)
- Orpheus caps GPU memory at 60%

**Model-specific setup**:
- Fish Speech: Requires manual checkpoint download via `huggingface-cli` (see README.md § Fish Speech Manual Setup)
- CosyVoice: Uses git submodule (`models/cosyvoice/CosyVoice`)
- CosyVoice expects file path for reference audio (not pre-loaded tensor)

