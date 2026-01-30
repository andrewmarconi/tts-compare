# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tts-compare (tts-compare) is a TTS model comparison framework with a Textual TUI for side-by-side evaluation of 8 open-source text-to-speech models. Each model runs in an isolated Python environment managed by `uv`.

## Commands

```bash
# Setup all model environments (creates isolated .venv per model)
./setup_models.sh

# Run the interactive TUI
python main.py

# Test a single model directly (example: kokoro)
cd models/kokoro && uv run python app.py <<< '{"text":"Hello","voice_preset":"af_heart","output_path":"out.wav"}'
```

## Architecture

**Main app** (`main.py`): Textual TUI with 3 screens — ParamsScreen (input text/voice config), ModelSelectScreen (pick models + per-model settings), ExecutionScreen (subprocess execution with real-time log streaming).

**Model wrapper pattern**: Each model lives in `models/<name>/` with:
- `manifest.json` — metadata (voice presets, emotion tags, capabilities like `supports_description`, `supports_reference_audio`)
- `app.py` — wrapper that reads JSON from stdin, writes WAV file, emits `TIMING:` JSON lines on stderr
- `pyproject.toml` + `.venv/` — isolated dependencies via `uv`

**Model discovery**: `discover_models()` in `main.py` scans `models/` for directories containing both `manifest.json` and `app.py`.

**Subprocess execution**: Models run via `uv run python app.py` in their own directory. Input is JSON on stdin (`text`, `description`, `voice_preset`, `reference_audio_path`, `output_path`). Output is the file path on stdout, timing/logs on stderr.

**Benchmarking**: Results (load time, inference time per model) are collected into `output/results.json`.

## Adding a New Model

Create `models/<name>/` with `app.py`, `manifest.json`, and `pyproject.toml`. The app must accept JSON on stdin and write a WAV file. Emit `TIMING:` lines on stderr for benchmarking. Run `uv sync` in the model directory to create its venv.

## Notable Constraints

- Root project requires Python 3.12; Chatterbox and CosyVoice require Python 3.10–3.11
- Qwen3-TTS requires SoX installed on the system
- Fish Speech and CosyVoice have stub implementations (incomplete)
- CosyVoice uses a git submodule (`models/cosyvoice/CosyVoice`)
- GPU recommended; Orpheus caps GPU memory at 60%
