# Architecture

## Overview

tts-compare uses a subprocess-per-model architecture. The main process runs a [Textual](https://textual.textualize.io/) TUI and orchestrates model execution. Each model runs in its own isolated Python environment via `uv run`, communicating over stdin/stdout/stderr.

```
main.py (Textual TUI, Python 3.12)
  ├── discovers models/ directories
  ├── for each selected model:
  │     ├── spawns: uv run python app.py  (in model's own .venv)
  │     ├── sends JSON config on stdin
  │     ├── streams stderr to TUI in real-time
  │     ├── reads output file path from stdout
  │     └── collects TIMING: lines from stderr
  └── writes output/results.json
```

## TUI Screens

`main.py` implements three sequential screens:

- **ParamsScreen** — text input, voice description, reference audio path, output directory
- **ModelSelectScreen** — checkboxes per model, dropdowns for voice presets, displays capabilities
- **ExecutionScreen** — `RichLog` widget streaming subprocess output, summary on completion

## Model Discovery

`discover_models()` scans the `models/` directory for subdirectories containing both `manifest.json` and `app.py`. Models are sorted alphabetically by directory name.

## Subprocess Protocol

Each model's `app.py` follows the same contract:

**Input** (JSON on stdin):
```json
{
  "text": "Text to speak",
  "description": "Voice description (optional)",
  "voice_preset": "voice_name (optional)",
  "reference_audio_path": "/absolute/path/to/audio.wav (optional)",
  "output_path": "/absolute/path/to/output.wav"
}
```

**Output:**
- **stdout**: the output WAV file path (last line)
- **stderr**: informational log lines (streamed to TUI) and timing data
- **stderr timing line**: `TIMING:{"model_load_ms": 1234.5, "model_inference_ms": 567.8}`
- **exit code**: 0 on success, non-zero on failure

## Dependency Isolation

Each model has its own `pyproject.toml` and `.venv/`, managed by `uv`. This allows models to have conflicting dependencies (e.g., different Python versions, different torch versions). The root project only depends on `textual` for the TUI.

## GPU Memory Management

After each model finishes, the main process runs a cleanup subprocess that calls `gc.collect()` and `torch.cuda.empty_cache()`, followed by a brief pause, to release GPU memory before the next model starts.

## Benchmark Output

Results are written to `output/results.json`:

```json
[
  {
    "model": "Kokoro-82M",
    "model_load_ms": 4027.0,
    "model_inference_ms": 903.0,
    "settings": {
      "text": "...",
      "description": "",
      "voice_preset": "af_heart",
      "reference_audio_path": ""
    }
  }
]
```
