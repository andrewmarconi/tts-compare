# Adding a New Model

To add a new TTS model to the framework, create a new directory under `models/` with three files.

## 1. Create the directory

```bash
mkdir models/mymodel
cd models/mymodel
```

## 2. `manifest.json` — Model metadata

```json
{
    "name": "My Model",
    "voice_presets": ["voice1", "voice2"],
    "emotion_tags": ["<laugh>", "<sigh>"],
    "supports_description": false,
    "supports_reference_audio": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Display name shown in the TUI |
| `voice_presets` | string[] | Available voice names (empty if not supported) |
| `emotion_tags` | string[] | Tags users can embed in text (empty if not supported) |
| `supports_description` | bool | Whether the model accepts a voice description string |
| `supports_reference_audio` | bool | Whether the model can clone from a reference WAV |

## 3. `pyproject.toml` — Dependencies

```toml
[project]
name = "tts-compare-mymodel"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch",
    "soundfile",
    "mymodel-package",
]
```

## 4. `app.py` — Model wrapper

The wrapper must:

1. Read JSON from stdin
2. Load the model and generate audio
3. Write a WAV file to the specified output path
4. Print timing data to stderr
5. Print the output file path to stdout

```python
#!/usr/bin/env python3
import json
import sys
import time

import numpy as np
import soundfile as sf


def generate(text: str, output_path: str) -> str:
    t_load = time.perf_counter()
    print("Loading model...", file=sys.stderr)
    # model = load_your_model()
    load_ms = (time.perf_counter() - t_load) * 1000

    t_infer = time.perf_counter()
    print("Generating speech...", file=sys.stderr)
    # audio, sr = run_inference(model, text)
    infer_ms = (time.perf_counter() - t_infer) * 1000

    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    sf.write(output_path, audio, sr)
    print(
        f"TIMING:{json.dumps({'model_load_ms': round(load_ms, 1), 'model_inference_ms': round(infer_ms, 1)})}",
        file=sys.stderr,
    )
    print("Done.", file=sys.stderr)
    return output_path


if __name__ == "__main__":
    params = json.loads(sys.stdin.read())
    path = generate(
        text=params["text"],
        output_path=params["output_path"],
    )
    print(path)
```

## 5. Install and test

```bash
cd models/mymodel
uv sync                     # Creates isolated .venv

# Test directly
echo '{"text": "Hello", "output_path": "test.wav"}' | uv run python app.py
```

The model will automatically appear in the TUI on the next run of `python main.py`.
