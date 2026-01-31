# tts-compare

A side-by-side comparison framework for open-source text-to-speech models. Provides an interactive terminal UI to generate speech from the same input text across multiple models and compare quality, latency, and features.

## Supported Models

| Model | Params | Voice Presets | Emotion Tags | Voice Description | Reference Audio | Platform |
|-------|--------|:---:|:---:|:---:|:---:|:---:|
| Maya1 | 3B | - | 17 | Yes | - | All |
| Kokoro-82M | 82M | 11 | - | - | - | All |
| Chatterbox | 500M | - | 5 | - | Yes | All |
| Orpheus-3B | 3B | 8 | 8 | - | - | Linux only |
| Qwen3-TTS | 1.7B | - | - | Yes | - | All |
| Fish Speech 1.5 | 500M | - | - | - | Yes | All |
| CosyVoice 2 | 500M | - | 1 | Yes | Yes | All |
| XTTS v2 | 500M | - | - | - | Yes | All |

## Quickstart

### Prerequisites

- Python 3.12 (some models require 3.10-3.11, handled automatically)
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU (recommended; MPS supported on Apple Silicon)
- SoX — required by Qwen3-TTS
  - macOS: `brew install sox`
  - Debian/Ubuntu: `sudo apt install sox libsox-fmt-all`
  - Installed automatically by `setup_models.sh`

### Setup

```bash
# Clone with submodules (CosyVoice)
git clone --recurse-submodules https://github.com/<your-repo>/tts-compare.git
cd tts-compare

# Install root dependencies (Textual TUI)
uv sync

# Set up all model environments (creates isolated .venv per model)
./setup_models.sh
```

### Run

```bash
python main.py
```

The TUI walks you through three screens:

1. **Input** — enter text, voice description, and reference audio path
2. **Model Selection** — pick which models to run and configure per-model voice presets
3. **Execution** — watch real-time logs as each model generates speech

Output WAV files are saved to `output/` and benchmark results to `output/results.json`.

### Test a Single Model

```bash
cd models/kokoro
echo '{"text": "Hello world", "voice_preset": "af_heart", "output_path": "out.wav"}' | uv run python app.py
```

## Downloading Models

Most models download their weights automatically on first run. Some require manual steps. Visit each model's page below to accept any license agreements before running.

| Model | Source | Auto-download? | Notes |
|-------|--------|:-:|-------|
| [Maya1](https://huggingface.co/maya-research/maya1) | HuggingFace | Yes | ~6 GB. Also downloads [SNAC codec](https://huggingface.co/hubertsiuzdak/snac_24khz) |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | HuggingFace (via pip) | Yes | Downloaded by the `kokoro` package on first use |
| [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) | HuggingFace | Yes | Downloaded by the `chatterbox-tts` package |
| [Orpheus-3B](https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod) | HuggingFace | Yes | ~6 GB. Uses vLLM backend. **Linux only** (vLLM not supported on macOS/Windows) |
| [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | HuggingFace | Yes | ~3.4 GB |
| [Fish Speech 1.5](https://huggingface.co/fishaudio/openaudio-s1-mini) | HuggingFace | **No** | See manual download below |
| [CosyVoice 2](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) | ModelScope | Yes | ~1 GB. Downloaded on first use |
| [XTTS v2](https://huggingface.co/coqui/XTTS-v2) | HuggingFace (via coqui) | Yes | Downloaded by the `TTS` package on first use |

### Fish Speech Manual Setup

Fish Speech requires downloading the checkpoint manually:

```bash
pip install huggingface_hub[cli]
cd models/fish_speech
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
touch .project-root   # Required by pyrootutils
```

You may need to log in first (`huggingface-cli login`) if prompted.

## Troubleshooting

### Maya1: "Separator is not found" Error

If Maya1 fails with `Separator is not found, and chunk exceed the limit`, the cached model files are corrupted. Fix by clearing the cache:

```bash
rm -rf ~/.cache/huggingface/hub/models--maya-research--maya1
```

Then run `python main.py` again to re-download the model.

### CosyVoice 2: "Separator is not found" Error

Similar to Maya1, corrupted ModelScope cache can be fixed with:

```bash
rm -rf ~/.cache/modelscope/hub/iic___CosyVoice2-0.5B
```

## Documentation

See [docs/](docs/) for detailed documentation:

- [Architecture](docs/architecture.md) — how the framework is structured
- [Models](docs/models.md) — detailed info on each model, capabilities, and known issues
- [Adding a Model](docs/adding-a-model.md) — how to add a new TTS model to the framework
