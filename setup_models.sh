#!/usr/bin/env bash
# Setup script — runs 'uv sync' in each model directory to create isolated venvs.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

echo "============================================"
echo "  tts-compare — Setting up model environments"
echo "============================================"

# System dependency: SoX is required by Qwen3-TTS (qwen-tts uses the sox binary)
if ! command -v sox &>/dev/null; then
    echo "[system] Installing SoX (required by Qwen3-TTS)..."
    sudo apt-get install -y sox libsox-fmt-all || echo "[warn] Could not install sox — install manually: sudo apt-get install sox libsox-fmt-all"
fi

for model_dir in "$MODELS_DIR"/*/; do
    model_name="$(basename "$model_dir")"

    if [ ! -f "$model_dir/pyproject.toml" ]; then
        echo "[skip] $model_name — no pyproject.toml"
        continue
    fi

    echo ""
    echo "--- $model_name ---"
    cd "$model_dir"

    # Pre-install setuptools for models with no-build-isolation packages
    if grep -q 'no-build-isolation-package' pyproject.toml 2>/dev/null; then
        echo "  (pre-installing build deps for no-build-isolation packages)"
        uv venv --quiet 2>/dev/null || true
        uv pip install setuptools "numpy<2.0" 2>&1 || true
    fi

    if uv sync 2>&1; then
        echo "  ✓ $model_name ready"
    else
        echo "  ✗ $model_name failed (see errors above)"
    fi
done

echo ""
echo "============================================"
echo "  Setup complete. Run: python main.py"
echo "============================================"
