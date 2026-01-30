#!/usr/bin/env bash
# Setup script — runs 'uv sync' in each model directory to create isolated venvs.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

# Detect current platform
case "$(uname -s)" in
    Linux*)     CURRENT_PLATFORM="linux";;
    Darwin*)    CURRENT_PLATFORM="macos";;
    *)          CURRENT_PLATFORM="other";;
esac

echo "============================================"
echo "  tts-compare — Setting up model environments"
echo "  Platform: $CURRENT_PLATFORM"
echo "============================================"

# System dependency: SoX is required by Qwen3-TTS (qwen-tts uses the sox binary)
if ! command -v sox &>/dev/null; then
    echo "[system] Installing SoX (required by Qwen3-TTS)..."

    if command -v brew &>/dev/null; then
        echo "  Detected Homebrew (macOS)"
        brew install sox || echo "[warn] Could not install sox — install manually: brew install sox"
    elif command -v apt-get &>/dev/null; then
        echo "  Detected apt (Debian/Ubuntu)"
        sudo apt-get install -y sox libsox-fmt-all || echo "[warn] Could not install sox — install manually: sudo apt-get install sox libsox-fmt-all"
    else
        echo "[warn] Could not detect package manager (brew or apt-get)"
        echo "       Please install SoX manually for your system:"
        echo "       - macOS: brew install sox"
        echo "       - Debian/Ubuntu: sudo apt-get install sox libsox-fmt-all"
        echo "       - Other systems: check your package manager documentation"
    fi
fi

for model_dir in "$MODELS_DIR"/*/; do
    model_name="$(basename "$model_dir")"

    if [ ! -f "$model_dir/pyproject.toml" ]; then
        echo "[skip] $model_name — no pyproject.toml"
        continue
    fi

    # Check platform compatibility from manifest.json
    if [ -f "$model_dir/manifest.json" ]; then
        platform_check=$(python3 -c "
import json
try:
    with open('$model_dir/manifest.json') as f:
        manifest = json.load(f)
    platforms = manifest.get('platforms', None)
    if platforms is None or '$CURRENT_PLATFORM' in platforms:
        print('compatible')
    else:
        print('incompatible:' + ','.join(platforms))
except Exception:
    print('compatible')  # Default to compatible if we can't parse
")

        if [[ "$platform_check" == incompatible:* ]]; then
            supported_platforms="${platform_check#incompatible:}"
            echo ""
            echo "--- $model_name ---"
            echo "  [skip] requires $supported_platforms (current: $CURRENT_PLATFORM)"
            continue
        fi
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
