#!/usr/bin/env bash
# Fix SafeTensors corruption for Maya1 and CosyVoice by clearing cache and re-downloading

set -e

echo "============================================"
echo "  Fixing SafeTensors Corruption"
echo "============================================"
echo ""

# Function to clear and re-download a model
fix_model() {
    local cache_path=$1
    local model_id=$2
    local model_name=$3

    echo "[$model_name] Clearing cache..."
    if [ -d "$cache_path" ]; then
        rm -rf "$cache_path"
        echo "[$model_name] Cache cleared: $cache_path"
    else
        echo "[$model_name] No cache found (already cleared)"
    fi

    echo "[$model_name] Re-downloading using huggingface-cli..."
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$model_id" --resume-download
        echo "[$model_name] ✓ Download complete"
    else
        echo "[$model_name] ⚠ huggingface-cli not found. Install with:"
        echo "  pip install huggingface_hub[cli]"
        echo "[$model_name] Will download automatically on next run"
    fi
}

echo "--- Maya1 ---"
fix_model \
    "$HOME/.cache/huggingface/hub/models--maya-research--maya1" \
    "maya-research/maya1" \
    "Maya1"

echo ""
echo "--- CosyVoice 2 ---"
echo "[CosyVoice 2] Note: Uses ModelScope, not HuggingFace"
echo "[CosyVoice 2] Clearing cache..."
COSY_CACHE="$HOME/.cache/modelscope/hub/iic___CosyVoice2-0.5B"
if [ -d "$COSY_CACHE" ]; then
    rm -rf "$COSY_CACHE"
    echo "[CosyVoice 2] Cache cleared: $COSY_CACHE"
else
    echo "[CosyVoice 2] No cache found (already cleared)"
fi
echo "[CosyVoice 2] Will download automatically on next run"

echo ""
echo "============================================"
echo "  Cache Cleanup Complete"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Run 'python main.py' to re-download models"
echo "2. If errors persist, download may be timing out"
echo "3. Check your internet connection and try again"
echo ""
echo "For manual Maya1 download:"
echo "  pip install huggingface_hub[cli]"
echo "  huggingface-cli download maya-research/maya1"
echo ""
