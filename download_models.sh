#!/usr/bin/env bash
# Pre-download script — downloads model weights for all platform-compatible models
# Run this after setup_models.sh to avoid download delays during first inference

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Detect current platform
case "$(uname -s)" in
    Linux*)     CURRENT_PLATFORM="linux";;
    Darwin*)    CURRENT_PLATFORM="macos";;
    *)          CURRENT_PLATFORM="other";;
esac

echo "============================================"
echo "  Pre-downloading model weights"
echo "  Platform: $CURRENT_PLATFORM"
echo "============================================"
echo ""
echo "This will download model weights (~15-20 GB total)."
echo "Downloads happen in parallel where possible."
echo "Errors are logged but don't stop the script."
echo ""

mkdir -p "$OUTPUT_DIR"

# Test input for inference
TEST_TEXT="Hello world"
TEST_AUDIO="$SCRIPT_DIR/audiosample.wav"

download_model() {
    local model_dir=$1
    local model_name=$2
    local display_name=$3

    echo "[$model_name] Downloading..."

    # Create test output path
    local output_path="$OUTPUT_DIR/.preload_${model_name}.wav"

    # Build minimal test params
    local params=$(cat <<EOF
{
    "text": "$TEST_TEXT",
    "description": "",
    "voice_preset": "",
    "reference_audio_path": "$TEST_AUDIO",
    "output_path": "$output_path"
}
EOF
)

    # Run model to trigger download
    cd "$model_dir"
    if echo "$params" | timeout 300 uv run python app.py > /dev/null 2>&1; then
        echo "[$model_name] ✓ Downloaded successfully"
        rm -f "$output_path"  # Clean up test output
        return 0
    else
        echo "[$model_name] ✗ Download failed or timed out (may require manual setup)"
        return 1
    fi
}

# Track progress
total=0
successful=0
failed=0

for model_dir in "$MODELS_DIR"/*/; do
    model_name="$(basename "$model_dir")"

    # Skip if no manifest
    if [ ! -f "$model_dir/manifest.json" ]; then
        continue
    fi

    # Check platform compatibility
    platform_check=$(python3 -c "
import json
try:
    with open('$model_dir/manifest.json') as f:
        manifest = json.load(f)
    platforms = manifest.get('platforms', None)
    if platforms is None or '$CURRENT_PLATFORM' in platforms:
        print('compatible')
        print(manifest.get('name', '$model_name'))
    else:
        print('incompatible')
except Exception:
    print('compatible')
    print('$model_name')
")

    if echo "$platform_check" | grep -q "^incompatible"; then
        continue
    fi

    # Extract display name
    display_name=$(echo "$platform_check" | sed -n '2p')

    total=$((total + 1))

    echo ""
    echo "--- $display_name ---"

    if download_model "$model_dir" "$model_name" "$display_name"; then
        successful=$((successful + 1))
    else
        failed=$((failed + 1))
    fi
done

# Clean up any remaining test files
rm -f "$OUTPUT_DIR/.preload_"*.wav

echo ""
echo "============================================"
echo "  Download Summary"
echo "============================================"
echo "  Total models: $total"
echo "  ✓ Successful: $successful"
echo "  ✗ Failed: $failed"
echo ""

if [ $failed -eq 0 ]; then
    echo "All model weights downloaded successfully!"
    echo "Run 'python main.py' to start using the models."
else
    echo "Some models failed to download automatically."
    echo "These may require manual setup (see README.md):"
    echo "  - Fish Speech: Manual checkpoint download required"
    echo "  - CosyVoice 2: Manual git submodule setup required"
    echo "  - Maya1: May need cache clear if corrupted"
fi

echo "============================================"
