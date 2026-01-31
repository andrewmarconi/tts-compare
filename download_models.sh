#!/usr/bin/env bash
# Pre-download script — downloads model weights for all platform-compatible models
# Run this after setup_models.sh to avoid download delays during first inference

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
echo "Press Ctrl+C once to skip a model, twice quickly to exit."
echo "Use --force to re-download models that were previously successful."
echo ""

FORCE=0
if [ "${1:-}" = "--force" ]; then
    FORCE=1
fi

mkdir -p "$OUTPUT_DIR"

# Allow Ctrl+C to skip current model (press twice quickly to exit)
SKIP_CURRENT=0
LAST_INT_TIME=0
trap 'now=$(date +%s); if [ $((now - LAST_INT_TIME)) -le 2 ]; then echo ""; echo "Exiting..."; exit 1; fi; LAST_INT_TIME=$now; SKIP_CURRENT=1; echo ""; echo "[Skipping current model... press Ctrl+C again within 2s to exit]"' INT

# Test input for inference
TEST_TEXT="Hello world"
TEST_AUDIO="$SCRIPT_DIR/audiosample.wav"

download_model() {
    local model_dir=$1
    local model_name=$2
    local display_name=$3

    # Skip if already downloaded (unless --force)
    if [ "$FORCE" -eq 0 ] && [ -f "$model_dir/.downloaded" ]; then
        echo "[$model_name] Already downloaded (use --force to re-download)"
        return 3
    fi

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

    # Run model to trigger download, showing progress from stderr
    cd "$model_dir"
    local log_file="$OUTPUT_DIR/.preload_${model_name}.log"
    SKIP_CURRENT=0

    echo "$params" | timeout 600 uv run python app.py > /dev/null 2> >(
        # Show download progress lines (from huggingface_hub, pip, etc.) while filtering noise
        while IFS= read -r line; do
            # Show lines with download progress indicators
            if echo "$line" | grep -qiE '%|download|fetch|pulling|cloning|resolving|bytes'; then
                printf "\r\033[K  [$model_name] %s" "$(echo "$line" | tail -c 120)"
            fi
            echo "$line" >> "$log_file"
        done
        printf "\r\033[K"
    ) &
    local pid=$!

    # Wait for process, checking for skip signal
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$SKIP_CURRENT" -eq 1 ]; then
            kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            echo "[$model_name] ⊘ Skipped by user"
            rm -f "$output_path"
            return 2
        fi
        sleep 0.5
    done

    wait "$pid"
    local exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        echo "[$model_name] ✓ Downloaded successfully"
        touch "$model_dir/.downloaded"
        rm -f "$output_path" "$log_file"
        return 0
    else
        echo "[$model_name] ✗ Download failed or timed out (see $log_file for details)"
        return 1
    fi
}

# Track progress
total=0
successful=0
failed=0
skipped=0
cached=0

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

    download_model "$model_dir" "$model_name" "$display_name"
    case $? in
        0) successful=$((successful + 1)) ;;
        2) skipped=$((skipped + 1)) ;;
        3) cached=$((cached + 1)) ;;
        *) failed=$((failed + 1)) ;;
    esac
done

# Clean up any remaining test files
rm -f "$OUTPUT_DIR/.preload_"*.wav

echo ""
echo "============================================"
echo "  Download Summary"
echo "============================================"
echo "  Total models: $total"
echo "  ✓ Successful: $successful"
echo "  ● Already cached: $cached"
echo "  ⊘ Skipped: $skipped"
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
