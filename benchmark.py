#!/usr/bin/env python3
"""Run a single TTS model N times with default settings and collect results.

Usage:
    python benchmark.py <model_dir_name> <run_count>

Example:
    python benchmark.py kokoro 5
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_DIR = REPO_ROOT / "output"

DEFAULT_TEXT = (
    "Wait\u2014would you believe what just happened? The caf\u00e9 served amazing "
    "cr\u00e8me br\u00fbl\u00e9e and fresh croissants while a gentle breeze carried the "
    "scent of spring flowers across the garden, and birds sang their morning "
    "songs with pure joy!"
)
DEFAULT_DESCRIPTION = "warm male voice, mid-50s"
DEFAULT_REFERENCE_AUDIO = str(REPO_ROOT / "audiosample.wav")


def next_run_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [int(d.name) for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    return OUTPUT_DIR / f"{max(existing, default=0) + 1:03d}"


def load_manifest(model_dir: Path) -> dict:
    with open(model_dir / "manifest.json") as f:
        manifest = json.load(f)
    manifest["dir_name"] = model_dir.name
    return manifest


async def run_model(model_dir: Path, params: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    proc = await asyncio.create_subprocess_exec(
        "uv", "run", "python", "-u", "app.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(model_dir),
        env=env,
        limit=1024 * 1024,
    )
    proc.stdin.write(json.dumps(params).encode())
    proc.stdin.close()

    timing = {}
    async for raw_line in proc.stderr:
        line = raw_line.decode().rstrip()
        if not line:
            continue
        if line.startswith("TIMING:"):
            timing = json.loads(line[7:])
        else:
            print(f"  {line}", file=sys.stderr)

    stdout = await proc.stdout.read()
    await proc.wait()
    return {"returncode": proc.returncode, "stdout": stdout.decode(), "timing": timing}


async def gpu_cleanup(model_dir: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    proc = await asyncio.create_subprocess_exec(
        "uv", "run", "python", "-c",
        "import gc; gc.collect(); import torch;"
        " torch.cuda.empty_cache() if torch.cuda.is_available() else None;"
        " torch.mps.empty_cache() if hasattr(torch, 'mps') and torch.backends.mps.is_available() else None",
        cwd=str(model_dir),
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <model> <runcount>", file=sys.stderr)
        sys.exit(1)

    model_name = sys.argv[1]
    run_count = int(sys.argv[2])

    model_dir = MODELS_DIR / model_name
    if not (model_dir / "app.py").exists():
        print(f"Error: model '{model_name}' not found in {MODELS_DIR}", file=sys.stderr)
        available = [d.name for d in sorted(MODELS_DIR.iterdir())
                     if d.is_dir() and (d / "app.py").exists()]
        print(f"Available: {', '.join(available)}", file=sys.stderr)
        sys.exit(1)

    manifest = load_manifest(model_dir)
    display_name = manifest["name"]
    presets = manifest.get("voice_presets", [])
    voice_preset = presets[0] if presets else ""

    ref_path = DEFAULT_REFERENCE_AUDIO if os.path.isfile(DEFAULT_REFERENCE_AUDIO) else ""

    print(f"Benchmarking {display_name} x{run_count}")
    print(f"  Voice preset: {voice_preset or '(none)'}")
    print(f"  Description: {DEFAULT_DESCRIPTION}")
    print(f"  Reference audio: {ref_path or '(none)'}")
    print()

    for i in range(1, run_count + 1):
        output_dir = next_run_dir()
        output_dir.mkdir(parents=True)
        output_path = str(output_dir / f"{model_name}.wav")

        params = {
            "text": DEFAULT_TEXT,
            "description": DEFAULT_DESCRIPTION,
            "voice_preset": voice_preset,
            "reference_audio_path": ref_path,
            "output_path": output_path,
        }

        print(f"[{i}/{run_count}] Run → {output_dir.name}")
        result = asyncio.run(run_model(model_dir, params))

        timing = result.get("timing", {})
        if result["returncode"] == 0:
            out_lines = result["stdout"].strip().splitlines()
            out_path = out_lines[-1] if out_lines else ""
            load_s = timing.get("model_load_ms", 0) / 1000
            infer_s = timing.get("model_inference_ms", 0) / 1000
            print(f"  ✓ Load: {load_s:.1f}s | Inference: {infer_s:.1f}s")
        else:
            print(f"  ✗ Failed (exit {result['returncode']})")

        # Write results.json for this run
        if timing:
            benchmarks = [{
                "model": display_name,
                "model_load_ms": timing.get("model_load_ms"),
                "model_inference_ms": timing.get("model_inference_ms"),
                "settings": {
                    "text": params["text"],
                    "description": params["description"],
                    "voice_preset": params["voice_preset"],
                    "reference_audio_path": params["reference_audio_path"],
                },
            }]
            with open(output_dir / "results.json", "w") as f:
                json.dump(benchmarks, f, indent=2)

        # GPU cleanup between runs
        asyncio.run(gpu_cleanup(model_dir))
        time.sleep(2)

    print(f"\nDone. {run_count} runs completed.")


if __name__ == "__main__":
    main()
