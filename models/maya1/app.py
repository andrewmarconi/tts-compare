#!/usr/bin/env python3
"""Maya1 TTS — reads JSON from stdin, writes WAV to output_path."""

import os
import sys
import time

# Add repo root to path to import shared utilities
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tts_common
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

from codec import (
    SAMPLE_RATE,
    CODE_END_TOKEN_ID,
    build_prompt,
    extract_snac_codes,
    unpack_snac_from_7,
)

MODEL_ID = "maya-research/maya1"


def generate(text: str, description: str, output_path: str) -> str:
    t_load = time.perf_counter()
    print("Loading Maya1 model...", file=sys.stderr)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as e:
        if "Separator is not found" in str(e) or "chunk exceed the limit" in str(e):
            print("Error: Model files appear corrupted. Try re-downloading:", file=sys.stderr)
            print(f"  rm -rf ~/.cache/huggingface/hub/models--maya-research--maya1", file=sys.stderr)
            print("Then run again to re-download.", file=sys.stderr)
        raise

    device = tts_common.get_device()

    print("Loading SNAC decoder...", file=sys.stderr)
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    load_ms = (time.perf_counter() - t_load) * 1000

    prompt = build_prompt(tokenizer, description or "", text)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t_infer = time.perf_counter()
    print("Generating speech...", file=sys.stderr)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            min_new_tokens=28,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=CODE_END_TOKEN_ID,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0, inputs["input_ids"].shape[1]:].tolist()
    snac_tokens = extract_snac_codes(generated_ids)

    if len(snac_tokens) < 7:
        print("Error: not enough audio tokens generated.", file=sys.stderr)
        sys.exit(1)

    print("Decoding audio with SNAC...", file=sys.stderr)
    levels = unpack_snac_from_7(snac_tokens)
    codes_tensor = [
        torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
        for level in levels
    ]

    with torch.inference_mode():
        z_q = snac_model.quantizer.from_codes(codes_tensor)
        audio = snac_model.decoder(z_q)[0, 0].cpu().numpy()

    if len(audio) > 2048:
        audio = audio[2048:]

    infer_ms = (time.perf_counter() - t_infer) * 1000

    print("Saving audio...", file=sys.stderr)
    tts_common.save_audio(audio, SAMPLE_RATE, output_path)
    tts_common.emit_timing(load_ms, infer_ms)
    print("Done.", file=sys.stderr)
    return output_path


if __name__ == "__main__":
    params = tts_common.read_params()
    path = generate(
        text=params["text"],
        description=params.get("description", ""),
        output_path=params["output_path"],
    )
    print(path)
