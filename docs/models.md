# Model Reference

## Maya1 (3B)

Zero-shot voice design via text descriptions and the richest emotion tag support of any model in the framework.

- **Voice Description**: Yes — describe the voice you want (e.g., "Warm male voice, 30s, American accent")
- **Emotion Tags**: `<laugh>`, `<laugh_harder>`, `<sigh>`, `<chuckle>`, `<gasp>`, `<angry>`, `<excited>`, `<whisper>`, `<cry>`, `<scream>`, `<sing>`, `<snort>`, `<exhale>`, `<gulp>`, `<giggle>`, `<sarcastic>`, `<curious>`
- **Engine**: HuggingFace `transformers` + SNAC codec decoder
- **Model ID**: `maya-research/maya1`
- **Python**: 3.12
- **Known Issues**: "Separator is not found" error indicates corrupted model cache. Fix: `rm -rf ~/.cache/huggingface/hub/models--maya-research--maya1` and re-run.

## Kokoro-82M (82M)

The fastest model — lightweight at only 82M parameters with 11 built-in voice presets.

- **Voice Presets**: `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`, `am_adam`, `am_michael`, `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`
- **Naming Convention**: `{a=American,b=British}{f=female,m=male}_{name}`
- **Python**: 3.12

## Chatterbox (500M)

Voice cloning from reference audio with limited emotion support.

- **Reference Audio**: Yes — provide a WAV file to clone the voice
- **Emotion Tags**: `[laugh]`, `[cough]`, `[chuckle]`, `[sniffle]`, `[sigh]`
- **Python**: 3.10-3.11 (not compatible with 3.12)

## Orpheus-3B (3B)

Production-oriented model using vLLM backend with named voice presets and emotion support.

- **Voice Presets**: `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`
- **Emotion Tags**: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`
- **GPU Memory**: Caps utilization at 60% to avoid OOM
- **Python**: 3.12

## Qwen3-TTS (1.7B)

Instruct-based voice design with streaming-optimized latency.

- **Voice Description**: Yes
- **System Dependency**: Requires SoX (`sudo apt install sox libsox-fmt-all`)
- **Python**: 3.12

## Fish Speech 1.5 (500M)

Multilingual TTS using a dual-model architecture (LLM + DAC codec).

- **Reference Audio**: Yes
- **Checkpoint**: Requires downloading `openaudio-s1-mini` checkpoint into `models/fish_speech/checkpoints/openaudio-s1-mini/`
- **Python**: 3.12
- **Known Issues**: The `pyrootutils` package expects a `.project-root` marker file. You may need to create one: `touch models/fish_speech/.project-root`

## CosyVoice 2 (500M)

Streaming TTS architecture from Alibaba, loaded via git submodule.

- **Voice Description**: Yes (instruct mode)
- **Reference Audio**: Required — CosyVoice 2 cannot generate without a reference audio sample
- **Emotion Tags**: `[laughter]`
- **Git Submodule**: `models/cosyvoice/CosyVoice/` (clone via `git submodule update --init`)
- **Python**: 3.10-3.11 (not compatible with 3.12)
- **Known Issues**: "Separator is not found" error indicates corrupted model cache. Fix: `rm -rf ~/.cache/modelscope/hub/iic___CosyVoice2-0.5B` and re-run.

## XTTS v2 (500M)

Multilingual TTS from Coqui with voice cloning support.

- **Reference Audio**: Yes — provide a WAV file for voice cloning
- **Language**: Currently hardcoded to English in the wrapper
- **Python**: 3.12
