# cv3-lora-pipeline

CosyVoice 3 LLM LoRA fine-tuning pipeline. Extracts and trains LoRA adapters on the Qwen2-based LLM component of CosyVoice 3, producing lightweight adapter weights (<10MB) that can be loaded on top of any CosyVoice 3 base model.

Reference: [UtterTune](https://github.com/shuheikatoinfo/UtterTune)

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/notedit/cv3-lora-pipeline.git
cd cv3-lora-pipeline

# Or initialize submodules after cloning
git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt

# Install CosyVoice dependencies
cd submodules/CosyVoice
pip install -r requirements.txt
cd ../..
```

Download a CosyVoice 3 pretrained model (e.g. `Fun-CosyVoice3-0.5B`) and place it under `pretrained_models/`.

## Pipeline

### 1. Extract Speech Tokens

Convert WAV files into discrete speech tokens using the CosyVoice 3 ONNX tokenizer:

```bash
python -m scripts.extract_speech_tokens \
    --wav_root data/corpus/speaker_001/wav \
    --out_dir data/speech_tokens/speaker_001 \
    --onnx_path pretrained_models/CosyVoice3-0.5B/speech_tokenizer_v3.onnx
```

### 2. Prepare Manifest

Build a TSV manifest joining transcripts with pre-extracted speech tokens:

```bash
python -m scripts.prepare_manifest \
    --corpus_root data/corpus \
    --token_root data/speech_tokens \
    --out data/manifests/all.tsv
```

The manifest format is: `spk_id<TAB>text<TAB>token.npy<TAB>wav_path`

### 3. Train LoRA

Fine-tune the LLM with LoRA:

```bash
python -m scripts.train --config configs/train/default.yaml
```

Key training features:
- PEFT LoRA applied to LLM attention layers (q/k/v/o_proj)
- Gradient masking ensures only new token embeddings are updated (existing embeddings frozen)
- HuggingFace Trainer with cosine LR schedule

### 4. Inference

Synthesize speech with the trained LoRA adapter:

```bash
python -m scripts.infer \
    --base_model pretrained_models/CosyVoice3-0.5B \
    --lora_dir experiments/cv3/default \
    --texts "Text to synthesize." \
    --prompt_wav prompts/wav/example.wav \
    --prompt_text "Prompt transcript."
```

## Output Format

Training produces two files in the output directory:

| File | Description |
|------|-------------|
| `adapter_model.safetensors` + `adapter_config.json` | LoRA adapter weights (PEFT format) |
| `embed_patch.safetensors` | New token embedding rows |

Both files are required for inference. Load them with:
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, lora_dir)
```

## Configuration

See `configs/train/default.yaml` for all training parameters:

- **LoRA**: rank=16, alpha=64, dropout=0.05, targets=q/k/v/o_proj
- **Training**: lr=1e-4, batch_size=8, max_steps=20000, cosine scheduler, fp16
- **Data**: 5% validation split, 30s max audio duration

## Architecture

```
CosyVoice3
  +-- CosyVoice3Model
  |     +-- CosyVoice3LM (extends Qwen2LM)
  |     |     +-- llm: Qwen2ForCausalLM  <-- LoRA applied here
  |     |     +-- speech_embedding
  |     |     +-- llm_decoder
  |     +-- flow (DiT)
  |     +-- hift (vocoder)
  +-- CosyVoiceFrontEnd
```

Only the LLM attention layers receive LoRA adapters. The flow model, vocoder, and speech tokenizer remain frozen.
