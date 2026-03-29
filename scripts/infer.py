#!/usr/bin/env python3
"""
CosyVoice 3 + LoRA inference script
====================================
Load a pretrained LoRA adapter and synthesize speech.

Usage:
    python -m scripts.infer \
        --base_model pretrained_models/CosyVoice3-0.5B \
        --lora_dir experiments/cv3/default \
        --texts "Hello world." \
        --prompt_wav prompts/wav/example.wav \
        --prompt_text "Example prompt transcript."
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from logging import getLogger, StreamHandler, INFO
from pathlib import Path
from typing import List

import numpy as np
import safetensors.torch as st
import torch
import torchaudio
from peft import PeftModel

# Add CosyVoice submodule to path
COSYVOICE_ROOT = os.path.join(os.path.dirname(__file__), "..", "submodules", "CosyVoice")
sys.path.insert(0, COSYVOICE_ROOT)
sys.path.insert(0, os.path.join(COSYVOICE_ROOT, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice3
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False


def load_wav(path: Path, sr_out: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != sr_out:
        wav = torchaudio.functional.resample(wav, sr, sr_out)
    return wav


def trim_wav(wav: torch.Tensor, sr: int, trigger_level: float = 7.0) -> torch.Tensor:
    trimmed = torchaudio.functional.vad(wav, sr, trigger_level=trigger_level)
    if trimmed.shape[-1] > 0:
        trimmed_rev = torchaudio.functional.vad(
            trimmed.flip(-1), sr, trigger_level=trigger_level
        )
        trimmed = trimmed_rev.flip(-1)
    return trimmed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True,
                    help="CosyVoice3 base model directory")
    ap.add_argument("--lora_dir", type=Path, default=None,
                    help="LoRA adapter directory (output of train.py)")
    ap.add_argument("--texts", required=True,
                    help="Sentences to synthesize, separated by | or path to text file")
    ap.add_argument("--prompt_wav", type=Path, required=True,
                    help="Prompt WAV file (<4s recommended)")
    ap.add_argument("--prompt_text", required=True,
                    help="Transcription for the prompt WAV")
    ap.add_argument("--out_dir", type=Path, default="wavs_out")
    ap.add_argument("--trim_out", action="store_true",
                    help="Trim silence from synthesized speech")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU inference")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    SAMPLE_RATE = 16000

    # 1. Load base CosyVoice3 model
    logger.info("Loading CosyVoice3 from %s", args.base_model)
    cv3 = CosyVoice3(model_dir=args.base_model, fp16=False)

    # 2. Optionally attach LoRA adapter
    if args.lora_dir is not None:
        base_model = cv3.model.llm

        tok = get_qwen_tokenizer(
            token_path=f"{args.base_model}/CosyVoice-BlankEN",
            skip_special_tokens=True,
        )

        new_tokens = ["<PHON_START>", "<PHON_END>"]
        added = tok.tokenizer.add_special_tokens(
            {"additional_special_tokens": new_tokens}
        )
        logger.info("Tokens added: %d", added)

        tok.special_tokens["additional_special_tokens"].extend(
            [t for t in new_tokens
             if t not in tok.special_tokens["additional_special_tokens"]]
        )
        base_model.llm.model.resize_token_embeddings(len(tok.tokenizer))
        new_ids = tok.tokenizer.convert_tokens_to_ids(new_tokens)

        # Attach LoRA weights
        logger.info("Loading LoRA from %s", args.lora_dir)
        hf_model = PeftModel.from_pretrained(
            base_model,
            args.lora_dir,
            is_trainable=False,
            torch_dtype=torch.float32,
        )
        hf_model.to(device).eval()

        # Load new token embeddings
        embed_patch_path = args.lora_dir / "embed_patch.safetensors"
        rows = st.load_file(str(embed_patch_path))["embed_rows"].to(device)
        with torch.no_grad():
            hf_model.base_model.llm.model.get_input_embeddings().weight[new_ids] = rows

        cv3.model.llm = hf_model
        cv3.frontend.tokenizer = tok
        logger.info("LoRA adapter loaded successfully")

    # 3. Prepare I/O
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if Path(args.texts).is_file():
        sentences: List[str] = [
            ln.strip()
            for ln in Path(args.texts).read_text("utf-8").splitlines()
            if ln.strip()
        ]
    else:
        sentences = [s.strip() for s in args.texts.split("|") if s.strip()]

    prompt_speech_16k = load_wav(args.prompt_wav, SAMPLE_RATE)
    prompt_speech_16k = trim_wav(prompt_speech_16k, SAMPLE_RATE)

    if Path(args.prompt_text).is_file():
        prompt_text = Path(args.prompt_text).read_text("utf-8").strip()
    else:
        prompt_text = args.prompt_text

    # 4. Synthesize
    for idx, sentence in enumerate(sentences):
        logger.info("[%03d] Synthesizing: '%s'", idx + 1, sentence[:50])

        t0 = time.perf_counter()
        wav_iter = cv3.inference_zero_shot(
            tts_text=sentence,
            prompt_text=prompt_text,
            prompt_speech_16k=prompt_speech_16k,
        )
        wav_dict = next(wav_iter)
        dt = time.perf_counter() - t0

        wav = wav_dict["tts_speech"]
        out_path = args.out_dir / f"{idx + 1:03d}.wav"
        torchaudio.save(
            str(out_path), wav, cv3.sample_rate, format="wav", encoding="PCM_S"
        )
        logger.info("  Saved: %s (%.2fs, %.2fs audio)",
                     out_path, dt, wav.shape[-1] / cv3.sample_rate)

        if args.trim_out and wav.shape[-1] > 0:
            trimmed = trim_wav(wav, cv3.sample_rate)
            if trimmed.shape[-1] > 0:
                out_trimmed = out_path.with_name(f"{idx + 1:03d}_trimmed.wav")
                torchaudio.save(
                    str(out_trimmed), trimmed, cv3.sample_rate,
                    format="wav", encoding="PCM_S"
                )
                logger.info("  Trimmed: %s", out_trimmed)

    logger.info("All %d sentences synthesized.", len(sentences))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    main()
