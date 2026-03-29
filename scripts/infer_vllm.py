#!/usr/bin/env python3
"""
CosyVoice 3 + LoRA inference with vLLM acceleration
====================================================
Uses vLLM's LLMEngine for efficient LLM inference with dynamic LoRA
adapter loading via LoraRequest.  The flow/vocoder stages still run
through CosyVoice3's native pipeline.

Requirements:
    pip install vllm>=0.9.0

Usage:
    # With dynamic LoRA adapter
    python -m scripts.infer_vllm \
        --base_model pretrained_models/CosyVoice3-0.5B \
        --lora_dir experiments/cv3/default \
        --texts "Hello world." \
        --prompt_wav prompts/wav/example.wav \
        --prompt_text "Prompt transcript."

    # Base model only (no LoRA, vLLM acceleration only)
    python -m scripts.infer_vllm \
        --base_model pretrained_models/CosyVoice3-0.5B \
        --texts "Hello world." \
        --prompt_wav prompts/wav/example.wav \
        --prompt_text "Prompt transcript."
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import warnings
from logging import getLogger, StreamHandler, INFO
from pathlib import Path
from typing import List

import numpy as np
import safetensors.torch as st
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Register custom vLLM model BEFORE any vLLM engine is created
# ---------------------------------------------------------------------------
from vllm import ModelRegistry  # noqa: E402

# Add CosyVoice submodule to path
COSYVOICE_ROOT = os.path.join(os.path.dirname(__file__), "..", "submodules", "CosyVoice")
sys.path.insert(0, COSYVOICE_ROOT)
sys.path.insert(0, os.path.join(COSYVOICE_ROOT, "third_party", "Matcha-TTS"))

from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM  # noqa: E402

ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402
from cosyvoice.cli.model import export_cosyvoice2_vllm  # noqa: E402
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer  # noqa: E402

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Audio helpers (shared with infer.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# vLLM engine setup with LoRA support
# ---------------------------------------------------------------------------

def setup_vllm_engine(cv3, base_model_dir: str, enable_lora: bool = False,
                      max_lora_rank: int = 16, max_loras: int = 4,
                      gpu_memory_utilization: float = 0.2):
    """Create vLLM LLMEngine with optional LoRA support.

    This mirrors CosyVoice3Model.load_vllm() but adds enable_lora and
    related parameters that the built-in method does not expose.
    """
    from vllm import EngineArgs, LLMEngine

    vllm_model_dir = os.path.join(base_model_dir, "vllm")

    # Export Qwen2 weights to HuggingFace format for vLLM consumption
    device = next(cv3.model.llm.llm.parameters()).device
    export_cosyvoice2_vllm(cv3.model.llm, vllm_model_dir, device)

    # Build engine args
    kwargs = dict(
        model=vllm_model_dir,
        skip_tokenizer_init=True,
        enable_prompt_embeds=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if enable_lora:
        kwargs.update(
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=max_loras,
        )

    engine_args = EngineArgs(**kwargs)
    cv3.model.llm.vllm = LLMEngine.from_engine_args(engine_args)
    cv3.model.llm.lock = threading.Lock()

    # Free GPU memory occupied by transformer layers (vLLM owns them now)
    del cv3.model.llm.llm.model.model.layers
    torch.cuda.empty_cache()

    logger.info("vLLM engine created (enable_lora=%s)", enable_lora)


# ---------------------------------------------------------------------------
# Monkey-patch inference_wrapper to inject LoraRequest
# ---------------------------------------------------------------------------

def patch_inference_wrapper(llm_module, lora_request):
    """Replace inference_wrapper on the LLM module so that every
    vllm.add_request() call includes the given LoraRequest."""
    from vllm import SamplingParams

    original_stop_token_ids = llm_module.stop_token_ids

    @torch.inference_mode()
    def inference_wrapper_with_lora(self, lm_input, sampling, min_len, max_len, uuid):
        sampling_params = SamplingParams(
            top_k=sampling,
            stop_token_ids=original_stop_token_ids,
            min_tokens=min_len,
            max_tokens=max_len,
        )
        with self.lock:
            self.vllm.add_request(
                uuid,
                {"prompt_embeds": lm_input.squeeze(0).to(torch.bfloat16).to(lm_input.device)},
                sampling_params,
                lora_request=lora_request,
            )
            self.vllm_output_queue[uuid] = queue.Queue()

        out_tokens = []
        while True:
            with self.lock:
                if self.vllm_output_queue[uuid].empty():
                    request_outputs = self.vllm.step()
                    for ro in request_outputs:
                        top_ids = list(ro.outputs[0].token_ids)[-1]
                        self.vllm_output_queue[ro.request_id].put(top_ids)
            if not self.vllm_output_queue[uuid].empty():
                top_ids = self.vllm_output_queue[uuid].get()
                if top_ids in original_stop_token_ids:
                    break
                yield top_ids
                out_tokens.append(top_ids)
                if len(out_tokens) == max_len:
                    break
            time.sleep(0.001)

        with self.lock:
            self.vllm_output_queue.pop(uuid)

    import types
    llm_module.inference_wrapper = types.MethodType(
        inference_wrapper_with_lora, llm_module
    )
    logger.info("Patched inference_wrapper with LoraRequest")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.2,
                    help="Fraction of GPU memory for vLLM KV-cache (default: 0.2)")
    ap.add_argument("--max_loras", type=int, default=4,
                    help="Max concurrent LoRA adapters in vLLM (default: 4)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("vLLM requires CUDA. Use scripts/infer.py for CPU inference.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    SAMPLE_RATE = 16000
    use_lora = args.lora_dir is not None

    # 1. Load base CosyVoice3 model (without vLLM — we set it up manually)
    logger.info("Loading CosyVoice3 from %s", args.base_model)
    cv3 = CosyVoice3(model_dir=args.base_model, fp16=False)

    # 2. If LoRA provided, apply embed_patch to local embedding table
    #    (embed_patch modifies embed_tokens which stays local, outside vLLM)
    if use_lora:
        logger.info("Applying embed_patch from %s", args.lora_dir)
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

        # Load new token embeddings into local embed_tokens table
        embed_patch_path = args.lora_dir / "embed_patch.safetensors"
        if embed_patch_path.exists():
            rows = st.load_file(str(embed_patch_path))["embed_rows"].to("cuda")
            with torch.no_grad():
                base_model.llm.model.get_input_embeddings().weight[new_ids] = rows
            logger.info("embed_patch applied for tokens %s", new_tokens)

        cv3.frontend.tokenizer = tok

    # 3. Set up vLLM engine (with LoRA support if adapter provided)
    logger.info("Setting up vLLM engine (enable_lora=%s)", use_lora)
    setup_vllm_engine(
        cv3,
        args.base_model,
        enable_lora=use_lora,
        max_lora_rank=16,
        max_loras=args.max_loras,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # 4. If LoRA provided, patch inference_wrapper to include LoraRequest
    if use_lora:
        from vllm.lora.request import LoraRequest

        lora_request = LoraRequest(
            lora_name="lora_adapter",
            lora_int_id=1,
            lora_path=str(args.lora_dir.resolve()),
        )
        patch_inference_wrapper(cv3.model.llm, lora_request)
        logger.info("LoRA adapter registered: %s", args.lora_dir)

    # 5. Prepare I/O
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

    # 6. Synthesize (same API as infer.py — vLLM is used transparently)
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
