#!/usr/bin/env python3
"""
LoRA fine-tuning for CosyVoice 3 LLM
=====================================
Applies PEFT LoRA to the Qwen2-based LLM inside CosyVoice 3,
trains on text + speech-token pairs, and outputs:
  1. LoRA adapter weights (HuggingFace PEFT format)
  2. embed_patch.safetensors (new token embedding rows)

TSV manifest (4 columns):
    spk_id <TAB> text <TAB> token.npy <TAB> wav_path

Only *text* and *speech-token ids* are used for training.

Usage:
    python -m scripts.train --config configs/train/default.yaml
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from logging import getLogger, StreamHandler, INFO
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model

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


class CV3Trainer(Trainer):
    """Custom Trainer that delegates loss computation to CosyVoice3LM.forward()."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_dict = model(inputs, self.args.device)
        loss = loss_dict["loss"]
        return (loss, loss_dict) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
        with torch.no_grad():
            outputs = model(inputs, self.args.device)
        loss = outputs["loss"]
        return (loss, None, None)


class TSVSpeechDataset(Dataset):
    """Reads 4-column TSV manifest and returns (text, speech_ids, wav_path)."""

    def __init__(self, tsv_path: str):
        self.rows: List[Tuple[str, torch.Tensor, str]] = []
        for ln in Path(tsv_path).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            _, txt, npy, wav = ln.split("\t")
            ids = torch.from_numpy(np.load(npy)).long()
            self.rows.append((txt, ids, wav))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_fn(batch, tokenizer):
    """Build inputs matching CosyVoice3LM.forward() batch dict format."""
    texts, speech_ids, _ = zip(*batch)

    txt_lists = [
        tokenizer.encode(t, allowed_special=tokenizer.special_tokens) for t in texts
    ]

    pad_id = tokenizer.encode(
        "<|endoftext|>", allowed_special=tokenizer.special_tokens
    )[0]
    max_len = max(len(x) for x in txt_lists)
    txt_tok = torch.full((len(txt_lists), max_len), pad_id, dtype=torch.long)

    for i, ids in enumerate(txt_lists):
        txt_tok[i, : len(ids)] = torch.tensor(ids)

    txt_len = torch.tensor([len(ids) for ids in txt_lists], dtype=torch.int32)

    sp_pad = pad_sequence(speech_ids, batch_first=True, padding_value=0)
    sp_len = torch.tensor([t.size(0) for t in speech_ids], dtype=torch.int32)

    return {
        "text_token": txt_tok,
        "text_token_len": txt_len,
        "speech_token": sp_pad,
        "speech_token_len": sp_len,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/default.yaml",
                    help="YAML configuration file")
    ap.add_argument("--resume_from_checkpoint", default=None,
                    help="Path to checkpoint dir or 'True' for auto-resume")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    seed = cfg["training"]["seed"]
    rng = torch.Generator().manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 1. Load CosyVoice3 model
    logger.info("Loading CosyVoice3 from %s", cfg["base_model"])
    cv3 = CosyVoice3(model_dir=cfg["base_model"], fp16=False)
    base_model = cv3.model.llm  # CosyVoice3LM wrapping Qwen2ForCausalLM

    # 2. Expand vocabulary with special tokens
    tok = get_qwen_tokenizer(
        token_path=f"{cfg['base_model']}/CosyVoice-BlankEN",
        skip_special_tokens=True,
    )

    new_tokens = ["<PHON_START>", "<PHON_END>"]
    added = tok.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    logger.info("Tokens added to vocabulary: %d", added)

    tok.special_tokens["additional_special_tokens"].extend(
        [t for t in new_tokens
         if t not in tok.special_tokens["additional_special_tokens"]]
    )
    base_model.llm.model.resize_token_embeddings(len(tok.tokenizer))

    # 3. Apply LoRA to LLM attention layers
    lora_cfg = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
    )
    llm = get_peft_model(base_model, lora_cfg)

    # 4. Set up gradient masking for embeddings
    new_ids = tok.tokenizer.convert_tokens_to_ids(new_tokens)
    emb = llm.base_model.llm.model.get_input_embeddings()
    emb.weight.requires_grad_(True)

    existing_mask = torch.ones(
        emb.num_embeddings, dtype=torch.bool, device=emb.weight.device
    )
    existing_mask[new_ids] = False  # False = keep gradient for new tokens

    def mask_grad(grad):
        grad[existing_mask] = 0
        return grad

    emb.weight.register_hook(mask_grad)

    # 5. Wire LoRA model back into CosyVoice3
    cv3.model.llm = llm
    cv3.frontend.tokenizer = tok

    # 6. Dataset split
    ds_full = TSVSpeechDataset(cfg["manifest"])
    n = len(ds_full)
    n_val = int(n * cfg["val_ratio"])
    n_train = n - n_val
    train_ds, val_ds = random_split(ds_full, [n_train, n_val], generator=rng)
    logger.info("Dataset: %d train, %d val", n_train, n_val)

    # 7. Train
    trainer = CV3Trainer(
        model=llm,
        args=TrainingArguments(**cfg["training"]),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: collate_fn(batch, tok),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 8. Save LoRA weights
    output_dir = cfg["training"]["output_dir"]
    llm.save_pretrained(output_dir)
    logger.info("LoRA adapter saved to %s", output_dir)

    # 9. Save new token embeddings separately
    weight = emb.weight.detach().cpu()
    embed_rows = weight[new_ids]
    save_file(
        {"embed_rows": embed_rows},
        str(Path(output_dir) / "embed_patch.safetensors"),
    )
    logger.info("Embedding patch saved to %s/embed_patch.safetensors", output_dir)


if __name__ == "__main__":
    main()
