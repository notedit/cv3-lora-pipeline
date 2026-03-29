#!/usr/bin/env python3
"""
WAV to speech token extractor using CosyVoice speech_tokenizer_v3.onnx.
Produces a one-to-one `.npy` per WAV and writes a helper TSV mapping.

The ONNX model is distributed with CosyVoice3 release assets:
    pretrained_models/CosyVoice3-0.5B/speech_tokenizer_v3.onnx
"""

import argparse
from logging import getLogger, StreamHandler, INFO
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tqdm
import torchaudio
import whisper

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False

ap = argparse.ArgumentParser()
ap.add_argument("--wav_root", type=Path, required=True,
                help="Directory containing source WAV files")
ap.add_argument("--out_dir", type=Path, required=True,
                help="Output directory for .npy token files")
ap.add_argument("--onnx_path", type=Path, required=True,
                help="Path to speech_tokenizer_v3.onnx from CosyVoice3 release")
ap.add_argument("--max_duration", type=float, default=30.0,
                help="Skip audio files longer than this (seconds)")
args = ap.parse_args()

option = ort.SessionOptions()
option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = ["CPUExecutionProvider"]
sess = ort.InferenceSession(str(args.onnx_path), providers=providers)

args.out_dir.mkdir(parents=True, exist_ok=True)
manifest_lines = []

for wav in tqdm.tqdm(sorted(args.wav_root.rglob("*.wav"))):
    audio, sr = torchaudio.load(wav, backend="soundfile")

    if sr != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if audio.shape[1] / 16000 > args.max_duration:
        logger.warning("Skipping %s: longer than %.0fs", wav.name, args.max_duration)
        continue

    feat = whisper.log_mel_spectrogram(audio, n_mels=128)
    tokens = (
        sess.run(
            None,
            {
                sess.get_inputs()[0].name: feat.detach().cpu().numpy(),
                sess.get_inputs()[1].name: np.array(
                    [feat.shape[2]], dtype=np.int32
                ),
            },
        )[0]
        .flatten()
        .tolist()
    )
    out = args.out_dir / f"{wav.stem}.npy"
    np.save(out, tokens)
    manifest_lines.append(f"{wav}\t{out}")

tsv_path = args.out_dir / "manifest.tsv"
with open(tsv_path, "w", encoding="utf-8") as f:
    f.write("\n".join(manifest_lines))

logger.info("Done: %d files -> %s", len(manifest_lines), args.out_dir)
