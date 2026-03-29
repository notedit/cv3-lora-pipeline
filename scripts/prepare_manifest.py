#!/usr/bin/env python3
"""
Join corpus transcripts with pre-extracted speech token paths to produce
a 4-column TSV manifest for training.

Columns: spk_id <TAB> text <TAB> token.npy <TAB> wav_path

Expected corpus layout:
    corpus_root/
        speaker_001/
            trans.txt        # lines: "utterance_id:transcript text"
            wav/
                utterance_id.wav
        speaker_002/
            ...
"""

import argparse
import csv
from logging import getLogger, StreamHandler, INFO
from pathlib import Path

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False

ap = argparse.ArgumentParser()
ap.add_argument("--corpus_root", type=Path, required=True,
                help="Root folder containing speaker directories")
ap.add_argument("--token_root", type=Path, required=True,
                help="Root folder created by extract_speech_tokens.py")
ap.add_argument("--out", type=Path, default=Path("data/manifests/all.tsv"),
                help="Output TSV manifest path")
args = ap.parse_args()

rows = []

for spk_id, spkdir in enumerate(sorted(args.corpus_root.iterdir())):
    if not spkdir.is_dir():
        continue

    trans_file = spkdir / "trans.txt"
    if not trans_file.exists():
        logger.warning("No trans.txt found in %s, skipping", spkdir)
        continue

    with trans_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line or ":" not in line:
                continue
            uid, trans = line.split(":", 1)
            uid = uid.strip()
            trans = trans.strip()

            tok = args.token_root / spkdir.name / f"{uid}.npy"
            wav = spkdir / "wav" / f"{uid}.wav"

            if not tok.exists():
                logger.warning("Token file not found: %s", tok)
                continue

            rows.append([spk_id, trans, str(tok), str(wav)])

args.out.parent.mkdir(parents=True, exist_ok=True)

with args.out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(rows)

logger.info("Manifest written: %s (%d rows)", args.out, len(rows))
