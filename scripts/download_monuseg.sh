#!/usr/bin/env bash
# MoNuSeg 2018 (H&E, 40x, 1000x1000 tiles from TCGA), mirrored on Hugging Face.
# License: CC BY-NC-SA 4.0 - non-commercial use only.
set -e
mkdir -p raw
base=https://huggingface.co/datasets/RationAI/MoNuSeg/resolve/main/data
for s in train test; do
  curl -L -o raw/$s.parquet $base/$s-00000-of-00001.parquet
done
python scripts/prepare_monuseg.py --src raw --out data/monuseg
