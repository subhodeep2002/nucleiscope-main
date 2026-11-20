"""Convert the MoNuSeg parquet files (Hugging Face: RationAI/MoNuSeg) into
image PNGs + uint16 instance maps.

    python scripts/prepare_monuseg.py --src raw --out data/monuseg
"""
import argparse
import io
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def decode(b):
    return np.array(Image.open(io.BytesIO(b)))


def convert(src, out, split):
    os.makedirs(os.path.join(out, split), exist_ok=True)
    rows = pq.read_table(os.path.join(src, f"{split}.parquet")).to_pylist()
    for r in rows:
        img = decode(r["image"]["bytes"])[..., :3]
        inst = np.zeros(img.shape[:2], np.uint16)
        for i, m in enumerate(r["instances"], 1):
            inst[decode(m["bytes"]).astype(bool)] = i
        name = r["patient"]
        Image.fromarray(img).save(os.path.join(out, split, f"{name}.png"))
        np.save(os.path.join(out, split, f"{name}_inst.npy"), inst)
    print(f"{split}: {len(rows)} images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw")
    ap.add_argument("--out", default="data/monuseg")
    a = ap.parse_args()
    for s in ("train", "test"):
        convert(a.src, a.out, s)
