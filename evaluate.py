"""Evaluate on the MoNuSeg test split (14 images, 7 organs).

    python evaluate.py --ckpt runs/unet/best.pt --tta
    python evaluate.py --baseline          # classical pipeline, no model
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

from nucleiscope import baseline
from nucleiscope.data import load_split
from nucleiscope.infer import Predictor, load_model, pick_device
from nucleiscope.metrics import aji, dice, iou, pq
from nucleiscope.post import instances


def overlay(img, lab, gt=None):
    """Prediction outlines in green, ground truth in yellow."""
    out = img.copy()
    if gt is not None:
        out[find_boundaries(gt, mode="inner")] = (255, 255, 0)
    out[find_boundaries(lab, mode="inner")] = (0, 255, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/monuseg/test")
    ap.add_argument("--ckpt", default="runs/unet/best.pt")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--overlays", type=int, default=3)
    ap.add_argument("--fg-t", type=float, default=0.5)
    ap.add_argument("--mk-t", type=float, default=0.5)
    a = ap.parse_args()

    name = "classical" if a.baseline else "unet" + ("_tta" if a.tta else "")
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    if not a.baseline:
        dev = pick_device()
        p = Predictor(load_model(a.ckpt, dev), tile=512, overlap=64, tta=a.tta, device=dev)

    rows = []
    for k, (img, gt, nm) in enumerate(load_split(a.data)):
        lab = baseline.segment(img) if a.baseline else instances(p(img), a.fg_t, a.mk_t)
        q, dq, sq = pq(gt, lab)
        rows.append({"image": nm, "dice": float(dice(gt, lab)), "iou": float(iou(gt, lab)), "aji": float(aji(gt, lab)),
                     "pq": float(q), "dq": float(dq), "sq": float(sq), "n_gt": int(gt.max()), "n_pred": int(lab.max())})
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
        if k < a.overlays:
            cv2.imwrite(os.path.join(out, f"{nm}.png"), cv2.cvtColor(overlay(img, lab, gt), cv2.COLOR_RGB2BGR))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_image.csv"), index=False)
    mean = {c: round(float(df[c].mean()), 4) for c in ["dice", "iou", "aji", "pq", "dq", "sq"]}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(mean, f, indent=2)
    print(name, mean)


if __name__ == "__main__":
    main()
