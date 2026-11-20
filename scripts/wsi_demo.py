"""Stitch test tiles into a 5000x4000 'mini slide' and run the whole-slide pipeline.

    python scripts/wsi_demo.py --ckpt runs/unet/best.pt --out results/wsi_demo
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nucleiscope import wsi  # noqa: E402
from nucleiscope.infer import Predictor, load_model, pick_device  # noqa: E402
from nucleiscope.morph import summarize  # noqa: E402


def mosaic(test_dir, n=12, cols=4, pad=500):
    f = sorted(glob.glob(os.path.join(test_dir, "*.png")))[:n]
    t = [np.array(Image.open(x).convert("RGB")) for x in f]
    rows = [np.concatenate(t[i:i + cols], 1) for i in range(0, len(t), cols)]
    m = np.concatenate(rows, 0)
    # white border = empty glass, like the edge of a real slide
    return np.pad(m, ((pad, pad), (pad, pad), (0, 0)), constant_values=255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/monuseg/test")
    ap.add_argument("--ckpt", default="runs/unet/best.pt")
    ap.add_argument("--out", default="results/wsi_demo")
    ap.add_argument("--fg-t", type=float, default=0.5)
    ap.add_argument("--mk-t", type=float, default=0.5)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    m = mosaic(a.data)
    dev = pick_device()
    p = Predictor(load_model(a.ckpt, dev), device=dev)
    nuc, info, dens = wsi.run(wsi.ArrayReader(m, 0.25), p, log=lambda s: None, fg_t=a.fg_t, mk_t=a.mk_t)
    info.update(summarize(nuc, info["tissue_mm2"]))
    nuc.round(3).to_csv(os.path.join(a.out, "nuclei.csv"), index=False)
    np.save(os.path.join(a.out, "density.npy"), dens)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(info, f, indent=2)
    ov = m.copy()
    for x, y in zip(nuc.cx.astype(int), nuc.cy.astype(int)):
        cv2.circle(ov, (x, y), 3, (0, 200, 0), -1)
    ov = cv2.resize(ov, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(a.out, "overlay_small.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
