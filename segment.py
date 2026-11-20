"""Segment and measure every nucleus in an image or whole slide.

    python segment.py tile.png --mpp 0.25                 # any RGB image
    python segment.py slide.svs                           # WSI (needs openslide)
    python segment.py big.npy --mpp 0.25                  # huge RGB array (memmapped)

Writes to --out: nuclei.csv (one row per nucleus, slide coordinates, microns),
summary.json, and overlay.png (small images) or density.png (cellularity
heatmap, large images).
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

from nucleiscope import wsi
from nucleiscope.infer import Predictor, load_model, pick_device
from nucleiscope.morph import summarize

Image.MAX_IMAGE_PIXELS = None


def open_reader(path, mpp):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".svs", ".ndpi", ".mrxs", ".scn") or (ext in (".tif", ".tiff") and mpp is None):
        return wsi.SlideReader(path, mpp)
    if ext == ".npy":
        return wsi.ArrayReader(np.load(path, mmap_mode="r"), mpp or 0.25)
    return wsi.ArrayReader(np.array(Image.open(path).convert("RGB")), mpp or 0.25)


def heatmap(d):
    v = d / max(np.percentile(d[d > 0], 99), 1e-6) if (d > 0).any() else d
    return cv2.applyColorMap((np.clip(v, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ckpt", default="runs/unet/best.pt")
    ap.add_argument("--mpp", type=float, default=None, help="microns per pixel (read from slide if omitted)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--fg-t", type=float, default=0.5)
    ap.add_argument("--mk-t", type=float, default=0.5)
    ap.add_argument("--device", default=pick_device(), help="cuda / mps / cpu")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    r = open_reader(a.path, a.mpp)
    p = Predictor(load_model(a.ckpt, a.device), tile=512, overlap=64, tta=a.tta, device=a.device)
    nuc, info, dens = wsi.run(r, p, block=a.block, log=lambda s: None, fg_t=a.fg_t, mk_t=a.mk_t)
    info.update(summarize(nuc, info["tissue_mm2"]))
    nuc.round(3).to_csv(os.path.join(a.out, "nuclei.csv"), index=False)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))

    if r.w * r.h <= 25_000_000:  # full-res overlay only for small images
        img = r.read(0, 0, r.w, r.h)
        ov = img.copy()
        for x, y in zip(nuc.cx.astype(int), nuc.cy.astype(int)):
            cv2.circle(ov, (x, y), 3, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(a.out, "overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    hm = heatmap(dens)
    k = max(1, 600 // max(hm.shape[:2]))
    cv2.imwrite(os.path.join(a.out, "density.png"), cv2.resize(hm, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    main()
