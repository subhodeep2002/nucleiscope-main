"""Whole-slide processing for gigapixel images (e.g. 50K x 50K).

A slide that size is ~7.5 GB as raw RGB, so it never goes into memory at once.
We walk the slide in blocks, each read with a small halo of context:

    +---------------- halo ----------------+
    |   +-------- core block ---------+    |
    |   |  keep nuclei whose centroid  |   |
    |   |  falls inside the core       |   |
    |   +------------------------------+   |
    +--------------------------------------+

Nuclei cut by a block edge are fully visible in the neighbouring block's halo,
and the centroid rule assigns every nucleus to exactly one block - no
duplicates, no half-nuclei. Background blocks are skipped via a tissue mask.
"""
import time

import numpy as np
import pandas as pd

from .morph import measure
from .post import instances


class ArrayReader:
    """Reader over an in-memory or np.memmap RGB array."""

    def __init__(self, arr, mpp=0.25):
        self.a, self.mpp = arr, mpp
        self.h, self.w = arr.shape[:2]

    def read(self, x, y, w, h):
        out = np.full((h, w, 3), 255, np.uint8)  # outside slide = white glass
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, self.w), min(y + h, self.h)
        if x1 > x0 and y1 > y0:
            out[y0 - y:y1 - y, x0 - x:x1 - x] = self.a[y0:y1, x0:x1, :3]
        return out


class SlideReader(ArrayReader):
    """OpenSlide reader for .svs / .ndpi / .tiff (level 0)."""

    def __init__(self, path, mpp=None):
        import openslide
        self.s = openslide.OpenSlide(path)
        self.w, self.h = self.s.dimensions
        self.mpp = mpp or float(self.s.properties.get("openslide.mpp-x", 0.25))

    def read(self, x, y, w, h):
        return np.array(self.s.read_region((x, y), 0, (w, h)).convert("RGB"))


def tissue_mask(img, sat=20):
    """Tissue = saturated pixels (white glass and background are grey/white)."""
    mx, mn = img.max(2).astype(int), img.min(2).astype(int)
    return (mx - mn) > sat


def run(reader, predictor, block=2048, halo=64, min_tissue=0.02, cell=256, log=print, **post_kw):
    """Segment + measure every nucleus on the slide.

    Returns (nuclei DataFrame in slide coordinates, info dict, density grid).
    The density grid has one value per `cell` x `cell` pixels (nuclei/mm^2),
    i.e. a low-res heatmap of cellularity across the slide.
    """
    mpp = reader.mpp
    gh, gw = -(-reader.h // cell), -(-reader.w // cell)
    counts = np.zeros((gh, gw), np.int32)
    tissue_px = np.zeros((gh, gw), np.int64)
    rows, n_blocks, n_skip, t0 = [], 0, 0, time.time()
    for y in range(0, reader.h, block):
        for x in range(0, reader.w, block):
            bw, bh = min(block, reader.w - x), min(block, reader.h - y)
            img = reader.read(x - halo, y - halo, bw + 2 * halo, bh + 2 * halo)
            tm = tissue_mask(img)
            core = tm[halo:halo + bh, halo:halo + bw]
            n_blocks += 1
            if core.mean() < min_tissue:
                n_skip += 1
                continue
            # tissue area per heatmap cell
            for cy in range(0, bh, cell):
                for cx in range(0, bw, cell):
                    tissue_px[(y + cy) // cell, (x + cx) // cell] += core[cy:cy + cell, cx:cx + cell].sum()
            lab = instances(predictor(img), **post_kw)
            df = measure(lab, img, mpp)
            if len(df) == 0:
                continue
            keep = (df.cx >= halo) & (df.cx < halo + bw) & (df.cy >= halo) & (df.cy < halo + bh)
            df = df[keep].copy()
            df["cx"] += x - halo
            df["cy"] += y - halo
            np.add.at(counts, ((df.cy // cell).astype(int), (df.cx // cell).astype(int)), 1)
            rows.append(df)
            log(f"block ({x},{y}) nuclei={len(df)}")
    nuc = pd.concat(rows, ignore_index=True) if rows else measure(np.zeros((1, 1), int), None, mpp)
    nuc["id"] = np.arange(1, len(nuc) + 1)
    dt = time.time() - t0
    tissue_mm2 = tissue_px * (mpp / 1000) ** 2
    # cells with little tissue give unstable densities (tiny denominator) - leave them blank
    ok = tissue_px >= 0.25 * cell * cell
    density = np.where(ok, counts / np.maximum(tissue_mm2, 1e-12), 0)
    info = {
        "width": reader.w, "height": reader.h, "mpp": mpp,
        "blocks": n_blocks, "blocks_skipped": n_skip,
        "tissue_mm2": float(tissue_mm2.sum()),
        "seconds": round(dt, 1),
        "megapixels_per_s": round(reader.w * reader.h / 1e6 / max(dt, 1e-9), 2),
    }
    return nuc, info, density
