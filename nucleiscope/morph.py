"""Per-nucleus morphometry in physical units (microns).

Same idea as measuring objects against a reference of known size: the scanner
records microns-per-pixel (mpp), so pixel measurements convert directly.
"""
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from .stain import hematoxylin

PROPS = ("label", "centroid", "area", "perimeter", "equivalent_diameter_area",
         "axis_major_length", "axis_minor_length", "eccentricity", "solidity",
         "intensity_mean")


def measure(lab, img, mpp=0.25):
    if lab.max() == 0:
        return pd.DataFrame(columns=["id", "cx", "cy", "area_um2", "perimeter_um", "diam_um",
                                     "major_um", "minor_um", "eccentricity", "solidity",
                                     "circularity", "h_stain"])
    t = regionprops_table(lab, intensity_image=hematoxylin(img), properties=PROPS)
    area = t["area"].astype(float)
    per = np.maximum(t["perimeter"], 1e-6)
    return pd.DataFrame({
        "id": t["label"],
        "cx": t["centroid-1"], "cy": t["centroid-0"],
        "area_um2": area * mpp**2,
        "perimeter_um": per * mpp,
        "diam_um": t["equivalent_diameter_area"] * mpp,
        "major_um": t["axis_major_length"] * mpp,
        "minor_um": t["axis_minor_length"] * mpp,
        "eccentricity": t["eccentricity"],
        "solidity": t["solidity"],
        "circularity": np.clip(4 * np.pi * area / per**2, 0, 1),
        "h_stain": t["intensity_mean"],  # mean hematoxylin = chromatin density proxy
    })


def summarize(df, tissue_mm2=None):
    if len(df) == 0:
        return {"nuclei": 0}
    a = df["area_um2"]
    s = {
        "nuclei": int(len(df)),
        "area_um2_mean": round(a.mean(), 2),
        "area_um2_median": round(a.median(), 2),
        "area_cv": round(a.std() / a.mean(), 3),  # pleomorphism: size variation
        "diam_um_mean": round(df["diam_um"].mean(), 2),
        "circularity_mean": round(df["circularity"].mean(), 3),
        "h_stain_mean": round(df["h_stain"].mean(), 3),
    }
    if tissue_mm2:
        s["tissue_mm2"] = round(tissue_mm2, 4)
        s["nuclei_per_mm2"] = round(len(df) / tissue_mm2, 1)
    return s
