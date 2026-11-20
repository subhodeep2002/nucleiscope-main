"""Regenerate every README figure from YOUR checkpoint and results.

Run after training + evaluation (+ scripts/wsi_demo.py):
    python scripts/make_figures.py --ckpt runs/unet/best.pt --log runs/unet/log.csv \
        --results final --wsi results/wsi_demo
Writes assets/step*.png.
"""
import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nucleiscope.data import load_split, make_targets  # noqa: E402
from nucleiscope.infer import Predictor, load_model, pick_device, starts, window  # noqa: E402
from nucleiscope.post import instances  # noqa: E402
from nucleiscope.stain import hed_jitter, hematoxylin  # noqa: E402

# palette: blue = U-Net, orange = classical; text in neutral inks, never series colour
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def colorize(lab, img=None, alpha=0.5):
    """Random colour per nucleus, optionally blended over the H&E image."""
    rng = np.random.default_rng(3)
    lut = rng.integers(40, 255, (lab.max() + 1, 3)).astype(np.uint8)
    lut[0] = 0
    c = lut[lab]
    if img is None:
        return c
    out = img.copy()
    m = lab > 0
    out[m] = (out[m] * (1 - alpha) + c[m] * alpha).astype(np.uint8)
    out[find_boundaries(lab, mode="inner")] = 0
    return out


def panels(imgs, titles, path, cmap=None, w=3.2):
    fig, ax = plt.subplots(1, len(imgs), figsize=(w * len(imgs), w + 0.5))
    for a, im, t in zip(ax, imgs, titles):
        a.imshow(im, cmap=cmap if im.ndim == 2 else None, vmin=0 if im.ndim == 2 else None,
                 vmax=1 if im.ndim == 2 and im.dtype != np.uint8 else None)
        a.set_title(t)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/monuseg")
    ap.add_argument("--ckpt", default="runs/unet/best.pt")
    ap.add_argument("--log", default="runs/unet/log.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--wsi", default="results/wsi_demo")
    ap.add_argument("--fg-t", type=float, default=0.5)
    ap.add_argument("--mk-t", type=float, default=0.5)
    ap.add_argument("--out", default="assets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    o = lambda n: os.path.join(a.out, n)  # noqa: E731

    test = load_split(os.path.join(a.data, "test"))
    img, gt, _ = test[2]
    y, x, s = 300, 300, 360
    crop = lambda z: z[..., y:y + s, x:x + s] if z.ndim == 3 and z.shape[0] == 2 else z[y:y + s, x:x + s]  # noqa: E731

    # 1. data: image -> instance labels -> training targets
    t = make_targets(gt)
    panels([crop(img), colorize(crop(gt)), crop(t[0]), crop(t[1])],
           ["H&E input", "Labels (1 colour = 1 nucleus)", "Target 1: nucleus", "Target 2: boundary"],
           o("step1_data.png"), cmap="gray")

    # 2. stain augmentation + hematoxylin channel
    rng = np.random.default_rng(7)
    # training uses sigma=0.05, bias=0.05; exaggerated here so the change is visible
    jit = [hed_jitter(crop(img), rng, sigma=0.3, bias=0.2) for _ in range(3)]
    h = hematoxylin(crop(img))
    h = (h - h.min()) / (np.ptp(h) + 1e-6)
    panels([crop(img)] + jit + [h], ["Original", "Stain jitter #1*", "Stain jitter #2*", "Stain jitter #3*",
                                    "Hematoxylin channel"], o("step2_stain.png"), cmap="Blues", w=2.8)

    # 3. training curves (two panels, never a dual axis)
    if os.path.exists(a.log):
        lg = pd.read_csv(a.log)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        ax[0].plot(lg.epoch, lg.loss, color=BLUE, lw=2)
        ax[0].set(title="Training loss (BCE + Dice)", xlabel="epoch")
        ax[1].plot(lg.epoch, lg.val_aji, color=BLUE, lw=2)
        b = lg.val_aji.idxmax()
        ax[1].scatter(lg.epoch[b], lg.val_aji[b], s=60, color=BLUE, edgecolor=BG, lw=2, zorder=3)
        ax[1].text(0.97, 0.08, f"best {lg.val_aji[b]:.3f} at epoch {lg.epoch[b]}", transform=ax[1].transAxes,
                   ha="right", color=INK2)
        ax[1].set(title="Validation AJI (picks best.pt)", xlabel="epoch")
        fig.tight_layout()
        fig.savefig(o("step3_training.png"), dpi=110)
        plt.close(fig)
        print("wrote", o("step3_training.png"))

    # 4. prediction -> markers -> watershed instances
    dev = pick_device()
    p = Predictor(load_model(a.ckpt, dev), device=dev)
    pr = p(img)
    lab = instances(pr, a.fg_t, a.mk_t)
    mk = (pr[0] - pr[1]) > a.mk_t
    ov = crop(img).copy()
    ov[find_boundaries(crop(gt), mode="inner")] = (255, 255, 255)
    ov[find_boundaries(crop(lab), mode="inner")] = (0, 255, 0)
    panels([crop(pr[0]), crop(pr[1]), crop(mk).astype(np.float32), colorize(crop(lab), crop(img)), ov],
           ["U-Net: P(nucleus)", "U-Net: P(boundary)", "Markers = nucleus - boundary",
            "Watershed instances", "Pred (green) vs truth (white)"],
           o("step4_prediction.png"), cmap="magma", w=2.8)

    # 5. tiled inference: tile grid + blending weight
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 4))
    ax[0].imshow(img)
    for i, (ty, tx) in enumerate([(ty, tx) for ty in starts(1000, 512, 448) for tx in starts(1000, 512, 448)]):
        ax[0].add_patch(plt.Rectangle((tx, ty), 512, 512, fill=False, lw=1.5, ec=BLUE))
    ax[0].set_title("512-px tiles, 64-px overlap")
    ax[0].axis("off")
    im = ax[1].imshow(window(512, 64), cmap="Blues", vmin=0, vmax=1)
    ax[1].set_title("Blend weight per tile (cosine edges)")
    ax[1].axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(o("step5_tiling.png"), dpi=110)
    plt.close(fig)
    print("wrote", o("step5_tiling.png"))

    # 6. evaluation: classical vs U-Net
    cf, uf = os.path.join(a.results, "classical", "per_image.csv"), os.path.join(a.results, "unet", "per_image.csv")
    if os.path.exists(cf) and os.path.exists(uf):
        c, u = pd.read_csv(cf), pd.read_csv(uf)
        ms = ["dice", "aji", "pq", "dq", "sq"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.1, 1]})
        xs = np.arange(len(ms))
        for off, df, col, nm in [(-0.19, c, ORANGE, "Classical"), (0.19, u, BLUE, "U-Net")]:
            v = [df[m].mean() for m in ms]
            ax[0].bar(xs + off, v, 0.36, color=col, label=nm, edgecolor=BG, lw=2)
            for xi, vi in zip(xs + off, v):
                ax[0].text(xi, vi + 0.015, f"{vi:.2f}", ha="center", fontsize=8.5, color=INK2)
        ax[0].set_xticks(xs, [m.upper() for m in ms])
        ax[0].set(ylim=(0, 1), title="Mean test scores (14 unseen images)")
        ax[0].grid(axis="x", visible=False)
        ax[0].legend(frameon=False, loc="upper left", ncol=2)
        d = pd.DataFrame({"c": c.aji.values, "u": u.aji.values}).sort_values("u").reset_index(drop=True)
        for i, r in d.iterrows():
            ax[1].plot([r.c, r.u], [i, i], color=GRID, lw=2, zorder=1)
        ax[1].scatter(d.c, d.index, s=45, color=ORANGE, label="Classical", zorder=2, edgecolor=BG, lw=1.5)
        ax[1].scatter(d.u, d.index, s=45, color=BLUE, label="U-Net", zorder=3, edgecolor=BG, lw=1.5)
        ax[1].set_yticks([])
        ax[1].set(xlabel="AJI", title=f"Per-image AJI: U-Net better on {(u.aji > c.aji).sum()}/{len(u)}")
        ax[1].legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(o("step6_results.png"), dpi=110)
        plt.close(fig)
        print("wrote", o("step6_results.png"))

    # 7 + 8. morphometry and whole-slide demo
    nf = os.path.join(a.wsi, "nuclei.csv")
    if os.path.exists(nf):
        n = pd.read_csv(nf)
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
        for axi, col, lab_, rng_ in [(ax[0], "area_um2", "Area (µm²)", (0, 120)),
                                     (ax[1], "diam_um", "Equivalent diameter (µm)", (0, 14)),
                                     (ax[2], "circularity", "Circularity (1 = circle)", (0.3, 1))]:
            axi.hist(n[col].clip(*rng_), bins=40, range=rng_, color=BLUE, edgecolor=BG, lw=0.6)
            med = n[col].median()
            axi.axvline(med, color=INK, lw=1, ls="--")
            left = col == "circularity"  # skewed right, so label on the empty left side
            axi.text(0.03 if left else 0.98, 0.9, f"median {med:.2f}" if med < 2 else f"median {med:.1f}",
                     transform=axi.transAxes, ha="left" if left else "right", color=INK, fontsize=9)
            axi.set(xlabel=lab_)
            axi.grid(axis="x", visible=False)
        ax[0].set_ylabel("nuclei")
        fig.suptitle(f"Per-nucleus morphometry ({len(n):,} nuclei, whole-slide demo)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(o("step7_morphometry.png"), dpi=110)
        plt.close(fig)
        print("wrote", o("step7_morphometry.png"))

        sm = os.path.join(a.wsi, "overlay_small.jpg")
        dn = os.path.join(a.wsi, "density.npy")
        if os.path.exists(sm) and os.path.exists(dn):
            ov = plt.imread(sm)
            k = ov.shape[1] / 5000
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
            ax[0].imshow(ov)
            for by in range(0, 4000, 2048):
                for bx in range(0, 5000, 2048):
                    ax[0].add_patch(plt.Rectangle((bx * k, by * k), 2048 * k, 2048 * k, fill=False, lw=2, ec=BLUE))
            ax[0].set_title("5000×4000 mosaic, processed in 2048-px blocks")
            ax[0].axis("off")
            dd = np.load(dn)
            im = ax[1].imshow(np.ma.masked_equal(dd, 0), cmap="Blues", interpolation="nearest")
            ax[1].set_title("Nuclei per mm² of tissue (64×64 µm cells)")
            ax[1].axis("off")
            fig.colorbar(im, ax=ax[1], fraction=0.04)
            fig.tight_layout()
            fig.savefig(o("step8_wsi.png"), dpi=110)
            plt.close(fig)
            print("wrote", o("step8_wsi.png"))


if __name__ == "__main__":
    main()
