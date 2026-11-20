import numpy as np
import torch
from skimage.draw import disk

from nucleiscope import wsi
from nucleiscope.data import make_targets
from nucleiscope.infer import Predictor
from nucleiscope.metrics import aji, dice, pq
from nucleiscope.morph import measure
from nucleiscope.post import instances
from nucleiscope.stain import Macenko, hed_jitter


def disks(shape, centers, r):
    lab = np.zeros(shape, np.int32)
    for i, (y, x) in enumerate(centers, 1):
        lab[disk((y, x), r, shape=shape)] = i
    return lab


def paint(lab):
    """Fake H&E: purple nuclei on pink tissue."""
    img = np.empty(lab.shape + (3,), np.uint8)
    img[:] = (235, 170, 210)
    img[lab > 0] = (90, 40, 140)
    return img


def test_metrics_perfect_and_empty():
    gt = disks((64, 64), [(16, 16), (40, 40)], 6)
    assert dice(gt, gt) == 1 and aji(gt, gt) == 1 and pq(gt, gt)[0] == 1
    assert aji(gt, np.zeros_like(gt)) == 0


def test_aji_penalises_merge():
    gt = disks((64, 64), [(32, 24), (32, 40)], 7)
    merged = (gt > 0).astype(np.int32)
    assert dice(gt, merged) == 1  # pixel metric can't see the merge...
    assert aji(gt, merged) < 0.6  # ...instance metric can


def test_watershed_splits_touching_nuclei():
    gt = disks((64, 64), [(32, 24), (32, 38)], 8)  # overlapping disks
    prob = make_targets(gt)
    lab = instances(prob, min_size=10)
    assert lab.max() == 2


def test_morphometry_units():
    r, mpp = 20, 0.5
    lab = disks((100, 100), [(50, 50)], r)
    df = measure(lab, paint(lab), mpp)
    assert abs(df.area_um2[0] - np.pi * (r * mpp) ** 2) / (np.pi * (r * mpp) ** 2) < 0.02
    assert abs(df.diam_um[0] - 2 * r * mpp) < 0.3
    assert df.circularity[0] > 0.9


class OneByOne(torch.nn.Module):
    """1x1 conv: no spatial context, so tiled output must equal full output."""
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv2d(3, 2, 1)

    def forward(self, x):
        return self.c(x)


def test_tiling_matches_full_image():
    torch.manual_seed(0)
    m = OneByOne()
    img = np.random.default_rng(0).integers(0, 255, (700, 900, 3), dtype=np.uint8)
    tiled = Predictor(m, tile=256, overlap=48)(img)
    with torch.no_grad():
        x = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1)[None]
        full = torch.sigmoid(m(x))[0].numpy()
    assert tiled.shape == (2, 700, 900)
    assert np.abs(tiled - full).max() < 1e-4


class Oracle:
    """Fake predictor: thresholds the purple colour, so WSI logic is tested alone."""
    def __call__(self, img):
        fg = (img[..., 0] < 150).astype(np.float32)
        return np.stack([fg, np.zeros_like(fg)])


def test_wsi_blocks_no_duplicates_or_losses():
    rng = np.random.default_rng(1)
    pts = [(y, x) for y in range(30, 1000, 45) for x in range(30, 1000, 45)]
    pts = [(y + rng.integers(-5, 6), x + rng.integers(-5, 6)) for y, x in pts]
    lab = disks((1000, 1000), pts, 9)
    reader = wsi.ArrayReader(paint(lab), mpp=0.25)
    nuc, info, dens = wsi.run(reader, Oracle(), block=256, halo=32, log=lambda s: None)
    assert len(nuc) == len(pts)  # nuclei on block edges counted exactly once
    got = np.sort(nuc[["cy", "cx"]].round().values, axis=0)
    want = np.sort(np.array(pts, float), axis=0)
    assert np.abs(got - want).max() <= 1
    assert info["blocks"] == 16 and dens.shape == (4, 4)


def test_stain_tools():
    lab = disks((128, 128), [(64, 64)], 30)
    img = paint(lab)
    same = hed_jitter(img, np.random.default_rng(0), sigma=0, bias=0)
    assert np.abs(same.astype(int) - img).max() <= 2
    rng = np.random.default_rng(0)
    tex = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    out = Macenko().fit(tex).transform(tex)
    assert out.shape == img.shape and out.dtype == np.uint8
