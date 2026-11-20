"""Dataset: random crops of H&E tiles with nucleus + boundary targets."""
import glob
import os

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
from torch.utils.data import Dataset

from .stain import hed_jitter


def load_split(root):
    """Return list of (image uint8 HxWx3, instance map uint16 HxW, name)."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, "*_inst.npy"))):
        name = os.path.basename(f)[:-9]
        img = np.array(Image.open(os.path.join(root, name + ".png")).convert("RGB"))
        out.append((img, np.load(f), name))
    return out


def make_targets(inst):
    """2-channel target: [nucleus foreground, nucleus boundary].

    Predicting the boundary separately is what lets us split touching nuclei
    later (marker = foreground - boundary, then watershed).
    """
    fg = inst > 0
    bd = find_boundaries(inst, mode="inner")
    bd = ndi.binary_dilation(bd) & fg
    return np.stack([fg, bd]).astype(np.float32)


def norm(img):
    """uint8 HWC -> float CHW in [-1, 1]."""
    return torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1)


class NucleiCrops(Dataset):
    def __init__(self, items, size=256, n=1000, aug=True):
        self.imgs = [x[0] for x in items]
        self.tgts = [make_targets(x[1]) for x in items]
        self.size, self.n, self.aug = size, n, aug

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # seeded from torch so runs are reproducible under torch.manual_seed
        rng = np.random.default_rng(int(torch.randint(0, 2**31, (1,))))
        k = rng.integers(len(self.imgs))
        img, t = self.imgs[k], self.tgts[k]
        h, w = img.shape[:2]
        y = rng.integers(0, h - self.size + 1)
        x = rng.integers(0, w - self.size + 1)
        img = img[y:y + self.size, x:x + self.size]
        t = t[:, y:y + self.size, x:x + self.size]
        if self.aug:
            r = rng.integers(4)
            img, t = np.rot90(img, r, (0, 1)), np.rot90(t, r, (1, 2))
            if rng.random() < 0.5:
                img, t = img[:, ::-1], t[:, :, ::-1]
            img = hed_jitter(np.ascontiguousarray(img), rng)
            # brightness / contrast
            img = np.clip(img * rng.uniform(0.9, 1.1) + rng.uniform(-15, 15), 0, 255).astype(np.uint8)
        return norm(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(t))
