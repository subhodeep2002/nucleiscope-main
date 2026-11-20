"""Sliding-window inference: works on any image size with bounded GPU memory."""
import numpy as np
import torch

from .data import norm


def pick_device(cpu=False):
    """CUDA (NVIDIA) > MPS (Apple Silicon, e.g. M1-M4) > CPU."""
    if cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def window(tile, overlap):
    """2D blending weight: flat in the middle, cosine taper over the overlap.

    Averaging overlapping tiles with this weight removes the seams you get from
    CNN edge effects at tile borders.
    """
    r = np.ones(tile, np.float32)
    if overlap:
        t = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap + 2)[1:-1])
        r[:overlap], r[-overlap:] = t, t[::-1]
    return np.outer(r, r)


def starts(n, tile, stride):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile, stride))
    return s + [n - tile]


class Predictor:
    """img (H, W, 3) uint8 -> prob (2, H, W) float32 [foreground, boundary]."""

    def __init__(self, model, tile=512, overlap=64, batch=4, tta=False, device="cpu", normalizer=None):
        self.m = model.to(device).eval()
        self.tile, self.ov, self.bs = tile, overlap, batch
        self.tta, self.dev = tta, device
        self.normalizer = normalizer
        self.w = window(tile, overlap)

    @torch.no_grad()
    def _run(self, x):
        p = torch.sigmoid(self.m(x))
        if self.tta:  # average over flips: cheap accuracy boost
            p = p + torch.sigmoid(self.m(x.flip(-1))).flip(-1)
            p = p + torch.sigmoid(self.m(x.flip(-2))).flip(-2)
            p = p / 3
        return p.cpu().numpy()

    def __call__(self, img):
        if self.normalizer is not None:
            img = self.normalizer.transform(img)
        h, w = img.shape[:2]
        t = self.tile
        ph, pw = max(0, t - h), max(0, t - w)
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        H, W = img.shape[:2]
        acc = np.zeros((2, H, W), np.float32)
        wsum = np.zeros((H, W), np.float32)
        pos = [(y, x) for y in starts(H, t, t - self.ov) for x in starts(W, t, t - self.ov)]
        for i in range(0, len(pos), self.bs):
            chunk = pos[i:i + self.bs]
            x = torch.stack([norm(img[y:y + t, x:x + t]) for y, x in chunk]).to(self.dev)
            out = self._run(x)
            for (y, x), o in zip(chunk, out):
                acc[:, y:y + t, x:x + t] += o * self.w
                wsum[y:y + t, x:x + t] += self.w
        return (acc / np.maximum(wsum, 1e-6))[:, :h, :w]


def load_model(path, device="cpu"):
    from .model import UNet
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = UNet(**ck["cfg"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval()
