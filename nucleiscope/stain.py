"""H&E stain handling: Macenko normalization and HED-space augmentation.

Slides from different labs/scanners have very different colours. Normalizing to
a reference stain (Macenko et al., 2009) and jittering stain concentrations
during training are the two standard ways to make a model robust to that.
"""
import numpy as np

# Ruifrok & Johnston H&E-DAB stain vectors (rows = H, E, DAB), OD space
HED = np.array([[0.65, 0.70, 0.29],
                [0.07, 0.99, 0.11],
                [0.27, 0.57, 0.78]])
HED = HED / np.linalg.norm(HED, axis=1, keepdims=True)
HED_INV = np.linalg.inv(HED)


def to_od(img):
    return -np.log((img.astype(np.float32) + 1) / 256)


def from_od(od):
    return np.clip(256 * np.exp(-od) - 1, 0, 255).astype(np.uint8)


def hed_jitter(img, rng, sigma=0.05, bias=0.05):
    """Randomly scale/shift each stain's concentration (Tellez et al., 2018)."""
    c = to_od(img).reshape(-1, 3) @ HED_INV
    a = rng.uniform(1 - sigma, 1 + sigma, 3)
    b = rng.uniform(-bias, bias, 3)
    c = c * a + b
    return from_od(c @ HED).reshape(img.shape)


def hematoxylin(img):
    """Hematoxylin (nuclear stain) concentration map, used for morphometry."""
    c = to_od(img).reshape(-1, 3) @ HED_INV
    return c[:, 0].reshape(img.shape[:2])


class Macenko:
    """Macenko stain normalizer. fit() on a reference tile, then transform()."""

    def __init__(self, alpha=1, beta=0.15):
        self.alpha, self.beta = alpha, beta
        self.ref = None
        self.ref_max = None

    def _stains(self, img):
        od = to_od(img).reshape(-1, 3)
        od = od[(od > self.beta).all(1)]  # drop background / white pixels
        if len(od) < 100:
            return None, None
        _, v = np.linalg.eigh(np.cov(od.T))
        v = v[:, [1, 2]]  # plane of the two largest eigenvectors
        proj = od @ v
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        lo, hi = np.percentile(phi, [self.alpha, 100 - self.alpha])
        s1 = v @ [np.cos(lo), np.sin(lo)]
        s2 = v @ [np.cos(hi), np.sin(hi)]
        # hematoxylin has the larger red OD component
        st = np.array([s1, s2]) if s1[0] > s2[0] else np.array([s2, s1])
        st = st * np.sign(st.sum(1, keepdims=True))
        st /= np.linalg.norm(st, axis=1, keepdims=True)
        c = np.linalg.lstsq(st.T, to_od(img).reshape(-1, 3).T, rcond=None)[0]
        return st, c

    def fit(self, img):
        st, c = self._stains(img)
        if st is None:
            raise ValueError("reference image has too little tissue")
        self.ref = st
        self.ref_max = np.percentile(c, 99, axis=1)
        return self

    def transform(self, img):
        st, c = self._stains(img)
        if st is None:  # mostly background, nothing to normalize
            return img
        c *= (self.ref_max / np.maximum(np.percentile(c, 99, axis=1), 1e-6))[:, None]
        return from_od((self.ref.T @ c).T).reshape(img.shape)
