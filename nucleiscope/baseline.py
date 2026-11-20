"""Classical (no deep learning) baseline: colour deconvolution + Otsu + watershed.

Kept so the deep model has something honest to beat, and because it is the
right first thing to try on a new stain/tissue before any labels exist.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import watershed

from .post import drop_small, relabel
from .stain import hematoxylin


def segment(img, min_size=30, min_dist=5):
    h = gaussian(hematoxylin(img), 1)
    mask = h > threshold_otsu(h)
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_fill_holes(mask)
    dist = ndi.distance_transform_edt(mask)
    pk = peak_local_max(dist, min_distance=min_dist, labels=ndi.label(mask)[0], exclude_border=False)
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = watershed(-dist, markers, mask=mask)
    return relabel(drop_small(lab, min_size))
