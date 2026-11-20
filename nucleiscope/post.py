"""Turn (foreground, boundary) probability maps into nucleus instances."""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


def drop_small(lab, n):
    """Remove labels with fewer than n pixels (version-independent)."""
    if lab.max() == 0:
        return lab
    sz = np.bincount(lab.ravel())
    small = sz < n
    small[0] = False
    lab = lab.copy()
    lab[small[lab]] = 0
    return lab


def instances(prob, fg_t=0.5, mk_t=0.5, min_size=30):
    """Marker-controlled watershed.

    prob: (2, H, W) float array [foreground, boundary].
    Markers are nucleus cores (fg high, boundary low); watershed then grows each
    core out to the foreground mask, so touching nuclei get separate labels.
    """
    fg, bd = prob[0], prob[1]
    mask = fg > fg_t
    markers, _ = ndi.label((fg - bd) > mk_t)
    markers = drop_small(markers, max(min_size // 3, 1))
    lab = watershed(-fg, markers, mask=mask)
    # foreground blobs that got no marker still count as one nucleus each
    rest, _ = ndi.label(mask & (lab == 0))
    lab[rest > 0] = rest[rest > 0] + lab.max()
    return relabel(drop_small(lab, min_size))


def relabel(lab):
    ids = np.unique(lab)
    ids = ids[ids > 0]
    lut = np.zeros(lab.max() + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[lab]
