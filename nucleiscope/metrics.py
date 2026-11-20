"""Segmentation metrics: pixel Dice/IoU and instance-level AJI and PQ."""
import numpy as np


def dice(a, b):
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s


def iou(a, b):
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


def _overlap(gt, pr):
    """Intersection matrix [n_gt+1, n_pr+1] and per-instance areas."""
    ng, np_ = int(gt.max()), int(pr.max())
    inter = np.bincount(gt.ravel().astype(np.int64) * (np_ + 1) + pr.ravel(),
                        minlength=(ng + 1) * (np_ + 1)).reshape(ng + 1, np_ + 1)
    return inter, inter.sum(1), inter.sum(0)


def aji(gt, pr):
    """Aggregated Jaccard Index (Kumar et al., 2017) - the MoNuSeg metric."""
    inter, ag, ap = _overlap(gt, pr)
    inter, ag, ap = inter[1:, 1:], ag[1:], ap[1:]
    if len(ag) == 0 or len(ap) == 0:
        return float(len(ag) == len(ap))
    union = ag[:, None] + ap[None, :] - inter
    j = np.where(inter > 0, inter / np.maximum(union, 1), 0)
    used = np.zeros(len(ap), bool)
    C = U = 0
    for g in range(len(ag)):
        if j[g].max() == 0:
            U += ag[g]
            continue
        p = j[g].argmax()
        C += inter[g, p]
        U += union[g, p]
        used[p] = True
    U += ap[~used].sum()
    return C / U if U else 1.0


def pq(gt, pr, t=0.5):
    """Panoptic Quality = detection quality (F1) x segmentation quality (mean IoU)."""
    inter, ag, ap = _overlap(gt, pr)
    inter, ag, ap = inter[1:, 1:], ag[1:], ap[1:]
    union = ag[:, None] + ap[None, :] - inter
    j = np.where(inter > 0, inter / np.maximum(union, 1), 0)
    gi, pi = np.nonzero(j > t)  # IoU > 0.5 matches are unique by construction
    tp = len(gi)
    fp, fn = len(ap) - tp, len(ag) - tp
    if tp == 0:
        return 0.0, 0.0, 0.0
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    sq = j[gi, pi].mean()
    return dq * sq, dq, sq
