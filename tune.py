"""Grid-search post-processing thresholds on the VALIDATION images (never test).

    python tune.py --ckpt runs/unet/best.pt
Then pass the printed values to evaluate.py / segment.py via --fg-t/--mk-t.
"""
import argparse
import itertools
import os

import numpy as np

from nucleiscope.data import load_split
from nucleiscope.infer import Predictor, load_model, pick_device
from nucleiscope.metrics import aji
from nucleiscope.post import instances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/monuseg")
    ap.add_argument("--ckpt", default="runs/unet/best.pt")
    ap.add_argument("--val", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()

    items = load_split(os.path.join(a.data, "train"))
    idx = np.random.default_rng(a.seed).permutation(len(items))[:a.val]  # same split as train.py
    dev = pick_device()
    p = Predictor(load_model(a.ckpt, dev), tta=a.tta, device=dev)
    probs = [(p(items[i][0]), items[i][1]) for i in idx]

    res = []
    for fg_t, mk_t in itertools.product([0.3, 0.4, 0.5], [0.5, 0.6, 0.7, 0.8, 0.9]):
        s = np.mean([aji(gt, instances(pr, fg_t, mk_t)) for pr, gt in probs])
        res.append((s, fg_t, mk_t))
        print(f"fg_t={fg_t} mk_t={mk_t} val AJI={s:.4f}", flush=True)
    s, fg_t, mk_t = max(res)
    print(f"best: --fg-t {fg_t} --mk-t {mk_t} (val AJI {s:.4f})")


if __name__ == "__main__":
    main()
