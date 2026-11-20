"""Train the nucleus U-Net on MoNuSeg.

    python train.py --data data/monuseg --epochs 60 --base 32          # GPU
    python train.py --data data/monuseg --epochs 15 --base 16 --cpu    # quick CPU run
"""
import argparse
import contextlib
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from nucleiscope.data import NucleiCrops, load_split
from nucleiscope.infer import Predictor, pick_device
from nucleiscope.metrics import aji, dice
from nucleiscope.model import UNet, dice_bce
from nucleiscope.post import instances


def validate(model, items, dev):
    p = Predictor(model, tile=512, overlap=64, device=dev)
    d, a = [], []
    for img, gt, _ in items:
        lab = instances(p(img))
        d.append(dice(gt, lab))
        a.append(aji(gt, lab))
    model.train()
    return float(np.mean(d)), float(np.mean(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/monuseg")
    ap.add_argument("--out", default="runs/unet")
    ap.add_argument("--val", type=int, default=5, help="train images held out for validation")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--iters", type=int, default=200, help="training steps per epoch")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=2, help="use 0 if data loading errors on macOS")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = pick_device(a.cpu)
    os.makedirs(a.out, exist_ok=True)

    items = load_split(os.path.join(a.data, "train"))
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(items))
    val = [items[i] for i in idx[:a.val]]
    tr = [items[i] for i in idx[a.val:]]
    print(f"train {len(tr)} | val {len(val)} | device {dev}")

    ds = NucleiCrops(tr, size=a.size, n=a.iters * a.bs)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0)

    cfg = {"base": a.base, "depth": 4, "out": 2}
    model = UNet(**cfg).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * a.iters, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    best, log = -1, []
    for ep in range(1, a.epochs + 1):
        t0, tot = time.time(), 0.0
        for x, t in dl:
            x, t = x.to(dev), t.to(dev)
            # mixed precision on NVIDIA only; MPS/CPU train in fp32
            amp = torch.autocast("cuda") if dev == "cuda" else contextlib.nullcontext()
            with amp:
                loss = dice_bce(model(x), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        vd, va = validate(model, val, dev)
        row = {"epoch": ep, "loss": round(tot / a.iters, 4), "val_dice": round(vd, 4),
               "val_aji": round(va, 4), "sec": round(time.time() - t0, 1)}
        log.append(row)
        print(json.dumps(row), flush=True)
        ck = {"model": model.state_dict(), "cfg": cfg, "epoch": ep, "val_aji": va}
        torch.save(ck, os.path.join(a.out, "last.pt"))
        if va > best:
            best = va
            torch.save(ck, os.path.join(a.out, "best.pt"))
        with open(os.path.join(a.out, "log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writeheader()
            w.writerows(log)
    print(f"best val AJI {best:.4f}")


if __name__ == "__main__":
    main()
