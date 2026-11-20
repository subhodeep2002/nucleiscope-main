"""U-Net with 2 output channels (foreground, boundary)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base=32, depth=4, out=2):
        super().__init__()
        ch = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([block(3, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([block(ch[i] * 2, ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], out, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        skips.pop()
        for u, d in zip(self.up, self.dec):
            x = d(torch.cat([u(x), skips.pop()], 1))
        return self.head(x)


def dice_bce(logits, t, w=(1.0, 2.0)):
    """BCE + soft Dice per channel; boundary channel weighted higher."""
    p = torch.sigmoid(logits)
    loss = 0
    for c, wc in enumerate(w):
        bce = F.binary_cross_entropy_with_logits(logits[:, c], t[:, c])
        inter = (p[:, c] * t[:, c]).sum()
        dice = 1 - (2 * inter + 1) / (p[:, c].sum() + t[:, c].sum() + 1)
        loss = loss + wc * (bce + dice)
    return loss
