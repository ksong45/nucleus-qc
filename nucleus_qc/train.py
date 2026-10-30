"""Train the five-view QC network on your own labeled masks (e.g., another tissue,
stain, or cancer type). Settings are those of the paper.

Expected folder layout (one folder per labeled tile; produced by
`nucleus-qc-export`, see README):
    <data>/<tile>/{accepted,reject,reject_border}/{rgb,tissue,raw,gray,blue}/<name>_<view>.png

    nucleus-qc-train --data labeled/ --out my_model_seed42.pth --seed 42
Train three seeds (42, 123, 456) and pass all three to `nucleus-qc --weights-dir`
(file names nucleus_qc_seed<seed>.pth) to use them as an ensemble.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .model import MEAN, STD, FiveViewNet

VIEWS = ["rgb", "tissue", "raw", "gray", "blue"]       # network order
CATEGORIES = {"accepted": 0, "reject": 1, "reject_border": 1}


def find_items(root):
    items = []
    for tile in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for cat, label in CATEGORIES.items():
            for f in sorted((tile / cat / "rgb").glob("*_rgb.png")):
                stem = f.name[:-len("_rgb.png")]
                paths = {v: tile / cat / v / f"{stem}_{v}.png" for v in VIEWS}
                if all(p.exists() for p in paths.values()):
                    items.append((paths, label))
    return items


class ViewDataset(Dataset):
    def __init__(self, items, size=224):
        self.items = items
        self.tf = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        paths, label = self.items[i]
        imgs = []
        for v in VIEWS:
            if v in ("gray", "blue"):
                a = cv2.cvtColor(cv2.imread(str(paths[v]), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
            else:
                a = cv2.cvtColor(cv2.imread(str(paths[v])), cv2.COLOR_BGR2RGB)
            imgs.append(a)
        seed = torch.randint(0, 2**31, (1,)).item()       # same augmentation for all views
        out = []
        for a in imgs:
            torch.manual_seed(seed)
            out.append(self.tf(a))
        return (*out, torch.tensor(label))


def pretrained_net():
    """FiveViewNet with ImageNet-initialized streams and the paper's frozen layers."""
    net = FiveViewNet()
    for s in net.streams:
        ref = models.efficientnet_b0(weights="IMAGENET1K_V1")
        ref.classifier = nn.Identity()
        s.load_state_dict(ref.state_dict())
        for i, (_, p) in enumerate(s.features.named_parameters()):
            if i < 40:                                     # stem and first stages frozen
                p.requires_grad = False
    return net


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def main():
    ap = argparse.ArgumentParser(description="Train the five-view nucleus QC network.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed); random.seed(args.seed)

    items = find_items(args.data)
    n_acc = sum(1 for _, y in items if y == 0)
    n_rej = len(items) - n_acc
    print(f"{len(items)} nuclei: {n_acc} accepted, {n_rej} rejected")
    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(ViewDataset(items), batch_size=32, shuffle=True,
                        num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = pretrained_net().to(device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, n_acc / n_rej]).to(device))
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    for ep in range(args.epochs):
        net.train(); tot = 0.0
        for batch in loader:
            x = [v.to(device) for v in batch[:-1]]; y = batch[-1].to(device)
            opt.zero_grad(); loss = crit(net(*x), y); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        print(f"epoch {ep + 1}/{args.epochs}  loss {tot / len(loader):.4f}")
    torch.save(net.state_dict(), args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
