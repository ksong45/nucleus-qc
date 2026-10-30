"""Nucleus-level QC of Cellpose-SAM masks: score every mask, flag likely errors,
and write a cleaned mask.

Command line:
    nucleus-qc --image tile.tif --mask tile_mask.tif --out results/
    nucleus-qc --image tile.tif --run-cellpose --out results/

Python:
    from nucleus_qc import run_qc
    table, clean_mask = run_qc(image, mask)
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from .model import TRANSFORM, load_model
from .views import iter_nuclei, load_image, model_inputs

SEEDS = (42, 123, 456)
# Mean of the four leave-one-patient-out validation (Youden) thresholds; the
# operating point used for the held-out evaluation in the paper.
DEFAULT_THRESHOLD = 0.489
RELEASE_URL = os.environ.get(
    "NUCLEUS_QC_RELEASE_URL",
    "https://github.com/ksong45/nucleus-qc/releases/download/v1.0")
WEIGHT_FILES = [f"nucleus_qc_seed{s}.pth" for s in SEEDS]


def weights_dir():
    d = Path(os.environ.get("NUCLEUS_QC_WEIGHTS", Path.home() / ".cache" / "nucleus_qc"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_weights(directory=None):
    """Paths to the three model files; downloads them from the release if missing."""
    d = Path(directory) if directory else weights_dir()
    paths = []
    for name in WEIGHT_FILES:
        p = d / name
        if not p.exists():
            print(f"Downloading {name} ...")
            torch.hub.download_url_to_file(f"{RELEASE_URL}/{name}", str(p))
        paths.append(p)
    return paths


def read_image(path):
    import imageio.v2 as imageio
    return imageio.imread(path)


def read_mask(path):
    path = str(path)
    if path.endswith(".npy"):                     # Cellpose *_seg.npy
        d = np.load(path, allow_pickle=True).item()
        m = d["masks"]
    else:
        import imageio.v2 as imageio
        m = imageio.imread(path)
    return np.asarray(m).astype(np.int32)


def run_cellpose(image_rgb, gpu):
    from cellpose import models                   # optional dependency
    net = models.CellposeModel(gpu=gpu)            # default cpsam model (Cellpose 4)
    masks = net.eval(image_rgb)[0]
    return np.asarray(masks).astype(np.int32)


@torch.no_grad()
def run_qc(image, mask, weights=None, threshold=DEFAULT_THRESHOLD, device=None,
           batch_size=64, progress=True):
    """Score every nucleus in `mask` (integer labels, 0 = background).

    Returns (rows, clean_mask): one dict per nucleus with the per-model and
    ensemble reject probabilities and the decision, and a copy of the mask with
    rejected nuclei set to 0.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image, image_rgb = load_image(np.asarray(image))
    mask = np.asarray(mask).astype(np.int32)
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image {image.shape[:2]}")
    nets = [load_model(p, device) for p in (weights or get_weights())]

    rows, batch, meta = [], [[] for _ in range(5)], []

    def flush():
        if not meta:
            return
        x = [torch.stack(v).to(device) for v in batch]
        probs = [torch.softmax(n(*x), dim=1)[:, 1].cpu().numpy() for n in nets]
        for j, (nid, cy, cx, area) in enumerate(meta):
            p = [float(q[j]) for q in probs]
            ens = float(np.mean(p))
            rows.append({"nucleus_id": nid, "centroid_y": round(cy, 1), "centroid_x": round(cx, 1),
                         "area_px": area, **{f"prob_seed{s}": round(v, 6) for s, v in zip(SEEDS, p)},
                         "prob_reject": round(ens, 6),
                         "decision": "reject" if ens >= threshold else "accept"})
        for v in batch:
            v.clear()
        meta.clear()

    for k, (nid, ys, xs) in enumerate(iter_nuclei(mask), start=1):
        for i, v in enumerate(model_inputs(image, image_rgb, mask, nid, ys, xs)):
            batch[i].append(TRANSFORM(v))
        meta.append((nid, float(ys.mean()), float(xs.mean()), int(len(ys))))
        if len(meta) == batch_size:
            flush()
            if progress:
                print(f"  scored {k} nuclei", end="\r")
    flush()
    rejected = [r["nucleus_id"] for r in rows if r["decision"] == "reject"]
    clean = mask.copy()
    clean[np.isin(clean, rejected)] = 0
    if progress:
        print(f"  scored {len(rows)} nuclei: {len(rows) - len(rejected)} accepted, "
              f"{len(rejected)} rejected (threshold {threshold})")
    return rows, clean


def save_overlay(image_rgb, mask, rows, path, max_side=2048):
    import cv2
    from skimage.segmentation import find_boundaries
    img = image_rgb[..., :3].copy() if image_rgb.ndim == 3 else np.stack([image_rgb] * 3, -1)
    rej = np.isin(mask, [r["nucleus_id"] for r in rows if r["decision"] == "reject"])
    b = find_boundaries(mask, mode="inner")
    img[b & ~rej] = [0, 200, 0]
    img[b & rej] = [255, 0, 0]
    s = min(1.0, max_side / max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser(description="Nucleus-level QC of Cellpose-SAM masks on H&E.")
    ap.add_argument("--image", required=True, help="H&E tile (RGB, 40x, ~0.23 um/pixel)")
    ap.add_argument("--mask", help="Cellpose label mask (.tif/.png or *_seg.npy)")
    ap.add_argument("--run-cellpose", action="store_true", help="segment with Cellpose-SAM first")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    ap.add_argument("--overlay", action="store_true", help="also save an overlay PNG")
    args = ap.parse_args()
    if not args.mask and not args.run_cellpose:
        ap.error("give --mask, or --run-cellpose to segment the tile first")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    image = read_image(args.image)
    if args.mask:
        mask = read_mask(args.mask)
    else:
        _, rgb = load_image(np.asarray(image))
        mask = run_cellpose(rgb[..., :3], gpu=torch.cuda.is_available())
        import tifffile
        tifffile.imwrite(out / f"{stem}_mask.tif", mask)
    weights = get_weights(args.weights_dir)
    rows, clean = run_qc(image, mask, weights, args.threshold, args.device)

    with open(out / f"{stem}_qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["nucleus_id"])
        w.writeheader()
        w.writerows(rows)
    import tifffile
    tifffile.imwrite(out / f"{stem}_mask_qc.tif", clean)
    if args.overlay:
        save_overlay(load_image(np.asarray(image))[1], mask, rows, out / f"{stem}_qc_overlay.png")
    print(f"Saved {stem}_qc.csv and {stem}_mask_qc.tif to {out}")


if __name__ == "__main__":
    main()
