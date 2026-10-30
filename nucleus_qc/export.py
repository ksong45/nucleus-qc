"""Write the five views of labeled nuclei in the training layout, so that
`nucleus-qc-train` can fine-tune or retrain the network on new data.

labels.csv: columns nucleus_id,label with label in {accept, reject, reject_border}.

    nucleus-qc-export --image tile.tif --mask tile_mask.tif --labels labels.csv --out labeled/
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from .predict import read_image, read_mask
from .views import context_views, iter_nuclei, load_image, nucleus_views, to_uint8

FOLDER = {"accept": "accepted", "reject": "reject", "reject_border": "reject_border"}


def export(image, mask, labels, out_dir, tile_name):
    """labels: {nucleus_id: 'accept'|'reject'|'reject_border'}. Files are written
    exactly as the paper's labeling interface wrote them."""
    image, image_rgb = load_image(np.asarray(image))
    base = Path(out_dir) / tile_name
    n = 0
    for nid, ys, xs in iter_nuclei(np.asarray(mask).astype(np.int32)):
        if nid not in labels:
            continue
        color, gray, blue = nucleus_views(image, mask, nid, ys, xs)
        outlined, context = context_views(image_rgb, mask, nid, ys, xs)
        cat = base / FOLDER[labels[nid]]
        name = f"nuc_{tile_name}_{nid}"
        for view, arr in [("rgb", color), ("gray", to_uint8(gray)), ("blue", to_uint8(blue)),
                          ("tissue", outlined), ("raw", context)]:
            (cat / view).mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(cat / view / f"{name}_{view}.png"), arr):
                raise IOError(f"could not write {cat / view / name}")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Export labeled nuclei for retraining.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.labels, newline="") as f:
        labels = {int(r["nucleus_id"]): r["label"].strip() for r in csv.DictReader(f)}
    bad = {v for v in labels.values() if v not in FOLDER}
    if bad:
        raise SystemExit(f"unknown labels {bad}; use accept, reject, reject_border")
    n = export(read_image(args.image), read_mask(args.mask), labels, args.out, Path(args.image).stem)
    print(f"Exported {n} nuclei to {args.out}")


if __name__ == "__main__":
    main()
