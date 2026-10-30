"""The tool's network inputs must equal, pixel for pixel, what training read:
labeling-notebook crops -> cv2.imwrite PNG -> cv2.imread + conversions."""
import os
import tempfile

import cv2
import numpy as np
from skimage.segmentation import find_boundaries

from nucleus_qc.views import iter_nuclei, load_image, model_inputs

BOX_SIZE = 100


def make_tile(seed=0, H=400, W=420):
    rng = np.random.default_rng(seed)
    img = rng.integers(120, 235, (H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), np.int32)
    centers = [(20, 20), (5, 200), (200, 410), (395, 5), (150, 150), (300, 300), (100, 260)]
    for k, (cy, cx) in enumerate(centers, start=1):
        r = int(rng.integers(6, 14))
        cv2.ellipse(mask, (cx, cy), (r, r + 3), int(rng.integers(0, 180)), 0, 360, k, -1)
        cv2.ellipse(img, (cx, cy), (r, r + 3), 0, 0, 360, (int(rng.integers(60, 120)), 40, 150), -1)
    cv2.ellipse(mask, (220, 80), (70, 45), 20, 0, 360, len(centers) + 1, -1)   # > 100 px: resize branch
    return img, mask


def notebook_views(image, image_rgb, mask, sample_ids):
    """Literal copy of the labeling notebook (cells 2 and 8)."""
    nuclei_color, nuclei_gray_norm, nuclei_blue_inv = [], [], []
    for nuc_id in sample_ids:
        ys, xs = np.where(mask == nuc_id)
        PAD = 10
        minr = max(ys.min() - PAD, 0); maxr = min(ys.max() + PAD + 1, image.shape[0])
        minc = max(xs.min() - PAD, 0); maxc = min(xs.max() + PAD + 1, image.shape[1])
        crop = image[minr:maxr, minc:maxc]
        region_mask = (mask[minr:maxr, minc:maxc] == nuc_id).astype(np.uint8)
        crop_masked = crop * region_mask[:, :, np.newaxis]
        crop_rgb = crop_masked
        if crop_rgb.shape[2] == 4:
            crop_rgb = crop_rgb[:, :, :3]
        crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = crop_rgb.shape[:2]
        if max(h, w) > BOX_SIZE:
            scale = BOX_SIZE / max(h, w)
            crop_rgb = cv2.resize(crop_rgb, (int(w * scale), int(h * scale)))
            crop_gray = cv2.resize(crop_gray, (int(w * scale), int(h * scale)))
            region_mask = cv2.resize(region_mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
            h, w = crop_rgb.shape[:2]
        y_offset = (BOX_SIZE - h) // 2; x_offset = (BOX_SIZE - w) // 2
        box_color = np.zeros((BOX_SIZE, BOX_SIZE, 3), dtype=np.uint8)
        box_color[y_offset:y_offset + h, x_offset:x_offset + w] = crop_rgb
        nuclei_color.append(box_color)
        total_intensity = crop_gray.sum()
        if total_intensity > 0:
            crop_gray /= total_intensity
        box_gray = np.zeros((BOX_SIZE, BOX_SIZE), dtype=np.float32)
        box_gray[y_offset:y_offset + h, x_offset:x_offset + w] = crop_gray
        nuclei_gray_norm.append(box_gray)
        blue_channel = crop_rgb[..., 2].astype(np.float32)
        blue_inv = (255 - blue_channel) * region_mask
        blue_inv = np.clip(blue_inv, 0, 255)
        total_intensity2 = blue_inv.sum()
        if total_intensity2 > 0:
            blue_inv /= total_intensity2
        box_blue_inv = np.zeros((BOX_SIZE, BOX_SIZE), dtype=np.float32)
        box_blue_inv[y_offset:y_offset + h, x_offset:x_offset + w] = blue_inv
        nuclei_blue_inv.append(box_blue_inv)
    tissue, raw = [], []
    for nuc_id in sample_ids:
        for out, outlined in ((tissue, True), (raw, False)):
            ys, xs = np.where(mask == nuc_id)
            cy = int(np.mean(ys)); cx = int(np.mean(xs)); PAD = 80
            minr = max(cy - PAD, 0); maxr = min(cy + PAD, image_rgb.shape[0])
            minc = max(cx - PAD, 0); maxc = min(cx + PAD, image_rgb.shape[1])
            c = image_rgb[minr:maxr, minc:maxc].copy()
            if c.ndim == 3 and c.shape[2] == 4:
                c = c[:, :, :3]
            if outlined:
                boundary = find_boundaries(mask[minr:maxr, minc:maxc] == nuc_id, mode='outer')
                if boundary.any():
                    c[boundary] = [255, 0, 0]
            h, w = c.shape[:2]
            if max(h, w) > BOX_SIZE:
                scale = BOX_SIZE / max(h, w)
                c = cv2.resize(c, (int(w * scale), int(h * scale)))
            h, w = c.shape[:2]
            box = np.zeros((BOX_SIZE, BOX_SIZE, 3), dtype=np.uint8)
            box[(BOX_SIZE - h) // 2:(BOX_SIZE - h) // 2 + h, (BOX_SIZE - w) // 2:(BOX_SIZE - w) // 2 + w] = c
            out.append(box)
    return nuclei_color, nuclei_gray_norm, nuclei_blue_inv, tissue, raw


def training_read(color, gray, blue, tissue, raw, d):
    """NucleiReviewer.save() writes, then MaskDataset.__getitem__ reads."""
    g = (gray / gray.max() * 255).astype(np.uint8) if gray.max() > 0 else gray
    b = (blue / blue.max() * 255).astype(np.uint8) if blue.max() > 0 else blue
    for name, a in [("rgb", color), ("gray", g), ("blue", b), ("tissue", tissue), ("raw", raw)]:
        cv2.imwrite(os.path.join(d, f"{name}.png"), a)
    rd = lambda n: cv2.cvtColor(cv2.imread(os.path.join(d, f"{n}.png")), cv2.COLOR_BGR2RGB)
    rg = lambda n: cv2.cvtColor(cv2.imread(os.path.join(d, f"{n}.png"), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
    return [rd("rgb"), rd("tissue"), rd("raw"), rg("gray"), rg("blue")]


def test_views_identical_to_training():
    img, mask = make_tile()
    image, image_rgb = load_image(img)
    ids = sorted(int(i) for i in np.unique(mask) if i)
    nb = notebook_views(image, image_rgb, mask, ids)
    tool = {nid: model_inputs(image, image_rgb, mask, nid, ys, xs) for nid, ys, xs in iter_nuclei(mask)}
    assert sorted(tool) == ids
    with tempfile.TemporaryDirectory() as d:
        for k, nid in enumerate(ids):
            expected = training_read(*(v[k] for v in nb), d)
            for e, t in zip(expected, tool[nid]):
                assert e.shape == t.shape == (100, 100, 3)
                assert np.array_equal(e, t), f"nucleus {nid} differs"


def test_export_then_train_reader_roundtrip():
    """export() + the training reader give the same arrays as the inference path."""
    from nucleus_qc.export import export
    from nucleus_qc.train import find_items, VIEWS
    img, mask = make_tile(1)
    image, image_rgb = load_image(img)
    ids = sorted(int(i) for i in np.unique(mask) if i)
    labels = {nid: ["accept", "reject", "reject_border"][k % 3] for k, nid in enumerate(ids)}
    with tempfile.TemporaryDirectory() as d:
        assert export(img, mask, labels, d, "t") == len(ids)
        items = find_items(d)
        assert len(items) == len(ids)
        tool = {nid: model_inputs(image, image_rgb, mask, nid, ys, xs) for nid, ys, xs in iter_nuclei(mask)}
        for paths, label in items:
            nid = int(paths["rgb"].name.split("_")[-2])
            assert label == (0 if labels[nid] == "accept" else 1)
            for v, t in zip(VIEWS, tool[nid]):
                a = cv2.imread(str(paths[v]), cv2.IMREAD_GRAYSCALE if v in ("gray", "blue") else cv2.IMREAD_COLOR)
                a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB if v in ("gray", "blue") else cv2.COLOR_BGR2RGB)
                assert np.array_equal(a, t), (nid, v)


if __name__ == "__main__":
    test_views_identical_to_training()
    test_export_then_train_reader_roundtrip()
    print("OK: inference and export views are pixel-identical to the training pipeline")
