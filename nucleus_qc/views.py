"""Five-view representation of each Cellpose-SAM mask.

This reproduces, pixel for pixel, the views the published models were trained
on (the labeling notebook's crops, written to PNG with OpenCV and read back by
the training code). Two details matter for exact reproduction:

* Tiles are read as RGB, but the crops were saved with ``cv2.imwrite`` (which
  assumes BGR) and read back with ``cv2.imread`` + ``COLOR_BGR2RGB``. The color
  and context views the networks saw therefore have the red and blue channels
  swapped relative to the tile. ``model_inputs`` applies the same swap.
* The grayscale view uses ``cv2.COLOR_BGR2GRAY`` on the RGB crop, as in the
  notebook.

Views (order expected by the network): color, outlined (tissue with the mask
boundary in red), context (tissue without boundary), gray, inverted blue.
"""
import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import find_boundaries

BOX_SIZE = 100      # canvas for all views
NUC_PAD = 10        # padding around the mask bounding box for nucleus views
CTX_PAD = 80        # half-size of the 160 x 160 context window


def load_image(img):
    """Same handling as the labeling notebook: returns (image, image_rgb)."""
    if img.ndim == 4:
        img = img[0]
    elif img.ndim == 3 and img.shape[0] <= 4:          # channel-first
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img_rgb = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] in (1, 2):
        img_rgb = np.repeat(img[:, :, :1], 3, axis=2)
    else:
        img_rgb = img
    return img, img_rgb


def _center(arr, h, w, dtype):
    shape = (BOX_SIZE, BOX_SIZE, 3) if arr.ndim == 3 else (BOX_SIZE, BOX_SIZE)
    box = np.zeros(shape, dtype=dtype)
    y0, x0 = (BOX_SIZE - h) // 2, (BOX_SIZE - w) // 2
    box[y0:y0 + h, x0:x0 + w] = arr
    return box


def nucleus_views(image, mask, nid, ys, xs):
    """Color (uint8 HxWx3, tile channel order), gray and inverted-blue (float32)."""
    minr, maxr = max(ys.min() - NUC_PAD, 0), min(ys.max() + NUC_PAD + 1, image.shape[0])
    minc, maxc = max(xs.min() - NUC_PAD, 0), min(xs.max() + NUC_PAD + 1, image.shape[1])
    crop = image[minr:maxr, minc:maxc]
    region = (mask[minr:maxr, minc:maxc] == nid).astype(np.uint8)
    if crop.ndim == 2:
        cm = crop * region
        crop_rgb = np.stack([cm] * 3, axis=-1)
        crop_gray = cm.astype(np.float32)
    else:
        crop_rgb = crop * region[:, :, np.newaxis]
        if crop_rgb.shape[2] == 4:
            crop_rgb = crop_rgb[:, :, :3]
        crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = crop_rgb.shape[:2]
    if max(h, w) > BOX_SIZE:
        s = BOX_SIZE / max(h, w)
        size = (int(w * s), int(h * s))
        crop_rgb = cv2.resize(crop_rgb, size)
        crop_gray = cv2.resize(crop_gray, size)
        region = cv2.resize(region, size, interpolation=cv2.INTER_NEAREST)
        h, w = crop_rgb.shape[:2]
    color = _center(crop_rgb, h, w, np.uint8)
    t = crop_gray.sum()
    if t > 0:
        crop_gray /= t
    gray = _center(crop_gray, h, w, np.float32)
    blue = (255 - crop_rgb[..., 2].astype(np.float32)) * region
    blue = np.clip(blue, 0, 255)
    t2 = blue.sum()
    if t2 > 0:
        blue /= t2
    blue = _center(blue, h, w, np.float32)
    return color, gray, blue


def context_views(image_rgb, mask, nid, ys, xs):
    """Outlined tissue and plain tissue context (uint8, tile channel order)."""
    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    minr, maxr = max(cy - CTX_PAD, 0), min(cy + CTX_PAD, image_rgb.shape[0])
    minc, maxc = max(cx - CTX_PAD, 0), min(cx + CTX_PAD, image_rgb.shape[1])
    out = []
    for outlined in (True, False):
        crop = image_rgb[minr:maxr, minc:maxc].copy()
        if crop.ndim == 3 and crop.shape[2] == 4:
            crop = crop[:, :, :3]
        if outlined:
            b = find_boundaries(mask[minr:maxr, minc:maxc] == nid, mode="outer")
            if b.any():
                crop[b] = [255, 0, 0]
        h, w = crop.shape[:2]
        if max(h, w) > BOX_SIZE:
            s = BOX_SIZE / max(h, w)
            crop = cv2.resize(crop, (int(w * s), int(h * s)))
        if crop.ndim == 3 and crop.shape[2] == 4:
            crop = crop[:, :, :3]
        h, w = crop.shape[:2]
        out.append(_center(crop, h, w, np.uint8))
    return out[0], out[1]


def to_uint8(a):
    """The notebook's save-time scaling of the gray and inverted-blue views."""
    return (a / a.max() * 255).astype(np.uint8) if a.max() > 0 else a.astype(np.uint8)


def model_inputs(image, image_rgb, mask, nid, ys, xs):
    """The five uint8 HxWx3 arrays, in network order, exactly as seen in training:
    [color, outlined, context, gray, inverted blue]."""
    color, gray, blue = nucleus_views(image, mask, nid, ys, xs)
    outlined, context = context_views(image_rgb, mask, nid, ys, xs)
    swap = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)        # PNG round trip in training
    g3 = lambda a: cv2.cvtColor(to_uint8(a), cv2.COLOR_GRAY2RGB)
    return [swap(color), swap(outlined), swap(context), g3(gray), g3(blue)]


def iter_nuclei(mask):
    """Yield (nucleus_id, ys, xs) with full-image coordinates, without scanning
    the whole tile once per nucleus."""
    for i, sl in enumerate(ndimage.find_objects(mask), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(mask[sl] == i)
        if len(ys):
            yield i, ys + sl[0].start, xs + sl[1].start
