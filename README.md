# nucleus-qc

**Nucleus-level quality control of Cellpose-SAM segmentations on H&E.**

Generalist segmenters such as [Cellpose-SAM](https://github.com/MouseLand/cellpose)
make nucleus segmentation easy at whole-slide scale, but some of their masks are
wrong: merged nuclei, partially segmented nuclei, outlines around non-nuclear
material, jagged boundaries, and nuclei truncated at a region border. Kept,
these masks distort nuclear size and shape measurements.

`nucleus-qc` scores **every mask** and flags the likely errors, so downstream
morphometry runs on correctly segmented nuclei without manual review. It needs
only the image and the mask, so it follows an existing Cellpose-SAM pipeline
unchanged.

<!-- Optional: add a figure here, e.g. docs/fig1.png (example masks in the five views). -->

## How it works

Each mask is represented by the five views a human reviewer uses: the masked
nucleus in color, grayscale, and inverted blue (chromatin), plus the
surrounding tissue with and without the mask outline. Each view goes through
its own ImageNet-pretrained EfficientNet-B0; the features are concatenated and
classified as *accept* or *reject*. Three models trained with different seeds
are averaged.

## Performance (oral cavity squamous cell carcinoma, H&E, 40×)

| Evaluation | AUC | Erroneous masks among kept nuclei | Nuclei kept |
|---|---|---|---|
| Leave-one-patient-out, 4 patients (14,952 masks) | 0.925 | 15.8% → 4.5% | 79% |
| Ten held-out patients (8,000 masks), evaluated once | **0.937** | 21.9% → 6.2% | 77% |

On the held-out patients it outperformed a random forest on hand-crafted
shape and stain features on all ten patients. See the paper for details.

## Install

```bash
pip install git+https://github.com/ksong45/nucleus-qc
# optional, to segment tiles directly:
pip install "nucleus-qc[cellpose] @ git+https://github.com/ksong45/nucleus-qc"
```

The pretrained models (three files, ~95 MB each) download automatically on
first use to `~/.cache/nucleus_qc` (change with `NUCLEUS_QC_WEIGHTS`). A GPU is
recommended but not required.

## Use

**Command line**, with an existing Cellpose mask (`.tif`/`.png` label image or
Cellpose `*_seg.npy`):

```bash
nucleus-qc --image tile.tif --mask tile_mask.tif --out results/ --overlay
```

or let it run Cellpose-SAM first (`pip install` the `cellpose` extra):

```bash
nucleus-qc --image tile.tif --run-cellpose --out results/
```

**Python:**

```python
import imageio.v2 as imageio
from nucleus_qc import run_qc

image = imageio.imread("tile.tif")
mask = imageio.imread("tile_mask.tif")
rows, clean_mask = run_qc(image, mask)        # rows: one dict per nucleus
```

### Outputs

| File | Content |
|---|---|
| `<tile>_qc.csv` | one row per nucleus: `nucleus_id`, centroid, `area_px`, the three model probabilities, `prob_reject` (ensemble), `decision` |
| `<tile>_mask_qc.tif` | the input mask with rejected nuclei removed (set to 0) |
| `<tile>_qc_overlay.png` | with `--overlay`: accepted outlines in green, rejected in red |

### Choosing the threshold

`decision` is `reject` when `prob_reject ≥ 0.489` (`--threshold`), the
operating point used in the paper, which balances the two recalls. Lower it to
remove more errors (cleaner measurements, fewer nuclei kept); raise it to keep
more nuclei. Because `prob_reject` is saved for every nucleus, you can also
re-threshold later without rerunning.

### Input requirements

* RGB H&E tiles at **40× (≈0.23 µm/pixel)**, the resolution of the training
  data. Other magnifications should be resampled to ≈0.23 µm/pixel first.
* Masks from **Cellpose-SAM** (`cpsam`, Cellpose ≥ 4) with default parameters.
  Masks from other segmenters may work but were not evaluated.
* As in the paper, pixels outside the region of interest may be set to zero;
  nuclei cut by that boundary are flagged as border errors.

## Retraining on new tissue, stains, or cancer types

The model learns from views rather than hand-designed rules, so it can be
retrained for other settings from labeled masks:

```bash
# 1. labels.csv: nucleus_id,label   (label: accept | reject | reject_border)
nucleus-qc-export --image tile.tif --mask tile_mask.tif --labels labels.csv --out labeled/
# 2. train three seeds with the paper's settings
for s in 42 123 456; do
  nucleus-qc-train --data labeled/ --seed $s --out mymodels/nucleus_qc_seed$s.pth
done
# 3. use them
nucleus-qc --image new.tif --mask new_mask.tif --weights-dir mymodels/ --out results/
```

A threshold for new models should be chosen on validation data (e.g., the
point maximizing reject recall + accept recall − 1).

## Reproducibility

`tests/test_views.py` checks that the views computed by this package are
pixel-identical to those used to train the released models.

```bash
python -m tests.test_views
```

## Citation

If you use this software, please cite:

> K. Song, N. S. Pathan, M. Shifat-E-Rabbi, A. Murthy, and G. K. Rohde,
> "Automated nucleus-level quality control of Cellpose segmentations in H&E
> oral cancer tissue," submitted to *IEEE International Symposium on
> Biomedical Imaging (ISBI)*, 2027.

## License

MIT (see `LICENSE`). The models were trained on de-identified slides under
University of Virginia IRB approval; no patient images are distributed.
