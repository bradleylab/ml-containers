#!/usr/bin/env python
"""Linear probe of AnySat features against NLCD 2021 land cover.

Reads the AOI written by ``fetch_aoi.py`` and the features written by
``anysat_features.py --dense s2`` for it, labels every patch with its modal
NLCD class, and fits the same linear classifier to each feature set:

  anysat_patch     the fused ``patch`` output (768 values)
  anysat_dense     the ``dense`` output for s2, averaged over the patch's
                   10 m sub-patches (1536 values: the fused patch token, then
                   the sub-patch feature)
  anysat_subpatch  the second half of ``anysat_dense`` alone (768 values)
  baseline         raw band means over the patch: 4 NAIP bands over its
                   pixels and 10 Sentinel-2 bands over its pixels and all
                   dates (14 values)

The three AnySat sets are scored separately because they behave differently:
the fused patch tokens of one tile are close to identical, while the
sub-patch half keeps local detail, which is why upstream points
segmentation to ``dense``. Without ``dense`` in the .npz only
``anysat_patch`` and the baseline are scored.

Evaluation is four-fold spatial cross-validation: each fold holds out one
quadrant of the AOI and trains on the other three, so no test patch has a
training neighbor from the same quadrant. A test patch whose class is absent
from that fold's training quadrants cannot be predicted by either method and
is left out of that fold, with the count reported. The metrics are macro-F1
over the classes present in the test quadrant (the primary one, because
deciduous forest dominates the AOI) and overall accuracy.

NLCD's legend is hierarchical: the first digit of a class code is its
Level I class (1 water, 2 developed, 3 barren, 4 forest, 5 shrubland,
7 herbaceous, 8 planted/cultivated, 9 wetlands). ``--level 1`` probes those;
``--level 2`` probes the full codes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NLCD_NODATA = 0
# lbfgs iteration cap: at the solver's default of 100, fits on the AnySat
# features end with a ConvergenceWarning.
MAX_ITER = 5000
EMBED_DIM = 768
QUADRANTS = ("NW", "NE", "SW", "SE")


def block_view(a: np.ndarray, k: int) -> np.ndarray:
    """(..., H, W) -> (..., H/k, W/k, k*k): the pixels of each k x k block."""
    *lead, h, w = a.shape
    b = a.reshape(*lead, h // k, k, w // k, k)
    b = np.moveaxis(b, -3, -2)
    return b.reshape(*lead, h // k, w // k, k * k)


def patch_labels(nlcd: np.ndarray, k: int, level: int) -> tuple[np.ndarray, np.ndarray]:
    """Modal NLCD class per patch and the fraction of pixels that hold it."""
    codes = nlcd // 10 if level == 1 else nlcd
    blocks = block_view(codes, k)
    valid = (block_view(nlcd, k) != NLCD_NODATA).all(axis=-1)
    labels, purity = np.zeros(blocks.shape[:2], np.int64), np.zeros(blocks.shape[:2])
    for idx in np.ndindex(*blocks.shape[:2]):
        values, counts = np.unique(blocks[idx], return_counts=True)
        labels[idx], purity[idx] = values[counts.argmax()], counts.max() / counts.sum()
    labels[~valid] = -1
    return labels, purity


def band_means(data: Path, n_rows: int) -> np.ndarray:
    """Baseline features: raw band means per patch, NAIP then Sentinel-2."""
    with rasterio.open(data / "naip.tif") as src:
        naip = src.read()
    with rasterio.open(data / "s2.tif") as src:
        s2 = src.read()  # (T*10, H, W), date-major
    s2 = s2.reshape(-1, 10, *s2.shape[1:]).mean(axis=0)
    naip_means = block_view(naip, naip.shape[1] // n_rows).mean(axis=-1)
    s2_means = block_view(s2, s2.shape[1] // n_rows).mean(axis=-1)
    return np.concatenate([naip_means, s2_means]).transpose(1, 2, 0)


def anysat_sets(npz: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    sets = {"anysat_patch": npz["patch"]}
    if "dense" in npz:
        dense = np.moveaxis(npz["dense"], -1, 0)
        pooled = block_view(dense, dense.shape[1] // npz["patch"].shape[0]).mean(axis=-1)
        pooled = pooled.transpose(1, 2, 0)
        sets["anysat_dense"] = pooled
        sets["anysat_subpatch"] = pooled[..., EMBED_DIM:]
    return sets


def quadrant_of(n_rows: int, n_cols: int) -> np.ndarray:
    r, c = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    return 2 * (r >= n_rows // 2) + (c >= n_cols // 2)


def evaluate(x: np.ndarray, y: np.ndarray, quadrant: np.ndarray) -> list[dict]:
    folds = []
    for q, name in enumerate(QUADRANTS):
        train, test = quadrant != q, quadrant == q
        seen = np.isin(y, np.unique(y[train]))
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=MAX_ITER))
        clf.fit(x[train], y[train])
        keep = test & seen
        pred = clf.predict(x[keep])
        folds.append(
            {
                "held_out": name,
                "n_train": int(train.sum()),
                "n_test": int(keep.sum()),
                "n_test_unseen_class": int((test & ~seen).sum()),
                "macro_f1": float(f1_score(y[keep], pred, labels=np.unique(y[keep]), average="macro")),
                "accuracy": float(accuracy_score(y[keep], pred)),
            }
        )
    return folds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="directory written by fetch_aoi.py")
    p.add_argument("--features", type=Path, required=True, help=".npz written by anysat_features.py")
    p.add_argument("--level", type=int, choices=(1, 2), default=1)
    p.add_argument("--out", type=Path, required=True, help="report .json")
    args = p.parse_args()

    features = anysat_sets(np.load(args.features))
    n_rows, n_cols, _ = features["anysat_patch"].shape
    with rasterio.open(args.data / "nlcd.tif") as src:
        nlcd = src.read(1)
    labels, purity = patch_labels(nlcd, nlcd.shape[0] // n_rows, args.level)
    quadrant = quadrant_of(n_rows, n_cols)
    features["baseline"] = band_means(args.data, n_rows)

    labeled = labels >= 0
    y, q = labels[labeled], quadrant[labeled]
    report = {
        "nlcd_level": args.level,
        "patch_grid": [n_rows, n_cols],
        "n_unlabeled_patches": int((~labeled).sum()),
        "median_label_purity": float(np.median(purity[labeled])),
        "class_counts_by_quadrant": {
            name: {int(c): int(n) for c, n in zip(*np.unique(y[q == i], return_counts=True))}
            for i, name in enumerate(QUADRANTS)
        },
        "results": {},
    }
    for name, f in features.items():
        folds = evaluate(f[labeled].astype(np.float64), y, q)
        report["results"][name] = {
            "n_features": int(f.shape[-1]),
            "folds": folds,
            "mean_macro_f1": float(np.mean([d["macro_f1"] for d in folds])),
            "mean_accuracy": float(np.mean([d["accuracy"] for d in folds])),
        }
    args.out.write_text(json.dumps(report, indent=2))

    print(f"NLCD level {args.level}; per held-out quadrant, macro-F1 / accuracy")
    for name, r in report["results"].items():
        cells = "  ".join(f"{d['held_out']} {d['macro_f1']:.3f}/{d['accuracy']:.3f}" for d in r["folds"])
        print(f"{name:15s} {cells}  mean {r['mean_macro_f1']:.3f}/{r['mean_accuracy']:.3f}")
    print("->", args.out)


if __name__ == "__main__":
    main()
