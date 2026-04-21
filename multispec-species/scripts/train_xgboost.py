#!/usr/bin/env python3
"""Approach 1: XGBoost species classifier from per-crown spectral stats.

Trains on ForestGEO-labelled crowns (spatial split, N/S halves) and
dumps metrics + model + per-crown predictions.

Usage:
    python train_xgboost.py \
        --spectra  crown_spectra.parquet \
        --labels   ams3d_census_matched_pairs.csv \
        --crowns   ams3d_crown_hulls_in_plot.gpkg \
        --min-per-class 20 \
        --out-dir  results/approach_1_xgboost/

Notes:
    - Labels come from the matched-pairs CSV keyed by (ams_x, ams_y).
      A spatial nearest-neighbour join to crown centroids supplies the
      crown_id → spcode link (tolerance 1 m).
    - Species with fewer than --min-per-class labelled crowns are
      collapsed to `other`.
    - Train/test split is by UTM northing: southern half = train,
      northern half = test (single-cut spatial split; swap and average
      if variance is a concern).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
import xgboost as xgb
import joblib


def load_labels(csv: Path, crowns_gpkg: Path, crown_id_col: str,
                tolerance_m: float = 1.0) -> pd.DataFrame:
    labels = pd.read_csv(csv)
    crowns = gpd.read_file(crowns_gpkg)
    if crown_id_col not in crowns.columns:
        raise KeyError(f"{crown_id_col!r} missing from crowns layer")
    if crowns.crs is None:
        raise ValueError(f"{crowns_gpkg} has no CRS")
    centroids = crowns.geometry.centroid
    crowns_pts = gpd.GeoDataFrame(
        {crown_id_col: crowns[crown_id_col]},
        geometry=centroids, crs=crowns.crs,
    )
    label_pts = gpd.GeoDataFrame(
        labels,
        geometry=[Point(xy) for xy in zip(labels["ams_x"], labels["ams_y"])],
        crs="EPSG:32615",
    ).to_crs(crowns.crs)
    joined = gpd.sjoin_nearest(
        label_pts, crowns_pts, how="inner",
        max_distance=tolerance_m, distance_col="match_dist_m_join",
    )
    return joined[[crown_id_col, "spcode"]].drop_duplicates(subset=[crown_id_col])


def collapse_rare(labels: pd.Series, min_per_class: int) -> pd.Series:
    counts = labels.value_counts()
    keep = counts[counts >= min_per_class].index
    return labels.where(labels.isin(keep), other="other")


def spatial_split(spectra: pd.DataFrame, crowns: gpd.GeoDataFrame,
                  crown_id_col: str) -> pd.Series:
    centroids = crowns.set_index(crown_id_col).geometry.centroid
    y_utm = centroids.y.reindex(spectra[crown_id_col].values).values
    cut = np.nanmedian(y_utm)
    return pd.Series(np.where(y_utm < cut, "train", "test"),
                     index=spectra.index, name="split")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spectra", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--crowns", type=Path, required=True)
    p.add_argument("--crown-id-col", default="crown_id")
    p.add_argument("--min-per-class", type=int, default=20)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    spectra = pd.read_parquet(args.spectra)
    labels = load_labels(args.labels, args.crowns, args.crown_id_col)
    df = spectra.merge(labels, on=args.crown_id_col, how="inner")
    print(f"spectra: {len(spectra)}  labelled: {len(labels)}  joined: {len(df)}")

    df["spcode"] = collapse_rare(df["spcode"], args.min_per_class)
    classes = sorted(df["spcode"].unique())
    label_map = {c: i for i, c in enumerate(classes)}
    df["y"] = df["spcode"].map(label_map)

    crowns = gpd.read_file(args.crowns)
    df["split"] = spatial_split(df, crowns, args.crown_id_col).values

    feat_cols = [c for c in df.columns
                 if c not in (args.crown_id_col, "spcode", "y", "split")]
    Xtr, ytr = df.loc[df.split == "train", feat_cols], df.loc[df.split == "train", "y"]
    Xte, yte = df.loc[df.split == "test", feat_cols], df.loc[df.split == "test", "y"]
    print(f"train: {len(ytr)}  test: {len(yte)}  classes: {len(classes)}")

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(classes),
        tree_method="hist", eval_metric="mlogloss",
        early_stopping_rounds=30, n_jobs=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

    yhat = model.predict(Xte)
    metrics = {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "classes": classes,
        "balanced_accuracy": float(balanced_accuracy_score(yte, yhat)),
        "macro_f1": float(f1_score(yte, yhat, average="macro")),
        "weighted_f1": float(f1_score(yte, yhat, average="weighted")),
        "per_class": classification_report(yte, yhat, target_names=classes,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(yte, yhat).tolist(),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    joblib.dump({"model": model, "label_map": label_map,
                 "feature_cols": feat_cols},
                args.out_dir / "model.joblib")

    Xall = df[feat_cols]
    yall_hat = model.predict(Xall)
    preds = df[[args.crown_id_col, "spcode", "split"]].copy()
    preds["pred"] = [classes[int(i)] for i in yall_hat]
    preds.to_parquet(args.out_dir / "predictions.parquet", index=False)

    print(f"balanced_accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"macro_f1: {metrics['macro_f1']:.3f}")
    print(f"→ {args.out_dir}")


if __name__ == "__main__":
    main()
