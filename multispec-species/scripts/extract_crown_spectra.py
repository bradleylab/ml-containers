#!/usr/bin/env python3
"""Extract per-crown spectral stats from an Altum-PT 5-band mosaic.

One row per crown, one column per (band, statistic) plus derived
vegetation indices. Output: parquet of shape (n_crowns, ~35 features).

Usage:
    python extract_crown_spectra.py \
        --crowns   /path/to/crown_hulls.gpkg \
        --raster   /path/to/5band_mosaic.tif \
        --crown-id-col crown_id \
        --out      /path/to/crown_spectra.parquet

Pre-requisites:
    pip install geopandas rasterio rasterstats pyarrow numpy pandas

Assumptions:
    - Raster has 5 bands in Altum-PT order: Blue, Green, Red, RedEdge, NIR
      (matches ProductNotes_TRC_20251013.txt band order).
    - Crowns may be in a different CRS; reprojected to raster CRS before
      zonal stats so the 6.57 cm raster is never resampled.
    - Reflectance is Float32 in 0..1. No rescaling applied.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats

BANDS = ("blue", "green", "red", "rededge", "nir")
STATS = ("mean", "median", "std", "percentile_10", "percentile_90")


def compute_vegetation_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add NDVI, NDRE, GNDVI, CIrededge computed on per-crown mean bands."""
    m = {b: df[f"{b}_mean"] for b in BANDS}
    eps = 1e-9
    df["ndvi"] = (m["nir"] - m["red"]) / (m["nir"] + m["red"] + eps)
    df["ndre"] = (m["nir"] - m["rededge"]) / (m["nir"] + m["rededge"] + eps)
    df["gndvi"] = (m["nir"] - m["green"]) / (m["nir"] + m["green"] + eps)
    df["ci_rededge"] = (m["nir"] / (m["rededge"] + eps)) - 1.0
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--crowns", type=Path, required=True)
    p.add_argument("--raster", type=Path, required=True)
    p.add_argument("--crown-id-col", default="crown_id")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with rasterio.open(args.raster) as src:
        raster_crs = src.crs
        n_bands = src.count
    assert n_bands == 5, f"expected 5-band mosaic, got {n_bands}"

    crowns = gpd.read_file(args.crowns)
    if args.crown_id_col not in crowns.columns:
        raise KeyError(f"{args.crown_id_col!r} not in crown layer; "
                       f"available: {list(crowns.columns)}")
    if crowns.crs != raster_crs:
        print(f"reprojecting crowns {crowns.crs} → {raster_crs}")
        crowns = crowns.to_crs(raster_crs)

    rows = []
    for band_idx, band_name in enumerate(BANDS, start=1):
        stats = zonal_stats(
            crowns.geometry,
            str(args.raster),
            band=band_idx,
            stats=list(STATS),
            nodata=None,
            geojson_out=False,
            all_touched=False,
        )
        df_band = pd.DataFrame(stats)
        df_band.columns = [f"{band_name}_{s}" for s in df_band.columns]
        rows.append(df_band)

    spectra = pd.concat(rows, axis=1)
    spectra[args.crown_id_col] = crowns[args.crown_id_col].values
    spectra = compute_vegetation_indices(spectra)

    cols = [args.crown_id_col] + [c for c in spectra.columns if c != args.crown_id_col]
    spectra = spectra[cols]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    spectra.to_parquet(args.out, index=False)
    print(f"wrote {len(spectra)} crowns × {spectra.shape[1] - 1} features → {args.out}")


if __name__ == "__main__":
    main()
