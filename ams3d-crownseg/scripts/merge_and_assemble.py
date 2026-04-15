#!/usr/bin/env python3
"""Assemble per-tile AMS3D output into a single COPC.

Four steps:
  1. Union per-tile parquet → single GeoPackage of tree attributes
  2. NN relabel orphan canopy points (points in tile A whose tree's
     centroid lives in tile B, so they have treeID=NA in tile A's LAZ)
  3. Bake random RGB per global tree ID into each tile LAZ
  4. PDAL writers.copc to produce a single COPC for the whole landscape

Run from inside the ams3d-crownseg container on Compute2, or locally with
the same dependencies installed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import KDTree as cKDTree
from shapely.geometry import Point


TILE_NAME_RE = re.compile(r"^tile_(-?\d+)_(-?\d+)$")
DEFAULT_NN_MAX_DIST = 3.0
# lidR encodes NA_integer_ as INT_MAX in LAS extra dims; also matches
# crownsegmentr's ground catch-all. Treat as unassigned downstream.
NA_ID = 2147483647


def union_parquets(parquet_dir: Path, out_gpkg: Path, crs: str) -> pd.DataFrame:
    """Union per-tile tree attributes into one GeoPackage."""
    files = sorted(parquet_dir.glob("*_trees.parquet"))
    print(f"unioning {len(files)} parquet files")
    dfs = [pq.read_table(f).to_pandas() for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  total rows: {len(df):,}")
    print(f"  core+real trees: {int((df['is_real'] == True).sum()):,}")

    real_only = df[df["is_real"] == True].copy()
    geom = [Point(x, y) for x, y in zip(real_only["centroid_x"], real_only["centroid_y"])]
    gdf = gpd.GeoDataFrame(real_only, geometry=geom, crs=crs)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, driver="GPKG")
    print(f"  wrote {out_gpkg}")
    return df


def relabel_orphans_and_color(laz_in: Path, laz_out: Path, trees_df: pd.DataFrame,
                               max_dist: float, palette_seed: int = 42) -> dict:
    """Relabel orphan points to nearest core tree, then bake random RGB.

    Orphan points = canopy points whose local tree was filtered out
    (centroid outside this tile's core). We look up the nearest tree
    centroid from *any* tile within max_dist m and assign that tree.
    """
    m = TILE_NAME_RE.match(laz_in.stem)
    if not m:
        raise ValueError(f"unexpected tile name: {laz_in.stem}")
    grid_x, grid_y = int(m.group(1)), int(m.group(2))

    las = laspy.read(str(laz_in))
    local_ids = np.asarray(las.treeID if "treeID" in las.point_format.dimension_names
                            else las.PredInstance, dtype=np.int64)
    classif = np.asarray(las.classification, dtype=np.int64)
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])

    is_assigned = (local_ids > 0) & (local_ids != NA_ID)
    is_ground = classif == 2
    is_orphan = ~is_assigned & ~is_ground  # above-ground with no tree

    # Build kdtree of ALL core-real centroids (across all tiles)
    real = trees_df[trees_df["is_real"] == True]
    centroid_xyz = np.column_stack([real["centroid_x"], real["centroid_y"],
                                      real["centroid_z"]])
    tree_tree = cKDTree(centroid_xyz)
    global_ids = real["global_id"].to_numpy()

    # Compose global_id for locally-assigned points
    #   high 16: grid_x+128, next 16: grid_y+128, low 32: local_id
    #   We use string ids for cross-compatibility with R's string form.
    def gid_string(local):
        return f"{grid_x}_{grid_y}_{int(local)}"

    all_gids = np.array(["" for _ in range(len(local_ids))], dtype=object)
    # Locally-assigned: directly build global id
    if is_assigned.any():
        all_gids[is_assigned] = [gid_string(l) for l in local_ids[is_assigned]]

    # Orphans: NN to nearest core centroid
    n_relabeled = 0
    if is_orphan.any():
        dists, idxs = tree_tree.query(xyz[is_orphan], k=1, workers=-1)
        within = dists <= max_dist
        relabel_gids = np.where(within, global_ids[idxs], "")
        orphan_positions = np.where(is_orphan)[0]
        all_gids[orphan_positions] = relabel_gids
        n_relabeled = int(within.sum())

    # Bake RGB per unique global_id via deterministic hash
    rng = np.random.default_rng(palette_seed)
    unique_gids, inverse = np.unique(all_gids, return_inverse=True)
    palette = rng.integers(30, 256, size=(len(unique_gids), 3), dtype=np.int32)
    empty_idx = np.where(unique_gids == "")[0]
    if len(empty_idx):
        palette[empty_idx[0]] = [40, 40, 40]  # unassigned stays dark gray
    red = palette[inverse, 0]
    green = palette[inverse, 1]
    blue = palette[inverse, 2]
    # Ground override
    red[is_ground]   = 90
    green[is_ground] = 65
    blue[is_ground]  = 40

    las.red   = (red   * 257).astype(np.uint16)
    las.green = (green * 257).astype(np.uint16)
    las.blue  = (blue  * 257).astype(np.uint16)

    # Drop buffer-zone points to prevent double-counting in the merged COPC.
    # Each tile's core is (grid_x*100 + 713459.02, grid_y*100 + 4265580.08)
    # to (+100, +100). Points outside core get dropped via laspy filter.
    OFFSET_X, OFFSET_Y, BUFFER = 713449.02, 4265570.08, 10
    core_minx = grid_x * 100 + OFFSET_X + BUFFER
    core_miny = grid_y * 100 + OFFSET_Y + BUFFER
    in_core = (xyz[:, 0] >= core_minx) & (xyz[:, 0] < core_minx + 100) & \
              (xyz[:, 1] >= core_miny) & (xyz[:, 1] < core_miny + 100)
    if not in_core.all():
        las.points = las.points[in_core]

    laz_out.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(laz_out))
    return {
        "tile": laz_in.stem,
        "n_assigned": int(is_assigned.sum()),
        "n_orphan": int(is_orphan.sum()),
        "n_relabeled": n_relabeled,
        "n_dropped_buffer": int((~in_core).sum()),
    }


def write_copc(laz_dir: Path, copc_out: Path) -> None:
    """Merge all per-tile LAZs into a single COPC via PDAL."""
    copc_out.parent.mkdir(parents=True, exist_ok=True)
    laz_glob = str(laz_dir / "*.laz")
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": laz_glob},
            {"type": "writers.copc", "filename": str(copc_out)},
        ]
    }
    pipeline_file = copc_out.parent / "merge_copc.json"
    pipeline_file.write_text(json.dumps(pipeline, indent=2))
    print(f"running pdal pipeline → {copc_out}")
    subprocess.run(["pdal", "pipeline", str(pipeline_file)], check=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seg-root", type=Path, required=True,
                   help="directory containing laz/, parquet/ subdirs")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--crs", default="EPSG:32615")
    p.add_argument("--max-dist", type=float, default=DEFAULT_NN_MAX_DIST)
    p.add_argument("--skip-merge", action="store_true",
                   help="only build gpkg + colored LAZs, skip final COPC")
    args = p.parse_args()

    t0 = time.perf_counter()

    # Step 1: union parquets
    gpkg_out = args.out_dir / "tyson_ams3d_trees.gpkg"
    trees_df = union_parquets(args.seg_root / "parquet", gpkg_out, args.crs)

    # Step 2+3: orphan relabel + color per tile
    colored_dir = args.out_dir / "colored_laz"
    laz_files = sorted((args.seg_root / "laz").glob("tile_*_*.laz"))
    print(f"\ncoloring + filtering {len(laz_files)} tiles")
    stats = []
    for i, f in enumerate(laz_files):
        s = relabel_orphans_and_color(f, colored_dir / f.name, trees_df, args.max_dist)
        stats.append(s)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(laz_files)}] {f.stem}")
    (args.out_dir / "relabel_stats.json").write_text(json.dumps(stats, indent=2))

    # Step 4: COPC
    if not args.skip_merge:
        copc_path = args.out_dir / "tyson.copc.laz"
        write_copc(colored_dir, copc_path)
        print(f"\nCOPC: {copc_path} ({copc_path.stat().st_size / 1e9:.2f} GB)")

    print(f"\ntotal elapsed: {(time.perf_counter() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
