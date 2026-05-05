#!/usr/bin/env python
"""Run treeX (TreeXAlgorithm) on a single LAS/LAZ tile.

This is a thin wrapper for the lab pipeline. Reads a LAS/LAZ tile,
runs TreeXAlgorithm with one of the published presets (default: ULS,
since our typical input is UAV lidar), and writes a segmented LAZ
plus a parquet of detected trunk positions and diameters.

Reproducibility: TreeXAlgorithm has a `random_seed` parameter; we set
it to a fixed default (42) and expose it on the CLI.

Usage
-----
run_treex.py \\
    --input /path/to/tile.laz \\
    --output-dir /path/to/output \\
    --preset uls \\
    [--seed 42] [--workers -1] [--vis-dir /path/to/visualizations]

Outputs (in --output-dir):
    <stem>_treex.laz       — input points + an `instance_id` column
    <stem>_treex_trunks.parquet — table of trunk centers (x, y) + diameter
"""

# pointtree + pointtorch are installed only inside the container image
# (ghcr.io/bradleylab/treex). Local static analyzers cannot resolve them.
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pointtorch import read
from pointtree.instance_segmentation import (
    TreeXAlgorithm,
    TreeXPresetTLS,
    TreeXPresetULS,
)


PRESETS = {
    "tls": TreeXPresetTLS,
    "uls": TreeXPresetULS,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run treeX (TreeXAlgorithm) on a single LAS/LAZ tile."
    )
    p.add_argument(
        "--input", required=True, type=Path,
        help="Input LAS/LAZ/PLY/CSV tile.",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory for outputs (created if missing).",
    )
    p.add_argument(
        "--preset", choices=sorted(PRESETS.keys()), default="uls",
        help="TreeX preset (TLS / PLS-style or ULS-style). Default: uls.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    p.add_argument(
        "--workers", type=int, default=-1,
        help="Worker count for parallel ops (-1 = all CPU threads, default).",
    )
    p.add_argument(
        "--vis-dir", type=Path, default=None,
        help="Optional directory for intermediate visualizations. "
             "Slows processing — recommended only for small tiles.",
    )
    p.add_argument(
        "--crs", type=str, default=None,
        help="Optional CRS string passed through to TreeXAlgorithm "
             "(e.g. 'EPSG:6347' for the Tyson UAV tiles).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.vis_dir is not None:
        args.vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading {args.input} …", flush=True)
    point_cloud = read(str(args.input))

    xyz = point_cloud[["x", "y", "z"]].to_numpy()
    intensities = (
        point_cloud["intensity"].to_numpy()
        if "intensity" in point_cloud.columns
        else None
    )
    print(
        f"  {xyz.shape[0]:,} points, "
        f"intensity={'yes' if intensities is not None else 'no'}",
        flush=True,
    )

    # TreeXPreset{TLS,ULS} contain `random_seed`, `num_workers`, and
    # `visualization_folder` among their fields. Passing the CLI values
    # explicitly *and* splatting **preset collides on those three keys
    # (`TypeError: got multiple values for keyword argument`). Build a
    # single kwargs dict instead, with CLI values overriding preset
    # defaults.
    preset = PRESETS[args.preset]()
    preset_kwargs = {**preset}  # same splat protocol as the upstream API
    preset_kwargs["random_seed"] = args.seed
    preset_kwargs["num_workers"] = args.workers
    preset_kwargs["visualization_folder"] = (
        str(args.vis_dir) if args.vis_dir else None
    )
    algorithm = TreeXAlgorithm(**preset_kwargs)

    stem = args.input.stem.replace(".laz", "").replace(".las", "")
    print(f"running TreeXAlgorithm (preset={args.preset}, seed={args.seed}) …", flush=True)
    instance_ids, trunk_positions, trunk_diameters = algorithm(
        xyz,
        intensities=intensities,
        point_cloud_id=stem,
        crs=args.crs,
    )
    n_trees = int((np.unique(instance_ids) >= 0).sum())
    print(f"  detected {n_trees} tree instances", flush=True)

    # Write segmented point cloud (input + instance_id column).
    # PointCloud.to() always writes x, y, z; pass only the additional
    # columns we want preserved alongside.
    point_cloud["instance_id"] = instance_ids
    seg_path = args.output_dir / f"{stem}_treex.laz"
    extra_cols = [
        c for c in point_cloud.columns if c not in ("x", "y", "z")
    ]
    point_cloud.to(str(seg_path), columns=extra_cols)
    print(f"  wrote {seg_path}", flush=True)

    # Write trunk table.
    trunks = pd.DataFrame(
        {
            "trunk_x": trunk_positions[:, 0],
            "trunk_y": trunk_positions[:, 1],
            "trunk_diameter": trunk_diameters,
        }
    )
    trunks_path = args.output_dir / f"{stem}_treex_trunks.parquet"
    trunks.to_parquet(trunks_path, index=False)
    print(f"  wrote {trunks_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
