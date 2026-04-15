#!/usr/bin/env python3
"""Bake random RGB per global tree ID across many per-tile LAZ files.

The per-tile LAZ files produced by run_tile_ams3d.R contain a local `treeID`
(int32). To render the whole landscape with a consistent random color per
tree, we synthesize a 64-bit global ID from each tile's (grid_x, grid_y)
prefix and the local ID, then hash it to a 24-bit RGB color.

This script is intended to run either:
  (a) per-tile, before COPC merging — bakes RGB into each LAZ
  (b) post-merge, if the merged COPC preserves a per-point `tile_name` attr

Usage:
  render_random_colors_u64.py --in-dir seg/cd40/laz --out-dir seg/cd40/laz_rgb
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import laspy
import numpy as np


RANDOM_SEED = 42
UNASSIGNED_RGB = np.array([40, 40, 40], dtype=np.uint16)  # dark gray
GROUND_RGB = np.array([90, 65, 40], dtype=np.uint16)       # brown
TILE_NAME_RE = re.compile(r"^tile_(-?\d+)_(-?\d+)$")
# lidR encodes NA_integer_ as INT_MAX when writing LAS extra dims.
# crownsegmentr also uses INT_MAX for the ground/low-veg catch-all cluster,
# so both end up as 2147483647 in the LAZ — treat as unassigned.
NA_ID = 2147483647


def parse_tile_name(stem: str) -> tuple[int, int]:
    m = TILE_NAME_RE.match(stem)
    if not m:
        raise ValueError(f"tile stem does not match tile_X_Y: {stem}")
    return int(m.group(1)), int(m.group(2))


def global_id_u64(grid_x: int, grid_y: int, local_id: np.ndarray) -> np.ndarray:
    """Pack (grid_x, grid_y, local_id) into a single uint64.

    Encoding: top 16 bits = grid_x + 128, next 16 = grid_y + 128,
    bottom 32 = local_id. Tile grids span roughly [-25, 10] so
    adding 128 keeps the shifted values positive and fitting in 16 bits.
    """
    gx = np.uint64(grid_x + 128)
    gy = np.uint64(grid_y + 128)
    return (gx << np.uint64(48)) | (gy << np.uint64(32)) | local_id.astype(np.uint64)


def hash_to_rgb(ids: np.ndarray) -> np.ndarray:
    """Deterministic random-looking RGB per unique id."""
    unique, inverse = np.unique(ids, return_inverse=True)
    rng = np.random.default_rng(RANDOM_SEED)
    palette = rng.integers(30, 256, size=(len(unique), 3), dtype=np.int32)
    return palette[inverse]


def process_tile(in_path: Path, out_path: Path) -> dict:
    grid_x, grid_y = parse_tile_name(in_path.stem)
    las = laspy.read(str(in_path))
    classif = np.asarray(las.classification, dtype=np.int64)

    # Prefer the local int32 treeID the R worker writes.
    if "treeID" in las.point_format.dimension_names:
        raw = np.asarray(las.treeID)
    elif "PredInstance" in las.point_format.dimension_names:
        raw = np.asarray(las.PredInstance)
    else:
        raise SystemExit(f"no treeID / PredInstance dim in {in_path}")

    if np.issubdtype(raw.dtype, np.floating):
        local_ids = np.where(np.isnan(raw), 0, raw).astype(np.int64)
    else:
        local_ids = raw.astype(np.int64)

    is_assigned = (local_ids > 0) & (local_ids != NA_ID)
    is_ground = classif == 2

    red = np.full(len(las.x), UNASSIGNED_RGB[0], dtype=np.int32)
    green = np.full(len(las.x), UNASSIGNED_RGB[1], dtype=np.int32)
    blue = np.full(len(las.x), UNASSIGNED_RGB[2], dtype=np.int32)

    ground_unassigned = is_ground & ~is_assigned
    red[ground_unassigned]   = GROUND_RGB[0]
    green[ground_unassigned] = GROUND_RGB[1]
    blue[ground_unassigned]  = GROUND_RGB[2]

    if is_assigned.any():
        gids = global_id_u64(grid_x, grid_y, local_ids[is_assigned])
        palette = hash_to_rgb(gids)
        red[is_assigned]   = palette[:, 0]
        green[is_assigned] = palette[:, 1]
        blue[is_assigned]  = palette[:, 2]

    las.red   = (red   * 257).astype(np.uint16)
    las.green = (green * 257).astype(np.uint16)
    las.blue  = (blue  * 257).astype(np.uint16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_path))
    return {"tile": in_path.stem, "n_assigned": int(is_assigned.sum()),
            "n_unique_local": int(len(np.unique(local_ids[is_assigned])) if is_assigned.any() else 0)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    tiles = sorted(args.in_dir.glob("tile_*_*.laz"))
    print(f"rendering {len(tiles)} tiles → {args.out_dir}")
    for i, t in enumerate(tiles):
        out = args.out_dir / t.name
        info = process_tile(t, out)
        print(f"[{i+1}/{len(tiles)}] {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
