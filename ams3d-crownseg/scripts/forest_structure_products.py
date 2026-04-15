#!/usr/bin/env python3
"""Generate forest structure products from AMS3D tree segmentation output.

Products:
  1. Per-tree attribute table (CSV + GeoPackage)
  2. Stem density map (GeoTIFF, trees/ha at configurable grid resolution)
  3. Gap fraction map (GeoTIFF, fraction of cells with no canopy above threshold)
  4. Vertical structure profiles (GeoTIFF stack, point density per height bin)
  5. Summary statistics (JSON)

Inputs: AMS3D-segmented LAZ with treeID attribute + original classified LAZ.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import laspy
import numpy as np

# Avoid import until needed for optional GeoTIFF writing
RASTERIO_AVAILABLE = False
try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    pass

GEOPANDAS_AVAILABLE = False
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    pass


# --- Constants ---
GROUND_CLASS = 2
MIN_TREE_POINTS = 500       # ignore micro-clusters
GROUND_CATCHALL_ID = 2147483647  # crownsegmentr ground label
DBH_ALLOMETRY_A = 0.6       # DBH (cm) = a * height^b (generic temperate hardwood)
DBH_ALLOMETRY_B = 1.1       # rough approximation — not species-specific


def normalize_heights(las: laspy.LasData) -> np.ndarray:
    """Compute height above ground via TIN of ground returns."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay, cKDTree

    classif = np.asarray(las.classification, dtype=np.int64)
    is_ground = classif == GROUND_CLASS
    gx, gy, gz = np.asarray(las.x)[is_ground], np.asarray(las.y)[is_ground], np.asarray(las.z)[is_ground]

    tri = Delaunay(np.column_stack([gx, gy]))
    interp = LinearNDInterpolator(tri, gz)
    ground_z = interp(np.asarray(las.x), np.asarray(las.y))

    nan_mask = np.isnan(ground_z)
    if nan_mask.any():
        gtree = cKDTree(np.column_stack([gx, gy]))
        _, idx = gtree.query(np.column_stack([np.asarray(las.x)[nan_mask],
                                               np.asarray(las.y)[nan_mask]]))
        ground_z[nan_mask] = gz[idx]

    return np.asarray(las.z) - ground_z


def compute_tree_attributes(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                             height: np.ndarray, tree_ids: np.ndarray) -> list[dict]:
    """Compute per-tree attributes."""
    unique_ids = np.unique(tree_ids)
    unique_ids = unique_ids[(unique_ids > 0) & (unique_ids != GROUND_CATCHALL_ID)]

    trees = []
    for tid in unique_ids:
        mask = tree_ids == tid
        n_pts = int(mask.sum())
        if n_pts < MIN_TREE_POINTS:
            continue

        tx, ty, tz, th = x[mask], y[mask], z[mask], height[mask]

        tree_height = float(np.max(th))
        crown_base_height = float(np.percentile(th, 5))
        crown_length = tree_height - crown_base_height

        # Crown centroid (density-weighted center)
        cx, cy = float(np.mean(tx)), float(np.mean(ty))

        # Crown area via 2D convex hull
        try:
            from scipy.spatial import ConvexHull
            pts_2d = np.column_stack([tx, ty])
            if len(pts_2d) >= 3:
                hull = ConvexHull(pts_2d)
                crown_area = float(hull.volume)  # 2D hull "volume" = area
                crown_perimeter = float(hull.area)  # 2D hull "area" = perimeter
            else:
                crown_area = 0.0
                crown_perimeter = 0.0
        except Exception:
            crown_area = 0.0
            crown_perimeter = 0.0

        crown_diameter = 2 * np.sqrt(crown_area / np.pi) if crown_area > 0 else 0.0

        # Crown volume via 3D convex hull
        try:
            pts_3d = np.column_stack([tx, ty, tz])
            if len(pts_3d) >= 4:
                hull_3d = ConvexHull(pts_3d)
                crown_volume = float(hull_3d.volume)
            else:
                crown_volume = 0.0
        except Exception:
            crown_volume = 0.0

        # Bounding box
        bbox_w = float(np.max(tx) - np.min(tx))
        bbox_h = float(np.max(ty) - np.min(ty))

        # Estimated DBH via generic allometry
        est_dbh_cm = DBH_ALLOMETRY_A * (tree_height ** DBH_ALLOMETRY_B)

        trees.append({
            "tree_id": int(tid),
            "x": round(cx, 2),
            "y": round(cy, 2),
            "height_m": round(tree_height, 2),
            "crown_base_height_m": round(crown_base_height, 2),
            "crown_length_m": round(crown_length, 2),
            "crown_area_m2": round(crown_area, 2),
            "crown_diameter_m": round(crown_diameter, 2),
            "crown_perimeter_m": round(crown_perimeter, 2),
            "crown_volume_m3": round(crown_volume, 2),
            "bbox_width_m": round(bbox_w, 2),
            "bbox_height_m": round(bbox_h, 2),
            "n_points": n_pts,
            "est_dbh_cm": round(est_dbh_cm, 1),
        })

    return trees


def make_raster_grid(x: np.ndarray, y: np.ndarray, res: float):
    """Create a regular grid covering the point extent."""
    xmin, xmax = np.floor(x.min()), np.ceil(x.max())
    ymin, ymax = np.floor(y.min()), np.ceil(y.max())
    cols = int(np.ceil((xmax - xmin) / res))
    rows = int(np.ceil((ymax - ymin) / res))
    return xmin, ymin, xmax, ymax, rows, cols


def compute_stem_density(trees: list[dict], x_all: np.ndarray, y_all: np.ndarray,
                          grid_res: float) -> tuple[np.ndarray, tuple]:
    """Stem density map (trees per hectare)."""
    xmin, ymin, xmax, ymax, rows, cols = make_raster_grid(x_all, y_all, grid_res)
    grid = np.zeros((rows, cols), dtype=np.float32)

    ha_per_cell = (grid_res ** 2) / 10000.0

    for t in trees:
        c = int((t["x"] - xmin) / grid_res)
        r = int((ymax - t["y"]) / grid_res)  # y-axis flipped for raster
        r = min(r, rows - 1)
        c = min(c, cols - 1)
        grid[r, c] += 1

    # Convert count to trees/ha
    density = grid / ha_per_cell

    return density, (xmin, ymin, xmax, ymax, rows, cols)


def compute_gap_fraction(height: np.ndarray, x: np.ndarray, y: np.ndarray,
                          grid_res: float, canopy_threshold: float = 5.0
                          ) -> tuple[np.ndarray, tuple]:
    """Gap fraction: fraction of ground-level area with no canopy above threshold."""
    xmin, ymin, xmax, ymax, rows, cols = make_raster_grid(x, y, grid_res)
    total_pts = np.zeros((rows, cols), dtype=np.float32)
    canopy_pts = np.zeros((rows, cols), dtype=np.float32)

    ci = np.clip(((x - xmin) / grid_res).astype(int), 0, cols - 1)
    ri = np.clip(((ymax - y) / grid_res).astype(int), 0, rows - 1)

    for i in range(len(x)):
        total_pts[ri[i], ci[i]] += 1
        if height[i] >= canopy_threshold:
            canopy_pts[ri[i], ci[i]] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        canopy_frac = np.where(total_pts > 0, canopy_pts / total_pts, 0)
    gap_fraction = 1.0 - canopy_frac

    return gap_fraction.astype(np.float32), (xmin, ymin, xmax, ymax, rows, cols)


def compute_vertical_profiles(height: np.ndarray, x: np.ndarray, y: np.ndarray,
                               grid_res: float, height_bins: list[float]
                               ) -> tuple[np.ndarray, tuple, list[str]]:
    """Vertical structure: point density per height bin per grid cell."""
    xmin, ymin, xmax, ymax, rows, cols = make_raster_grid(x, y, grid_res)
    n_bins = len(height_bins) - 1
    stack = np.zeros((n_bins, rows, cols), dtype=np.float32)

    ci = np.clip(((x - xmin) / grid_res).astype(int), 0, cols - 1)
    ri = np.clip(((ymax - y) / grid_res).astype(int), 0, rows - 1)
    bin_idx = np.digitize(height, height_bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    for i in range(len(x)):
        stack[bin_idx[i], ri[i], ci[i]] += 1

    # Normalize each cell to fraction of total points in that cell
    cell_totals = stack.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        stack_frac = np.where(cell_totals > 0, stack / cell_totals, 0).astype(np.float32)

    band_names = []
    for i in range(n_bins):
        band_names.append(f"{height_bins[i]:.0f}-{height_bins[i+1]:.0f}m")

    return stack_frac, (xmin, ymin, xmax, ymax, rows, cols), band_names


def write_geotiff(data: np.ndarray, bounds: tuple, path: Path, crs: str = "EPSG:32615",
                  band_names: list[str] | None = None) -> None:
    """Write a 2D or 3D array as a GeoTIFF."""
    if not RASTERIO_AVAILABLE:
        print(f"  [skip] rasterio not available, cannot write {path}")
        return

    xmin, ymin, xmax, ymax, rows, cols = bounds
    transform = from_bounds(xmin, ymin, xmax, ymax, cols, rows)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    n_bands = data.shape[0]
    with rasterio.open(path, 'w', driver='GTiff', height=rows, width=cols,
                       count=n_bands, dtype=data.dtype, crs=crs,
                       transform=transform, compress='deflate') as dst:
        for b in range(n_bands):
            dst.write(data[b], b + 1)
            if band_names and b < len(band_names):
                dst.set_band_description(b + 1, band_names[b])

    print(f"  wrote {path} ({n_bands} band{'s' if n_bands > 1 else ''}, {rows}×{cols})")


def process_tile(in_path: Path, out_dir: Path, grid_res: float,
                 height_bins: list[float], crs: str) -> dict:
    """Process one segmented LAZ tile into forest structure products."""
    t0 = time.perf_counter()
    tile_name = in_path.stem
    tile_dir = out_dir / tile_name
    tile_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {tile_name} ===")
    las = laspy.read(str(in_path))
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)

    # Get tree IDs
    if "treeID" in las.point_format.dimension_names:
        tree_ids = np.asarray(las.treeID)
        if np.issubdtype(tree_ids.dtype, np.floating):
            tree_ids = np.where(np.isnan(tree_ids), 0, tree_ids).astype(np.int64)
        else:
            tree_ids = tree_ids.astype(np.int64)
    elif "PredInstance" in las.point_format.dimension_names:
        tree_ids = np.asarray(las.PredInstance, dtype=np.int64)
    else:
        print("  ERROR: no tree ID field found")
        return {"tile": tile_name, "error": "no tree ID field"}

    # Height normalization
    print("  normalizing heights...")
    height = normalize_heights(las)

    # 1. Per-tree attributes
    print("  computing tree attributes...")
    trees = compute_tree_attributes(x, y, z, height, tree_ids)
    print(f"  {len(trees)} trees (>={MIN_TREE_POINTS} pts)")

    # Write CSV
    import csv
    csv_path = tile_dir / "tree_attributes.csv"
    if trees:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trees[0].keys())
            writer.writeheader()
            writer.writerows(trees)
        print(f"  wrote {csv_path}")

    # Write GeoPackage if geopandas available
    if GEOPANDAS_AVAILABLE and trees:
        gpkg_path = tile_dir / "tree_attributes.gpkg"
        gdf = gpd.GeoDataFrame(
            trees,
            geometry=[Point(t["x"], t["y"]) for t in trees],
            crs=crs
        )
        gdf.to_file(gpkg_path, driver="GPKG")
        print(f"  wrote {gpkg_path}")

    # 2. Stem density map
    print("  computing stem density...")
    density, bounds = compute_stem_density(trees, x, y, grid_res)
    write_geotiff(density, bounds, tile_dir / "stem_density_per_ha.tif", crs=crs)

    # 3. Gap fraction map
    print("  computing gap fraction...")
    gap_frac, bounds_gap = compute_gap_fraction(height, x, y, grid_res)
    write_geotiff(gap_frac, bounds_gap, tile_dir / "gap_fraction.tif", crs=crs)

    # 4. Vertical structure profiles
    print("  computing vertical profiles...")
    vstack, bounds_v, band_names = compute_vertical_profiles(height, x, y, grid_res, height_bins)
    write_geotiff(vstack, bounds_v, tile_dir / "vertical_structure.tif",
                  crs=crs, band_names=band_names)

    # 5. Summary statistics
    if trees:
        heights = [t["height_m"] for t in trees]
        areas = [t["crown_area_m2"] for t in trees]
        dbhs = [t["est_dbh_cm"] for t in trees]
        tile_area_ha = ((x.max() - x.min()) * (y.max() - y.min())) / 10000

        summary = {
            "tile": tile_name,
            "tile_area_ha": round(tile_area_ha, 3),
            "n_trees": len(trees),
            "stems_per_ha": round(len(trees) / tile_area_ha, 1),
            "mean_height_m": round(np.mean(heights), 2),
            "median_height_m": round(np.median(heights), 2),
            "max_height_m": round(np.max(heights), 2),
            "mean_crown_area_m2": round(np.mean(areas), 2),
            "median_crown_area_m2": round(np.median(areas), 2),
            "mean_est_dbh_cm": round(np.mean(dbhs), 1),
            "mean_gap_fraction": round(float(np.mean(gap_frac)), 4),
            "processing_time_s": round(time.perf_counter() - t0, 1),
        }
    else:
        summary = {"tile": tile_name, "n_trees": 0}

    summary_path = tile_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {summary_path}")
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, nargs="+",
                   help="Segmented LAZ file(s) with treeID attribute")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--grid-res", type=float, default=20.0,
                   help="Grid resolution for raster products (m) [default 20]")
    p.add_argument("--height-bins", type=str, default="0,5,10,15,20,25,35",
                   help="Comma-separated height bin edges (m) [default 0,5,10,15,20,25,35]")
    p.add_argument("--crs", type=str, default="EPSG:32615",
                   help="Coordinate reference system [default EPSG:32615]")
    args = p.parse_args()

    height_bins = [float(x) for x in args.height_bins.split(",")]

    summaries = []
    for f in args.input:
        if not f.exists():
            print(f"MISSING: {f}", file=sys.stderr)
            continue
        s = process_tile(f, args.out_dir, args.grid_res, height_bins, args.crs)
        summaries.append(s)

    # Write combined summary
    combined_path = args.out_dir / "all_summaries.json"
    combined_path.write_text(json.dumps(summaries, indent=2))
    print(f"\n=== All summaries → {combined_path} ===")
    for s in summaries:
        print(json.dumps(s))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
