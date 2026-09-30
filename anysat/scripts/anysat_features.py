#!/usr/bin/env python
"""Extract AnySat features from co-registered rasters covering one extent.

Inputs are a single-date very-high-resolution GeoTIFF (``naip``, ``aerial``,
``aerial-flair`` or ``spot``) and/or a Sentinel-2 time-series GeoTIFF with a
sidecar list of acquisition dates. Every raster must already sit on the same
extent in the same CRS, at the resolution AnySat assumes for its key; the
extent is cut into square tiles, each tile goes through the model, and the
per-tile outputs are reassembled into grids over the whole extent.

The Sentinel-2 stack is date-major: bands 1-10 are the first date in AnySat's
order (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12), bands 11-20 the second
date, and so on. The dates file holds one ISO date (YYYY-MM-DD) per line.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
import torch

DEFAULT_REPO = Path("/opt/anysat")

# Ground sampling distance, in meters, that AnySat's projector for each key
# assumes (hubconf.py, AnySat.res). An input at any other resolution gives
# the wrong extent check and the wrong patch geometry.
RESOLUTION_M = {"aerial": 0.2, "aerial-flair": 0.2, "spot": 1.0, "naip": 1.25, "s2": 10.0}
N_BANDS = {"aerial": 4, "aerial-flair": 5, "spot": 3, "naip": 4, "s2": 10}
VHR_KEYS = ("aerial", "aerial-flair", "spot", "naip")

# The projectors work on 10 m cells, so patch sizes must be multiples of 10 m
# (upstream README, "Extract Features").
CELL_M = 10


@dataclass(frozen=True)
class Scene:
    """Arrays for one extent, keyed as AnySat expects, plus georeferencing."""

    arrays: dict[str, np.ndarray]  # VHR: (C, H, W); s2: (T, 10, H, W)
    day_of_year: np.ndarray | None  # (T,), for s2
    bounds: tuple[float, float, float, float]
    crs: str


def load_model(repo: Path = DEFAULT_REPO, device: str = "cuda") -> torch.nn.Module:
    """Load pretrained AnySat from a local clone, offline.

    The hub entrypoint fetches weights through torch.hub's checkpoint cache;
    the image points TORCH_HOME at a cache holding the baked checkpoint, so
    no download happens. flash-attn is not installed in the image.
    """
    model = torch.hub.load(str(repo), "anysat", source="local", pretrained=True, flash_attn=False)
    return model.to(device).eval()


def day_of_year(iso_dates: list[str]) -> np.ndarray:
    """Day of year counted from 1 on 1 January.

    Upstream's README says 1 January = 0, but four of the five pretraining
    loaders use ``timetuple().tm_yday`` (1-based); PASTIS-HD's counts days
    from a fixed start date. The loaders are what the weights saw.
    """
    return np.array([date.fromisoformat(d).timetuple().tm_yday for d in iso_dates], dtype=np.int64)


def read_scene(vhr: Path | None, vhr_key: str, s2: Path | None, s2_dates: Path | None) -> Scene:
    """Read the rasters and check they share one extent and CRS."""
    arrays, bounds, crs, doy = {}, [], [], None
    if vhr is not None:
        with rasterio.open(vhr) as src:
            arrays[vhr_key] = src.read().astype(np.float32)
            _check_raster(src, vhr_key, arrays[vhr_key].shape[0])
            bounds.append(tuple(src.bounds))
            crs.append(src.crs.to_string())
    if s2 is not None:
        dates = [d.strip() for d in Path(s2_dates).read_text().splitlines() if d.strip()]
        with rasterio.open(s2) as src:
            flat = src.read().astype(np.float32)
            if flat.shape[0] != len(dates) * N_BANDS["s2"]:
                raise ValueError(f"{s2}: {flat.shape[0]} bands for {len(dates)} dates x 10 bands")
            arrays["s2"] = flat.reshape(len(dates), N_BANDS["s2"], *flat.shape[1:])
            _check_raster(src, "s2", N_BANDS["s2"])
            bounds.append(tuple(src.bounds))
            crs.append(src.crs.to_string())
        doy = day_of_year(dates)
    if not arrays:
        raise ValueError("give a VHR raster, a Sentinel-2 stack, or both")
    if len(set(crs)) > 1 or not np.allclose(bounds, bounds[0], atol=1e-6):
        raise ValueError(f"rasters do not share one extent and CRS: {bounds} {crs}")
    return Scene(arrays, doy, bounds[0], crs[0])


def _check_raster(src: rasterio.DatasetReader, key: str, n_bands: int) -> None:
    if n_bands != N_BANDS[key]:
        raise ValueError(f"{src.name}: {n_bands} bands, AnySat key '{key}' takes {N_BANDS[key]}")
    if not np.allclose(src.res, (RESOLUTION_M[key],) * 2):
        raise ValueError(
            f"{src.name}: resolution {src.res}, '{key}' needs {RESOLUTION_M[key]} m; resample first"
        )


def channel_stats(scene: Scene) -> dict[str, dict[str, list[float]]]:
    """Per-channel mean and standard deviation over the whole extent.

    Upstream normalizes each dataset by statistics computed on that dataset
    (src/data/*.py, compute_norm_vals) and ships none of them, so data from a
    new region is normalized by its own statistics. Compute them over the
    full extent, never per tile, so every tile shares one scaling.
    """
    stats = {}
    for key, a in scene.arrays.items():
        axes = (0, 2, 3) if a.ndim == 4 else (1, 2)
        stats[key] = {"mean": a.mean(axis=axes).tolist(), "std": a.std(axis=axes).tolist()}
    return stats


def normalize(scene: Scene, stats: dict) -> dict[str, np.ndarray]:
    out = {}
    for key, a in scene.arrays.items():
        mean, std = (np.asarray(stats[key][k], dtype=np.float32) for k in ("mean", "std"))
        shape = (1, -1, 1, 1) if a.ndim == 4 else (-1, 1, 1)
        out[key] = (a - mean.reshape(shape)) / std.reshape(shape)
    return out


def extract(
    model: torch.nn.Module,
    scene: Scene,
    stats: dict,
    tile_m: int,
    patch_m: int,
    device: str = "cuda",
    dense_key: str | None = None,
) -> dict[str, np.ndarray]:
    """Run AnySat tile by tile and reassemble the outputs over the extent.

    Returns ``tile`` (ny, nx, D), ``patch`` (ny*n, nx*n, D) with n =
    tile_m / patch_m, and, when ``dense_key`` is given, ``dense`` for that
    modality (rows, cols, 2D).
    """
    if patch_m % CELL_M or tile_m % patch_m:
        raise ValueError("patch_m must be a multiple of 10 m and divide tile_m")
    width_m = scene.bounds[2] - scene.bounds[0]
    height_m = scene.bounds[3] - scene.bounds[1]
    if width_m % tile_m or height_m % tile_m:
        raise ValueError(f"extent {width_m} x {height_m} m is not a whole number of {tile_m} m tiles")
    ny, nx = int(height_m // tile_m), int(width_m // tile_m)
    per_tile = tile_m // patch_m
    data = normalize(scene, stats)
    rows = {"tile": [], "patch": [], "dense": []}
    for i in range(ny):
        cols = {name: [] for name in rows}
        for j in range(nx):
            batch = {key: _cut(a, key, i, j, tile_m, device) for key, a in data.items()}
            if scene.day_of_year is not None:
                batch["s2_dates"] = torch.from_numpy(scene.day_of_year)[None].to(device)
            with torch.inference_mode():
                # 'all' is the class token followed by the patch tokens: the
                # 'tile' and 'patch' outputs of one forward pass.
                tokens = model(batch, patch_size=patch_m, output="all")[0]
                cols["tile"].append(tokens[0][None, None])
                cols["patch"].append(tokens[1:].reshape(per_tile, per_tile, -1))
                if dense_key:
                    cols["dense"].append(
                        model(batch, patch_size=patch_m, output="dense", output_modality=dense_key)[0]
                    )
        for name in rows:
            if cols[name]:
                rows[name].append(torch.cat(cols[name], dim=1))
    return {name: torch.cat(v, dim=0).float().cpu().numpy() for name, v in rows.items() if v}


def _cut(a: np.ndarray, key: str, i: int, j: int, tile_m: int, device: str) -> torch.Tensor:
    """Tile (i, j), counted from the north-west corner, as a batch of one."""
    n = int(round(tile_m / RESOLUTION_M[key]))
    tile = a[..., i * n : (i + 1) * n, j * n : (j + 1) * n]
    return torch.from_numpy(np.ascontiguousarray(tile))[None].to(device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vhr", type=Path, help="single-date VHR GeoTIFF")
    p.add_argument("--vhr-key", choices=VHR_KEYS, default="naip")
    p.add_argument("--s2", type=Path, help="date-major Sentinel-2 stack")
    p.add_argument("--s2-dates", type=Path, help="one ISO date per line, in stack order")
    p.add_argument("--stats", type=Path, help="JSON of per-key mean/std; default: computed from the inputs")
    p.add_argument("--tile-m", type=int, required=True, help="tile edge in meters")
    p.add_argument("--patch-m", type=int, required=True, help="patch edge in meters, multiple of 10")
    p.add_argument("--dense", choices=(*VHR_KEYS, "s2"), help="also return dense features for this key")
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--out", type=Path, required=True, help="output .npz")
    args = p.parse_args()

    scene = read_scene(args.vhr, args.vhr_key, args.s2, args.s2_dates)
    stats = json.loads(args.stats.read_text()) if args.stats else channel_stats(scene)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = extract(
        load_model(args.repo, device), scene, stats, args.tile_m, args.patch_m, device, args.dense
    )
    np.savez(args.out, **feats, bounds=np.array(scene.bounds), crs=scene.crs, stats=json.dumps(stats))
    print({k: v.shape for k, v in feats.items()}, "->", args.out)


if __name__ == "__main__":
    main()
