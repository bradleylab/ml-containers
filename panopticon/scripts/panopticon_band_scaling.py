"""Peak GPU memory and forward time of Panopticon as the band count rises.

Reads 224 x 224 windows from one Tanager-1 ortho surface-reflectance scene
(HDF-EOS5) or one EMIT L2A reflectance granule (netCDF-4), both through h5py,
and runs the baked teacher on nested subsets of the sensor's own bands, each
labeled with its center wavelength from the file.

Band-count tokens: an integer k takes k bands evenly spaced across the bands the
file flags as good; ``good`` takes every good band; ``all`` takes every native
band. Pixels that are fill or non-finite in a selected band are set to 0 after
per-band standardization and counted in ``filled_values``. Windows are chosen
to be complete on every good band, so only ``all`` can fill, and only in bands
the file flags as bad.

Each window is standardized per band with its own valid-pixel mean and standard
deviation. That is enough to time the model and to check its output is finite;
it is not upstream's normalization (training-split statistics per dataset).

Writes one JSON line per (band count, batch size) and never overwrites output.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from panopticon_model import embed, load_teacher

WINDOW = 224
STD_FLOOR = 1e-6
TANAGER_SR = "HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance"
TANAGER_MASKS = tuple(
    f"HDFEOS/GRIDS/HYP/Data Fields/{name}"
    for name in ("nodata_pixels", "beta_cloud_mask", "beta_cirrus_mask")
)
TANAGER_FILL = -9999.0  # Planet L2A surface-reflectance fill (TanagerFM constants.py)
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


@dataclass
class Windows:
    cubes: np.ndarray  # (N, C_native, 224, 224) float32, raw values
    valid: np.ndarray  # same shape, bool
    wavelengths_nm: np.ndarray
    good: np.ndarray  # (C_native,) bool
    offsets: list[tuple[int, int]]


def _grid(shape: tuple[int, int]):
    for row in range(0, shape[0] - WINDOW + 1, WINDOW):
        for col in range(0, shape[1] - WINDOW + 1, WINDOW):
            yield row, col


def read_tanager(path: Path, n_windows: int) -> Windows:
    """Grid windows free of nodata, cloud and cirrus and complete on every good band."""
    with h5py.File(path, "r") as f:
        sr = f[TANAGER_SR]
        wavelengths = np.asarray(sr.attrs["wavelengths"], dtype=np.float64)
        good = np.asarray(sr.attrs["good_wavelengths"]).astype(bool)
        if not sr.shape[0] == wavelengths.size == good.size:
            raise SystemExit(
                f"{path}: {sr.shape[0]} bands, {wavelengths.size} wavelengths, {good.size} flags"
            )
        flagged = np.zeros(sr.shape[1:], dtype=bool)
        for name in TANAGER_MASKS:
            flagged |= np.asarray(f[name][:]) > 0
        cubes, offsets = [], []
        for row, col in _grid(flagged.shape):
            if flagged[row : row + WINDOW, col : col + WINDOW].any():
                continue
            cube = np.asarray(sr[:, row : row + WINDOW, col : col + WINDOW], dtype=np.float32)
            if not _is_valid(cube[good], [TANAGER_FILL]).all():
                continue
            cubes.append(cube)
            offsets.append((row, col))
            if len(cubes) == n_windows:
                break
    stack = _stack(cubes, path)
    return Windows(stack, _is_valid(stack, [TANAGER_FILL]), wavelengths, good, offsets)


def read_emit(path: Path, n_windows: int) -> Windows:
    """Grid windows complete on every good band, bands moved first."""
    with h5py.File(path, "r") as f:
        refl = f["reflectance"]
        wavelengths = np.asarray(f["sensor_band_parameters/wavelengths"][:], dtype=np.float64)
        good = np.asarray(f["sensor_band_parameters/good_wavelengths"][:]).astype(bool)
        if not refl.shape[-1] == wavelengths.size == good.size:
            raise SystemExit(
                f"{path}: {refl.shape[-1]} bands, {wavelengths.size} wavelengths, {good.size} flags"
            )
        fills = [
            float(v)
            for key in ("_FillValue", "missing_value")
            if key in refl.attrs
            for v in np.asarray(refl.attrs[key]).reshape(-1)
        ]
        cubes, offsets = [], []
        for row, col in _grid(refl.shape[:2]):
            cube = np.moveaxis(
                np.asarray(refl[row : row + WINDOW, col : col + WINDOW, :], dtype=np.float32), -1, 0
            )
            if not _is_valid(cube[good], fills).all():
                continue
            cubes.append(cube)
            offsets.append((row, col))
            if len(cubes) == n_windows:
                break
    stack = _stack(cubes, path)
    return Windows(stack, _is_valid(stack, fills), wavelengths, good, offsets)


def _is_valid(values: np.ndarray, fills: list[float]) -> np.ndarray:
    return np.isfinite(values) & ~np.isin(values, fills)


def _stack(cubes: list[np.ndarray], path: Path) -> np.ndarray:
    if not cubes:
        raise SystemExit(f"no complete {WINDOW}x{WINDOW} window in {path}")
    return np.stack(cubes)


def select_bands(token: str, good: np.ndarray) -> np.ndarray:
    good_idx = np.flatnonzero(good)
    if token == "all":
        return np.arange(good.size)
    if token == "good":
        return good_idx
    k = int(token)
    if k > good_idx.size:
        raise SystemExit(f"asked for {k} bands but the file flags only {good_idx.size} as good")
    return good_idx[np.linspace(0, good_idx.size - 1, k).round().astype(int)]


def standardize(cubes: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-window, per-band standardization over valid pixels; invalid pixels become 0."""
    masked = np.where(valid, cubes, np.nan)
    mean = np.nanmean(masked, axis=(2, 3), keepdims=True)
    std = np.maximum(np.nanstd(masked, axis=(2, 3), keepdims=True), STD_FLOOR)
    out = np.where(valid, (cubes - np.nan_to_num(mean)) / np.nan_to_num(std, nan=1.0), 0.0)
    return out.astype(np.float32), int((~valid).sum())


def time_forward(model, x: torch.Tensor, ids: list[float], repeats: int, dtype) -> dict:
    device = x.device
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        embed(model, x, ids)  # warm-up: kernel selection and allocator growth
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            out = embed(model, x, ids)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)
    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "finite": bool(torch.isfinite(out["cls"]).all() and torch.isfinite(out["patch"]).all()),
        "cls_shape": list(out["cls"].shape),
    }


def run_case(model, windows: Windows, token: str, batch: int, args) -> dict:
    bands = select_bands(token, windows.good)
    picks = [i % len(windows.offsets) for i in range(batch)]
    cubes, filled = standardize(windows.cubes[picks][:, bands], windows.valid[picks][:, bands])
    ids = [float(v) for v in windows.wavelengths_nm[bands]]
    record = {
        "sensor": args.sensor,
        "file": args.path.name,
        "bands": token,
        "n_bands": int(bands.size),
        "batch": batch,
        "distinct_windows": len(set(picks)),
        "filled_values": filled,
        "wavelength_min_nm": min(ids),
        "wavelength_max_nm": max(ids),
        "dtype": args.dtype,
        "input_bytes": cubes.nbytes,
    }
    x = torch.from_numpy(cubes).cuda()
    try:
        record.update(status="ok", **time_forward(model, x, ids, args.repeats, DTYPES[args.dtype]))
    except torch.cuda.OutOfMemoryError:
        record["status"] = "oom"
    del x
    torch.cuda.empty_cache()
    return record


READERS = {"tanager": read_tanager, "emit": read_emit}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor", choices=sorted(READERS), required=True)
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--bands", nargs="+", default=["12", "50", "100", "200", "good", "all"])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new path")
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")

    windows = READERS[args.sensor](args.path, max(args.batch_sizes))
    for token in args.bands:
        select_bands(token, windows.good)  # fail on an impossible band count before any GPU work
    print(
        f"{args.path.name}: {windows.wavelengths_nm.size} native bands, {int(windows.good.sum())} good, "
        f"{len(windows.offsets)} windows at {windows.offsets}",
        flush=True,
    )
    model = load_teacher("cuda")
    weights_bytes = torch.cuda.memory_allocated()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as sink:
        for token in args.bands:
            for batch in args.batch_sizes:
                record = run_case(model, windows, token, batch, args)
                record["model_weights_bytes"] = weights_bytes
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
