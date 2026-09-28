"""Embed one multi-band raster patch with Panopticon.

Every band needs a channel id: its center wavelength in nm for optical data,
or a SAR category id from -1 to -12. The script refuses to run when the number
of ids differs from the number of bands.

Examples
--------
    python /opt/scripts/panopticon_embed.py --image /work/patch.tif \\
        --channel-ids 664 559 493 --normalize patch --out /work/emb.npz

    python /opt/scripts/panopticon_embed.py --image /work/cube.npy \\
        --channel-ids-file /work/wavelengths_nm.txt \\
        --normalize stats --band-stats /work/band_stats.json --out /work/emb.npz

``--normalize patch`` standardizes each band with the patch's own mean and
standard deviation. Upstream standardized each pretraining dataset with the
mean and standard deviation of its training split, so for a corpus, compute
those statistics once and pass them with ``--normalize stats``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from panopticon_model import ChannelMismatchError, embed, load_teacher, validate_channel_ids

STD_FLOOR = 1e-6


def read_image(path: Path) -> np.ndarray:
    """(C, H, W) float32 from .npy or a GeoTIFF."""
    if path.suffix == ".npy":
        image = np.load(path)
    elif path.suffix.lower() in (".tif", ".tiff"):
        with rasterio.open(path) as src:
            image = src.read()
            nodata = src.nodata
        # Fill pixels would enter the band statistics and the model as data.
        if nodata is not None and np.any(image == nodata):
            raise ValueError(f"{path} contains nodata pixels ({nodata}); crop to a complete patch")
    else:
        raise ValueError(f"unsupported extension {path.suffix}; use .npy, .tif or .tiff")
    if image.ndim != 3:
        raise ValueError(f"expected a (C, H, W) array, got shape {image.shape}")
    return image.astype(np.float32, copy=False)


def read_channel_ids(args: argparse.Namespace) -> list[float]:
    if args.channel_ids_file is None:
        return [float(v) for v in args.channel_ids]
    lines = args.channel_ids_file.read_text().split()
    return [float(v) for v in lines]


def standardize(image: np.ndarray, mode: str, stats_path: Path | None) -> np.ndarray:
    if mode == "none":
        return image
    if mode == "patch":
        mean = image.mean(axis=(1, 2))
        std = image.std(axis=(1, 2))
    else:
        stats = json.loads(stats_path.read_text())
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        if mean.shape != (image.shape[0],) or std.shape != (image.shape[0],):
            raise ChannelMismatchError(
                f"band stats have {mean.size} means and {std.size} stds for {image.shape[0]} bands"
            )
    # A constant band has zero spread; flooring keeps it at zero instead of NaN.
    return (image - mean[:, None, None]) / np.maximum(std, STD_FLOOR)[:, None, None]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True)
    ids = ap.add_mutually_exclusive_group(required=True)
    ids.add_argument("--channel-ids", type=float, nargs="+", help="one id per band, in band order")
    ids.add_argument("--channel-ids-file", type=Path, help="whitespace-separated ids, one per band")
    ap.add_argument("--normalize", choices=["patch", "stats", "none"], required=True)
    ap.add_argument("--band-stats", type=Path, help='JSON {"mean": [...], "std": [...]}')
    ap.add_argument("--resize", type=int, help="bilinear resize to N x N (N a multiple of 14)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.normalize == "stats" and args.band_stats is None:
        ap.error("--normalize stats needs --band-stats")
    return args


def main() -> int:
    args = parse_args()
    image = read_image(args.image)
    channel_ids = read_channel_ids(args)
    try:
        validate_channel_ids(image.shape[0], channel_ids)
        image = standardize(image, args.normalize, args.band_stats)
    except ChannelMismatchError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    if not np.isfinite(image).all():
        print("ERROR: the image contains NaN or infinite values after normalization", file=sys.stderr)
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).to(device)
    if args.resize:
        x = F.interpolate(x, size=(args.resize, args.resize), mode="bilinear", align_corners=False)
    model = load_teacher(device)
    out = embed(model, x, channel_ids)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cls=out["cls"].cpu().numpy(),
        patch=out["patch"].cpu().numpy(),
        channel_ids=np.asarray(channel_ids, dtype=np.float32),
        normalize=np.asarray(args.normalize),
        input_shape=np.asarray(x.shape),
    )
    print(f"cls {tuple(out['cls'].shape)} patch {tuple(out['patch'].shape)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
