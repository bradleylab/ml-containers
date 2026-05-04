"""
Extract DOFA embeddings from a multispectral image.

Loads `dofa_base_patch16_224` (or `_large_*`) via torchgeo, applies
the upstream's wavelength-conditioning hypernetwork over the user-
supplied band wavelengths, and writes a single embedding vector per
input image (768-D for Base, 1024-D for Large).

Inputs:
    --image           : .npy or .pt file containing a (C, H, W) tensor,
                        any C, dtype float32. Will be resized to 224x224
                        and unsqueezed to (1, C, H, W).
    --wavelengths     : space-separated list of band wavelengths in
                        MICROMETERS, length must equal C. Required.
                        Convenience flags: --sentinel2-12band,
                        --sentinel2-10band, --sentinel1, --naip-rgb.
    --variant         : base | large (default base)
    --out             : output .npz path (writes 'embedding' key)

Example (Sentinel-2 L1C 12-band, common L1C order):
    python /opt/scripts/dofa_embed.py \\
      --image /work/s2.npy \\
      --sentinel2-12band \\
      --variant base \\
      --out /work/embed.npz

Reference:
    Xiong, Z. et al. (2024) "Neural Plasticity-Inspired Foundation
    Model for Observing the Earth Crossing Modalities" (DOFA),
    arXiv:2403.15356.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# Sentinel-2 L1C nominal central wavelengths (micrometers). 12-band
# excludes B10 (cirrus) which the L2A products drop.
S2_12BAND_UM = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740,
                0.783, 0.842, 0.865, 0.945, 1.610, 2.190]
# 10-band drops B1 (coastal aerosol) + B9 (water vapor) — the common
# "land bands only" subset.
S2_10BAND_UM = [0.490, 0.560, 0.665, 0.705, 0.740,
                0.783, 0.842, 0.865, 1.610, 2.190]
# Sentinel-1 C-band SAR — VV and VH share the carrier wavelength.
S1_VV_VH_UM = [5.405, 5.405]
# NAIP / typical aerial RGB.
NAIP_RGB_UM = [0.665, 0.560, 0.490]


def load_image(path: Path) -> torch.Tensor:
    """Load a (C, H, W) tensor from .npy or .pt."""
    if path.suffix == ".npy":
        arr = np.load(path)
        t = torch.from_numpy(arr).to(torch.float32)
    elif path.suffix in (".pt", ".pth"):
        t = torch.load(path, weights_only=True)
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{path} is not a tensor (got {type(t)})")
        t = t.to(torch.float32)
    else:
        raise ValueError(
            f"unsupported extension {path.suffix}; use .npy / .pt"
        )
    if t.ndim != 3:
        raise ValueError(f"expected (C, H, W) tensor, got shape {tuple(t.shape)}")
    return t


def resolve_wavelengths(args) -> list[float]:
    if args.sentinel2_12band:
        return S2_12BAND_UM
    if args.sentinel2_10band:
        return S2_10BAND_UM
    if args.sentinel1:
        return S1_VV_VH_UM
    if args.naip_rgb:
        return NAIP_RGB_UM
    if args.wavelengths:
        return [float(w) for w in args.wavelengths]
    raise SystemExit(
        "ERROR: must pass --wavelengths or one of "
        "--sentinel2-{12,10}band / --sentinel1 / --naip-rgb"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", default="base", choices=["base", "large"])
    ap.add_argument("--wavelengths", nargs="*", default=None,
                    help="Space-separated wavelengths (micrometers). "
                         "Length must equal image channel count.")
    ap.add_argument("--sentinel2-12band", action="store_true")
    ap.add_argument("--sentinel2-10band", action="store_true")
    ap.add_argument("--sentinel1", action="store_true")
    ap.add_argument("--naip-rgb", action="store_true")
    args = ap.parse_args()

    waves = resolve_wavelengths(args)
    print(f"wavelengths: {waves} (n={len(waves)})", file=sys.stderr)

    img = load_image(args.image)
    c, h, w = img.shape
    if c != len(waves):
        print(f"ERROR: image has {c} bands but {len(waves)} wavelengths "
              "supplied", file=sys.stderr)
        return 2
    if h != 224 or w != 224:
        print(f"resizing from ({h}, {w}) to (224, 224) bilinear",
              file=sys.stderr)
        img = F.interpolate(img.unsqueeze(0), size=(224, 224),
                            mode="bilinear", align_corners=False).squeeze(0)
    x = img.unsqueeze(0)  # (1, C, 224, 224)

    print(f"loading DOFA-{args.variant}...", file=sys.stderr)
    if args.variant == "base":
        from torchgeo.models import dofa_base_patch16_224, DOFABase16_Weights
        model = dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE)
    else:
        from torchgeo.models import dofa_large_patch16_224, DOFALarge16_Weights
        model = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE)
    model.eval()

    with torch.no_grad():
        emb = model.forward_features(x, wavelengths=waves)
    emb_np = emb.detach().cpu().numpy().astype(np.float32)
    print(f"embedding shape: {tuple(emb_np.shape)}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        embedding=emb_np,
        wavelengths=np.asarray(waves, dtype=np.float32),
        variant=np.asarray(args.variant, dtype=object),
    )
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
