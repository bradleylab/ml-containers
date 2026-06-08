"""
Extract CROMA embeddings from a Sentinel-1 / Sentinel-2 chip.

Loads CROMA (Fuller, Millard & Green, NeurIPS 2023) via the upstream
`use_croma.PretrainedCROMA` module and runs a frozen forward pass over
a Sentinel-1 SAR chip (2 channels: VV, VH), a Sentinel-2 optical chip
(12 bands, cirrus B10 removed), or both. Writes the global-average-
pooled (GAP) embedding vector(s) — 768-D for Base, 1024-D for Large —
and, optionally, the per-patch token encodings.

CROMA is Sentinel-1/Sentinel-2-native: SAR is exactly 2 channels and
optical is exactly 12 bands. The spatial resolution must be a multiple
of 8 (the patch stride); the pretrained models use 120x120.

Inputs:
    --sar         : .npy or .pt file, (2, H, W) float tensor (VV, VH).
    --optical     : .npy or .pt file, (12, H, W) float tensor
                    (S2 L1C without B10/cirrus).
    --modality    : both | SAR | optical (default: inferred from which
                    of --sar / --optical are supplied).
    --variant     : base | large (default base).
    --image-resolution : square size fed to CROMA; multiple of 8
                    (default 120, the pretrained resolution). Inputs are
                    bilinearly resized to this if they differ.
    --save-encodings : also store the (1, N, D) per-patch tokens, not
                    just the pooled GAP vector(s).
    --out         : output .npz path.

At least one of --sar / --optical is required; modality 'both' needs
both. Examples:

    # Joint S1+S2 embedding
    python /opt/scripts/croma_embed.py \\
      --sar /work/s1.npy --optical /work/s2_12band.npy \\
      --variant base --out /work/embed.npz

    # SAR-only embedding
    python /opt/scripts/croma_embed.py \\
      --sar /work/s1.npy --modality SAR --out /work/s1_embed.npz

Reference:
    Fuller, A., Millard, K. & Green, J. R. (2023) "CROMA: Remote
    Sensing Representations with Contrastive Radar-Optical Masked
    Autoencoders", NeurIPS 2023. arXiv:2311.00566.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# Channel counts CROMA's encoders expect. These are fixed by the
# pretrained architecture, not tunable.
SAR_CHANNELS = 2
OPTICAL_CHANNELS = 12
WEIGHT_FILE = {"base": "CROMA_base.pt", "large": "CROMA_large.pt"}


def load_chw(path: Path) -> torch.Tensor:
    """Load a (C, H, W) float32 tensor from .npy or .pt."""
    if path.suffix == ".npy":
        t = torch.from_numpy(np.load(path)).to(torch.float32)
    elif path.suffix in (".pt", ".pth"):
        t = torch.load(path, weights_only=True)
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{path} is not a tensor (got {type(t)})")
        t = t.to(torch.float32)
    else:
        raise ValueError(f"unsupported extension {path.suffix}; use .npy / .pt")
    if t.ndim != 3:
        raise ValueError(f"expected (C, H, W), got shape {tuple(t.shape)}")
    return t


def prep(t: torch.Tensor, expected_c: int, res: int, name: str) -> torch.Tensor:
    """Validate channel count and resize to (1, C, res, res)."""
    c, h, w = t.shape
    if c != expected_c:
        raise SystemExit(
            f"ERROR: --{name} has {c} channels but CROMA expects {expected_c} "
            f"({'VV,VH' if name == 'sar' else 'S2 12-band, cirrus removed'})"
        )
    if (h, w) != (res, res):
        print(f"resizing {name} from ({h}, {w}) to ({res}, {res}) bilinear",
              file=sys.stderr)
        t = F.interpolate(t.unsqueeze(0), size=(res, res),
                          mode="bilinear", align_corners=False).squeeze(0)
    return t.unsqueeze(0)  # (1, C, res, res)


def resolve_modality(args) -> str:
    if args.modality:
        return args.modality
    if args.sar and args.optical:
        return "both"
    if args.sar:
        return "SAR"
    if args.optical:
        return "optical"
    raise SystemExit("ERROR: pass at least one of --sar / --optical")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sar", type=Path, default=None)
    ap.add_argument("--optical", type=Path, default=None)
    ap.add_argument("--modality", default=None, choices=["both", "SAR", "optical"])
    ap.add_argument("--variant", default="base", choices=["base", "large"])
    ap.add_argument("--image-resolution", type=int, default=120,
                    help="square size fed to CROMA; must be a multiple of 8")
    ap.add_argument("--save-encodings", action="store_true",
                    help="also store per-patch token encodings, not just GAP")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    res = args.image_resolution
    if res % 8 != 0:
        raise SystemExit(f"ERROR: --image-resolution must be a multiple of 8, got {res}")

    modality = resolve_modality(args)
    if modality in ("both", "SAR") and not args.sar:
        raise SystemExit(f"ERROR: modality '{modality}' needs --sar")
    if modality in ("both", "optical") and not args.optical:
        raise SystemExit(f"ERROR: modality '{modality}' needs --optical")

    sar = prep(load_chw(args.sar), SAR_CHANNELS, res, "sar") if args.sar else None
    opt = (prep(load_chw(args.optical), OPTICAL_CHANNELS, res, "optical")
           if args.optical else None)

    print(f"loading CROMA-{args.variant} (modality={modality}, res={res})...",
          file=sys.stderr)
    weights = hf_hub_download("antofuller/CROMA", WEIGHT_FILE[args.variant])
    # Provided in-container via PYTHONPATH=/opt/croma (pinned upstream module,
    # not a pip package); unresolved by local linters, present at runtime.
    from use_croma import PretrainedCROMA  # type: ignore[import-not-found]
    model = PretrainedCROMA(pretrained_path=weights, size=args.variant,
                            modality=modality, image_resolution=res).eval()

    with torch.no_grad():
        out = model(SAR_images=sar, optical_images=opt)

    payload: dict[str, np.ndarray] = {}
    for key, tensor in out.items():
        is_gap = key.endswith("_GAP")
        if is_gap or args.save_encodings:
            payload[key] = tensor.detach().cpu().numpy().astype(np.float32)
    # Plain (non-object) arrays so the .npz needs no pickle to round-trip.
    payload["variant"] = np.asarray(args.variant)
    payload["modality"] = np.asarray(modality)
    payload["image_resolution"] = np.asarray(res, dtype=np.int32)

    for key in out:
        if key.endswith("_GAP"):
            print(f"{key}: {tuple(out[key].shape)}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)  # type: ignore[arg-type]
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
