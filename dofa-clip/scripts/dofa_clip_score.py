"""
Score an image against text prompts using DOFA-CLIP via the upstream
open_clip path (Path B). Supports multispectral input via the
wavelength-conditioning hypernetwork that the original DOFA backbone
was trained with — the headline capability that distinguishes
DOFA-CLIP from `bradleylab/remoteclip`.

Default model: `earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO`
  (image trunk: ViT-L/14 with wavelength conditioning;
   text encoder: SigLIP-style so400m).

License note (CRITICAL): the so400m-384-EO weights are CC-BY-NC-4.0
per the HF model card. Use of this image is restricted to
non-commercial purposes; commercial use requires explicit upstream
permission (xiongzhitong@gmail.com). The container code itself is
Apache-2.0 (this repo) + Apache-2.0 (xiong-zhitong/DOFA-CLIP).

Inputs:
    --image           : path to RGB or multispectral image. RGB
                        formats: any Pillow-readable file. Multispectral:
                        .npy or .pt with a (C, H, W) tensor.
    --prompt          : text prompt; pass multiple times for a panel
    --prompts-file    : alternative — one prompt per line
    --wavelengths     : space-separated band wavelengths in micrometers,
                        length must equal C. Convenience flags:
                        --rgb (default; 3 bands at 0.665, 0.560, 0.490)
                        --sentinel2-12band, --sentinel2-10band,
                        --sentinel1.

Output: CSV with columns prompt, cosine, sigmoid_prob.

Cosine is the raw L2-normalised dot product (use absolute thresholds
for screening). sigmoid_prob is `sigmoid(cos * logit_scale + logit_bias)`
— the SigLIP-style score, which is per-prompt independent (does not
sum to 1 across the panel).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_MODEL = "hf-hub:earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO"

# Wavelength conventions in micrometers, matching the upstream README.
RGB_UM = [0.665, 0.560, 0.490]
S2_12BAND_UM = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740,
                0.783, 0.842, 0.865, 0.945, 1.610, 2.190]
S2_10BAND_UM = [0.490, 0.560, 0.665, 0.705, 0.740,
                0.783, 0.842, 0.865, 1.610, 2.190]
S1_VV_VH_UM = [5.405, 5.405]


def load_prompts(args) -> list[str]:
    if args.prompts_file:
        return [p.strip() for p in args.prompts_file.read_text().splitlines()
                if p.strip() and not p.startswith("#")]
    if args.prompt:
        return list(args.prompt)
    raise SystemExit("ERROR: pass --prompt or --prompts-file")


def resolve_wavelengths(args) -> list[float]:
    if args.sentinel2_12band:
        return S2_12BAND_UM
    if args.sentinel2_10band:
        return S2_10BAND_UM
    if args.sentinel1:
        return S1_VV_VH_UM
    if args.rgb:
        return RGB_UM
    if args.wavelengths:
        return [float(w) for w in args.wavelengths]
    return RGB_UM  # safe default


def load_image_tensor(path: Path, preprocess, n_bands_required: int) -> torch.Tensor:
    """Return a (1, C, H, W) tensor preprocessed for the model.

    For RGB inputs (n_bands_required=3) we use the open_clip
    preprocess pipeline (resize, normalise, etc.). For multispectral
    inputs we accept a .npy or .pt tensor and apply only resize +
    Tensor conversion — the open_clip preprocess won't accept a
    multispectral PIL image."""
    if path.suffix in (".npy",):
        arr = np.load(path)
        t = torch.from_numpy(arr).to(torch.float32)
        if t.ndim != 3:
            raise ValueError(f"expected (C, H, W), got shape {tuple(t.shape)}")
        if t.shape[-1] != t.shape[-2]:
            print(f"WARN: non-square spatial dims {t.shape[-2:]}",
                  file=sys.stderr)
        return F.interpolate(t.unsqueeze(0), size=(384, 384),
                             mode="bilinear", align_corners=False)
    if path.suffix in (".pt", ".pth"):
        t = torch.load(path, weights_only=True).to(torch.float32)
        return F.interpolate(t.unsqueeze(0), size=(384, 384),
                             mode="bilinear", align_corners=False)
    # RGB path via open_clip preprocess (returns (C, H, W) tensor)
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--prompt", action="append")
    ap.add_argument("--prompts-file", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL,
                    help="open_clip model id; default is the so400m-384-EO HF Hub mirror")
    ap.add_argument("--wavelengths", nargs="*", default=None,
                    help="Per-band wavelengths in micrometers")
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--sentinel2-12band", action="store_true")
    ap.add_argument("--sentinel2-10band", action="store_true")
    ap.add_argument("--sentinel1", action="store_true")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 2

    prompts = load_prompts(args)
    waves = resolve_wavelengths(args)
    print(f"prompts     : {len(prompts)}", file=sys.stderr)
    print(f"wavelengths : {waves} (n={len(waves)})", file=sys.stderr)
    print(f"model       : {args.model_id}", file=sys.stderr)

    print("loading DOFA-CLIP (open_clip Path B)...", file=sys.stderr)
    from open_clip import create_model_from_pretrained, get_tokenizer
    model, preprocess = create_model_from_pretrained(args.model_id)
    tokenizer = get_tokenizer(args.model_id)
    model.eval()

    img = load_image_tensor(args.image, preprocess, n_bands_required=len(waves))
    if img.shape[1] != len(waves):
        print(f"ERROR: image has {img.shape[1]} bands but {len(waves)} "
              "wavelengths supplied", file=sys.stderr)
        return 2
    wvs = torch.tensor(waves, dtype=torch.float32)

    text = tokenizer(prompts, context_length=model.context_length)
    with torch.no_grad():
        img_feat = model.visual.trunk(img, wvs)
        if isinstance(img_feat, tuple):
            img_feat = img_feat[0]
        txt_feat = model.encode_text(text)
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        cos = (img_feat @ txt_feat.T).squeeze(0)
        # SigLIP scoring: sigmoid(cos * logit_scale + logit_bias)
        scale = float(model.logit_scale.exp().item())
        bias = float(getattr(model, "logit_bias", torch.tensor(0.0)).item())
        prob = torch.sigmoid(cos * scale + bias)

    rows = []
    print(f"\nresults (logit_scale={scale:.3f}, logit_bias={bias:.3f}):",
          file=sys.stderr)
    for i, p in enumerate(prompts):
        row = {"prompt": p,
               "cosine": float(cos[i].item()),
               "sigmoid_prob": float(prob[i].item())}
        rows.append(row)
        print(f"  cos={row['cosine']:.4f}  p={row['sigmoid_prob']:.4f}  {p}",
              file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
