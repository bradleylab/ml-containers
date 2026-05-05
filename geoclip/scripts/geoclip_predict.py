"""
Predict the top-k GPS locations for an image using GeoCLIP
(Vivanco Cepeda et al., NeurIPS 2023). CLIP-style alignment between
photo embeddings and a learned location encoder over MP-16 (~4.7M
worldwide geo-tagged photos).

Usage (host):
    docker run --rm \\
      -v "$PWD:/work" \\
      ghcr.io/bradleylab/geoclip:v1 \\
      python /opt/scripts/geoclip_predict.py \\
        --image /work/photo.jpg \\
        --top-k 5 \\
        --out /work/preds.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from geoclip import GeoCLIP


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True,
                    help="Input image path (any format Pillow reads)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional CSV output (otherwise stdout table only)")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 2

    print("loading GeoCLIP...", file=sys.stderr)
    model = GeoCLIP()  # from_pretrained=True is the default
    model.eval()
    print(f"  image encoder : {type(model.image_encoder).__name__}",
          file=sys.stderr)
    print(f"  location enc  : {type(model.location_encoder).__name__}",
          file=sys.stderr)
    if hasattr(model, "gps_gallery"):
        print(f"  gps gallery   : {tuple(model.gps_gallery.shape)}",
              file=sys.stderr)

    with torch.no_grad():
        gps_top, prob_top = model.predict(str(args.image), top_k=args.top_k)

    gps_top = gps_top.detach().cpu()
    prob_top = prob_top.detach().cpu()

    rows = []
    print(f"\ntop {args.top_k} predicted locations:", file=sys.stderr)
    for rank in range(gps_top.shape[0]):
        lat = float(gps_top[rank, 0])
        lon = float(gps_top[rank, 1])
        prob = float(prob_top[rank])
        rows.append({"rank": rank + 1, "lat": lat, "lon": lon,
                     "prob": prob})
        print(f"  {rank + 1:2d}. lat={lat:+9.5f}  lon={lon:+10.5f}  "
              f"p={prob:.4f}",
              file=sys.stderr)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
