"""Extract dense DINOv3-SAT patch features for a set of RGB images.

Emits one ``.npz`` per input image holding a ``(grid_h, grid_w, 1024)`` feature
grid, plus a JSON sidecar recording the checkpoint, input size and
normalization actually used — so a downstream head can never silently train on
features it cannot reproduce.

Two details this wrapper exists to get right:

**Prefix tokens.** DINOv3 prepends a CLS token and register tokens, so a
512 px input returns 1029 tokens, not 32*32=1024. Reshaping the raw sequence to
a spatial grid silently scrambles it. The prefix count is read from the model
(``num_prefix_tokens``) rather than assumed.

**Normalization.** The SAT-493M weights carry their own mean/std, not
ImageNet's. Both are resolved from timm's ``pretrained_cfg``.

    python extract_features.py --input chips/ --out features/ --input-size 1024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

MODEL = "hf-hub:timm/vit_large_patch16_dinov3.sat493m"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_model(device: str) -> tuple[torch.nn.Module, dict]:
    model = timm.create_model(MODEL, pretrained=True, num_classes=0)
    model.eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    return model, cfg


def iter_images(root: Path) -> list[Path]:
    if root.is_file():
        if root.suffix.lower() == ".txt":
            return [Path(line) for line in root.read_text().split() if line]
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def preprocess(paths: list[Path], size: int, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    batch = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BICUBIC)
        # np.array (not asarray) — PIL's buffer is read-only and torch warns
        batch.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))
    x = torch.stack(batch).float().div_(255.0)
    return (x - mean_t) / std_t


@torch.no_grad()
def encode(
    model: torch.nn.Module, x: torch.Tensor, patch: int, device: str, fp16: bool
) -> np.ndarray:
    """Return ``(batch, grid, grid, channels)`` patch features, prefix stripped."""
    x = x.to(device)
    with torch.autocast(device_type=device.split(":")[0], enabled=fp16):
        tokens = model.forward_features(x)
    n_prefix = getattr(model, "num_prefix_tokens", 0)
    tokens = tokens[:, n_prefix:, :]
    grid = x.shape[-1] // patch
    expected = grid * grid
    if tokens.shape[1] != expected:
        raise ValueError(
            f"got {tokens.shape[1]} patch tokens after stripping {n_prefix} prefix "
            f"tokens, expected {expected} for a {x.shape[-1]}px input at patch {patch}"
        )
    b, _, c = tokens.shape
    return tokens.reshape(b, grid, grid, c).float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input", type=Path, required=True, help="image dir, file, or .txt list"
    )
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument(
        "--input-size",
        type=int,
        default=512,
        help="square input size in px; must be a multiple of the patch size (16). "
        "Larger = finer patch grid on the ground, at ~quadratic token cost.",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true", help="autocast the forward pass")
    ap.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = ap.parse_args()

    model, cfg = load_model(args.device)
    patch = model.patch_embed.patch_size[0]
    if args.input_size % patch:
        raise SystemExit(f"--input-size {args.input_size} is not a multiple of {patch}")

    paths = iter_images(args.input)
    if not paths:
        raise SystemExit(f"no images under {args.input}")
    args.out.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": MODEL,
        "input_size": args.input_size,
        "patch_size": patch,
        "grid": args.input_size // patch,
        "num_features": model.num_features,
        "num_prefix_tokens": getattr(model, "num_prefix_tokens", 0),
        "mean": list(cfg["mean"]),
        "std": list(cfg["std"]),
        "dtype": args.dtype,
        "timm": timm.__version__,
        "torch": torch.__version__,
    }
    (args.out / "_features_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)

    written = 0
    for start in range(0, len(paths), args.batch_size):
        chunk = paths[start : start + args.batch_size]
        x = preprocess(chunk, args.input_size, cfg["mean"], cfg["std"])
        feats = encode(model, x, patch, args.device, args.fp16)
        for path, feat in zip(chunk, feats, strict=True):
            np.savez_compressed(
                args.out / f"{path.stem}.npz", features=feat.astype(args.dtype)
            )
            written += 1
        print(f"{written}/{len(paths)}", flush=True)
    print(f"wrote {written} feature grids to {args.out}")


if __name__ == "__main__":
    main()
