#!/usr/bin/env python3
"""SAM 2 inference CLI — single entrypoint for the bradleylab/sam2 image.

Modes
-----
amg     Automatic mask generation. No prompts; SAM 2 segments every plausible
        object in the image. Best for "what's in this scene" workflows like
        boulder fields, tree crowns, cells.
point   Point-prompt segmentation. Pass --points '[[x,y],...]' (and optionally
        --point-labels '[1,1,0,...]' where 1 = foreground, 0 = background).
        Produces three candidate masks per prompt.
box     Box-prompt segmentation. Pass --boxes '[[x1,y1,x2,y2],...]'.

Output
------
A JSON file with image metadata + a list of mask records. Each mask carries
its bbox, area, predicted IoU, stability score, and a COCO-format RLE
(``size``, ``counts``) encoding of the binary mask. Decode with
``pycocotools.mask.decode`` or any COCO-compatible library.

Optional ``--save-masks-dir`` writes one binary PNG per mask alongside the
JSON, useful when you want to bypass RLE altogether.

Examples
--------
    # automatic mask generation
    sam2 --image scene.jpg --output masks.json --mode amg

    # box prompts
    sam2 --image scene.jpg --output masks.json --mode box \\
         --boxes '[[10,10,200,200],[300,50,500,400]]'

    # point prompts
    sam2 --image scene.jpg --output masks.json --mode point \\
         --points '[[100,150],[250,150]]' --point-labels '[1,1]'
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils  # COCO RLE encode/decode

log = logging.getLogger("sam2_infer")

DEFAULT_MODEL_ID = "facebook/sam2.1-hiera-large"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def select_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(path: Path) -> np.ndarray:
    """Load an RGB image as (H, W, 3) uint8 ndarray."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask as a COCO RLE dict (Fortran order, ASCII counts)."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def amg_to_records(masks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert SAM 2 automatic-mask-generator output dicts to JSON-friendly records."""
    out: list[dict[str, Any]] = []
    for i, m in enumerate(masks):
        seg = m["segmentation"]
        out.append({
            "id": i,
            "bbox": [int(v) for v in m["bbox"]],
            "area": int(m["area"]),
            "predicted_iou": float(m["predicted_iou"]),
            "stability_score": float(m["stability_score"]),
            "point_coords": [[float(p[0]), float(p[1])] for p in m["point_coords"]],
            "crop_box": [int(v) for v in m["crop_box"]],
            "rle": encode_rle(seg),
        })
    return out


def predictor_to_records(masks: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    """Convert SAM 2 image-predictor output (M, H, W) into JSON-friendly records."""
    out: list[dict[str, Any]] = []
    for i, (m, s) in enumerate(zip(masks, scores)):
        rle = encode_rle(m.astype(bool))
        ys, xs = np.where(m > 0)
        if xs.size == 0:
            bbox = [0, 0, 0, 0]
        else:
            bbox = [int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min()), int(ys.max() - ys.min())]
        out.append({
            "id": i,
            "bbox": bbox,
            "area": int(m.sum()),
            "predicted_iou": float(s),
            "rle": rle,
        })
    return out


def write_per_mask_pngs(records: list[dict[str, Any]], out_dir: Path) -> None:
    """Save each mask as a binary PNG named by mask id."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        rle = {"size": r["rle"]["size"], "counts": r["rle"]["counts"].encode("ascii")}
        m = mask_utils.decode(rle)
        Image.fromarray((m * 255).astype("uint8"), mode="L").save(out_dir / f"mask_{r['id']:06d}.png")


def run_amg(image: np.ndarray, model_id: str, device: str,
            points_per_side: int, pred_iou_thresh: float,
            stability_score_thresh: float, min_mask_area: int,
            multimask_output: bool) -> list[dict[str, Any]]:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    log.info(f"loading SAM 2 AMG: {model_id} on {device}")
    t0 = time.perf_counter()
    amg = SAM2AutomaticMaskGenerator.from_pretrained(
        model_id,
        device=device,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_area,
        multimask_output=multimask_output,
    )
    log.info(f"  loaded in {time.perf_counter() - t0:.1f}s")
    log.info("running AMG...")
    t0 = time.perf_counter()
    if device.startswith("cuda"):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            masks = amg.generate(image)
    else:
        with torch.inference_mode():
            masks = amg.generate(image)
    log.info(f"  AMG produced {len(masks)} masks in {time.perf_counter() - t0:.1f}s")
    return amg_to_records(masks)


def run_point_or_box(image: np.ndarray, model_id: str, device: str,
                     points: np.ndarray | None, point_labels: np.ndarray | None,
                     boxes: np.ndarray | None, multimask_output: bool
                     ) -> list[dict[str, Any]]:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    log.info(f"loading SAM 2 predictor: {model_id} on {device}")
    t0 = time.perf_counter()
    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    log.info(f"  loaded in {time.perf_counter() - t0:.1f}s")
    log.info("setting image embedding...")
    if device.startswith("cuda"):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                point_coords=points, point_labels=point_labels, box=boxes,
                multimask_output=multimask_output,
            )
    else:
        with torch.inference_mode():
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                point_coords=points, point_labels=point_labels, box=boxes,
                multimask_output=multimask_output,
            )
    if masks.ndim == 4:
        # Multiple prompts: shape (N, 3 or 1, H, W). Keep top-IoU per prompt.
        best = scores.argmax(axis=1)
        masks = np.stack([masks[i, best[i]] for i in range(masks.shape[0])])
        scores = np.stack([scores[i, best[i]] for i in range(scores.shape[0])])
    return predictor_to_records(masks.astype(bool), scores)


def parse_json_arg(text: str | None, name: str) -> Any | None:
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--{name}: invalid JSON ({e})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True, help="Input RGB image")
    ap.add_argument("--output", type=Path, required=True, help="Output JSON path")
    ap.add_argument("--mode", choices=["amg", "point", "box"], default="amg")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help=f"HF model id (default: {DEFAULT_MODEL_ID})")
    ap.add_argument("--device", default=None,
                    help="cuda | cpu | mps (default: auto-detect)")
    ap.add_argument("--save-masks-dir", type=Path, default=None,
                    help="Optional dir to also write one PNG per mask")
    ap.add_argument("--multimask-output", action="store_true", default=False,
                    help="Return all 3 candidate masks per prompt; for AMG, keep all 3")
    # AMG-only.
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--pred-iou-thresh", type=float, default=0.7)
    ap.add_argument("--stability-score-thresh", type=float, default=0.95)
    ap.add_argument("--min-mask-area", type=int, default=100)
    # Prompted modes.
    ap.add_argument("--points", type=str, default=None,
                    help='JSON list-of-lists, e.g. "[[100,150],[250,150]]"')
    ap.add_argument("--point-labels", type=str, default=None,
                    help='JSON list of 0/1, e.g. "[1,1,0]"')
    ap.add_argument("--boxes", type=str, default=None,
                    help='JSON list of [x1,y1,x2,y2] boxes')
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    log.info(f"=== sam2_infer ===")
    log.info(f"image: {args.image}  mode: {args.mode}  model: {args.model_id}")

    if not args.image.exists():
        log.error(f"input image not found: {args.image}")
        return 2

    image = load_image(args.image)
    log.info(f"loaded image shape={image.shape} dtype={image.dtype}")

    device = select_device(args.device)
    log.info(f"device: {device}  (HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')})")

    t0 = time.perf_counter()
    if args.mode == "amg":
        records = run_amg(
            image, model_id=args.model_id, device=device,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            min_mask_area=args.min_mask_area,
            multimask_output=args.multimask_output,
        )
    else:
        points = parse_json_arg(args.points, "points")
        point_labels = parse_json_arg(args.point_labels, "point-labels")
        boxes = parse_json_arg(args.boxes, "boxes")
        if args.mode == "point" and points is None:
            log.error("--points required for mode=point")
            return 2
        if args.mode == "box" and boxes is None:
            log.error("--boxes required for mode=box")
            return 2
        records = run_point_or_box(
            image, model_id=args.model_id, device=device,
            points=np.asarray(points, dtype=np.float32) if points is not None else None,
            point_labels=np.asarray(point_labels, dtype=np.int32) if point_labels is not None else None,
            boxes=np.asarray(boxes, dtype=np.float32) if boxes is not None else None,
            multimask_output=args.multimask_output,
        )

    payload = {
        "image": str(args.image.name),
        "image_shape": [int(x) for x in image.shape],
        "mode": args.mode,
        "model_id": args.model_id,
        "device": device,
        "n_masks": len(records),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "masks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    log.info(f"wrote {args.output} — {len(records)} masks, {payload['elapsed_s']}s")

    if args.save_masks_dir:
        write_per_mask_pngs(records, args.save_masks_dir)
        log.info(f"per-mask PNGs -> {args.save_masks_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
