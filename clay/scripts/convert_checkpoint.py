"""Write the encoder of the Clay v1.5 checkpoint to a safetensors file.

The published checkpoint (``made-with-clay/Clay``, ``v1.5/clay-v1.5.ckpt``) is a
Lightning checkpoint holding the whole masked-autoencoder: encoder, decoder and
the teacher used in training. Embeddings need only the encoder, the
``model.encoder.*`` tensors upstream's ``Embedder`` loads
(``claymodel/finetune/embedder/factory.py``). The file is read with PyTorch's
``weights_only`` loader: its pickle names only tensor-rebuild globals and
``OrderedDict``, which that loader allows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ENCODER_PREFIX = "model.encoder."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encoder_weights(checkpoint: Path) -> dict[str, torch.Tensor]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    encoder = {
        key.removeprefix(ENCODER_PREFIX): value
        for key, value in state.items()
        if key.startswith(ENCODER_PREFIX)
    }
    if not encoder:
        raise SystemExit(f"{checkpoint}: no tensors under {ENCODER_PREFIX}")
    # safetensors refuses tensors that share storage; clone() gives each its own.
    return {key: value.detach().contiguous().clone() for key, value in encoder.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expect-sha256", required=True)
    parser.add_argument("--source", required=True, help="where the checkpoint came from")
    args = parser.parse_args(argv)

    actual = sha256(args.checkpoint)
    if actual != args.expect_sha256:
        raise SystemExit(f"{args.checkpoint}: sha256 {actual}, expected {args.expect_sha256}")

    weights = encoder_weights(args.checkpoint)
    metadata = {"source": args.source, "source_sha256": actual, "kept": ENCODER_PREFIX}
    save_file(weights, str(args.output), metadata=metadata)
    summary = {"tensors": len(weights), "elements": sum(t.numel() for t in weights.values())}
    print(json.dumps({"output": str(args.output), **metadata, **summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
