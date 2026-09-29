"""Convert a CrossEarth training checkpoint to a safetensors file of its weights.

CrossEarth publishes its trained models as full mmengine checkpoints: the
weights under ``state_dict``, plus the training run's log, optimizer state and
schedule. Those are pickles, and unpickling can run code named in the file, so
the published files are never passed to ``torch.load`` as they are.

Instead the pickle is read by a restricted unpickler:

- the globals a tensor needs resolve to the functions PyTorch 2.1's own
  ``weights_only`` loader allows (``torch/_weights_only_unpickler.py``);
- the training-log globals mmengine writes into ``message_hub`` resolve to an
  inert placeholder class, so nothing is imported from numpy or mmengine and
  ``getattr`` is never called; those entries are discarded;
- any other global stops the conversion.

PyTorch 2.1 cannot extend its own allowlist (``torch.serialization
.add_safe_globals`` arrived in 2.4), and allowing these entries through it
would mean allowing ``builtins.getattr``. The placeholders keep that out.

Only the tensors upstream's inference model uses are kept: the backbone and
the decode head (``configs/_base_/models/CrossEarth_dinov2_mask2former.py``).
The checkpoints also hold modules of the unreleased training code, which are
dropped and listed in the output's metadata: a style augmentor, an auxiliary
head, and a convolution stack over the input image stored inside the backbone's
Reins adapter (``high_extractor``). The released ``Reins.forward_delta_feat``
takes the image but never uses it, and the authors state that only the
inference code is released (VisionXLab/CrossEarth issue #14).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

# PyTorch 2.1's weights_only allowlist, restricted to what these files use.
# Storage classes never reach find_class: torch.load's own unpickler wrapper
# resolves any global whose name contains "Storage" before delegating here.
ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch", "Tensor"): torch.Tensor,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._tensor", "_rebuild_from_type_v2"): torch._tensor._rebuild_from_type_v2,
}

# mmengine's training log (message_hub): HistoryBuffer objects, their
# statistics methods fetched with getattr, and numpy arrays and scalars. Found by
# listing every global the published checkpoints name (pickletools, no load).
TRAINING_LOG_GLOBALS = {
    ("mmengine.logging.history_buffer", "HistoryBuffer"),
    ("__builtin__", "getattr"),
    ("_codecs", "encode"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
}

INFERENCE_PREFIXES = ("backbone.", "decode_head.")
TRAINING_ONLY_PREFIXES = ("backbone.reins.high_extractor.",)


class Discarded:
    """Stands in for a training-log object: accepts any arguments or state, holds none."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __setstate__(self, state: object) -> None:
        pass


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if (module, name) in ALLOWED_GLOBALS:
            return ALLOWED_GLOBALS[(module, name)]
        if (module, name) in TRAINING_LOG_GLOBALS:
            return Discarded
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


class RestrictedPickleModule:
    """The pickle_module torch.load subclasses its unpickler from."""

    Unpickler = RestrictedUnpickler
    UnpicklingError = pickle.UnpicklingError

    @staticmethod
    def load(*args: object, **kwargs: object) -> object:
        raise pickle.UnpicklingError("only zip-format checkpoints are accepted")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dropped_module(key: str) -> str:
    for prefix in TRAINING_ONLY_PREFIXES:
        if key.startswith(prefix):
            return prefix.rstrip(".")
    return key.split(".", 1)[0]


def inference_weights(checkpoint: Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    """The inference tensors, and the modules dropped from the checkpoint."""
    loaded = torch.load(
        checkpoint,
        map_location="cpu",
        pickle_module=RestrictedPickleModule,
        weights_only=False,  # the restriction is RestrictedUnpickler's, see the docstring
    )
    state = loaded["state_dict"]
    kept = {
        key: value
        for key, value in state.items()
        if key.startswith(INFERENCE_PREFIXES) and not key.startswith(TRAINING_ONLY_PREFIXES)
    }
    if not kept:
        raise SystemExit(f"{checkpoint}: no tensors under {INFERENCE_PREFIXES}")
    dropped = sorted({dropped_module(key) for key in state if key not in kept})
    # safetensors refuses tensors that share storage; clone() gives each its own.
    return {key: value.detach().contiguous().clone() for key, value in kept.items()}, dropped


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

    weights, dropped = inference_weights(args.checkpoint)
    metadata = {"source": args.source, "source_sha256": actual, "dropped": ",".join(dropped)}
    save_file(weights, str(args.output), metadata=metadata)
    summary = {"tensors": len(weights), "elements": sum(t.numel() for t in weights.values())}
    print(json.dumps({"output": str(args.output), **metadata, **summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
