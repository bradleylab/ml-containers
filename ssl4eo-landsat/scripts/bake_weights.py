"""Download every SSL4EO-L checkpoint that TorchGeo can load into TORCH_HOME.

TorchGeo's loaders fetch through torchvision's `load_state_dict_from_url`,
which reads `$TORCH_HOME/hub/checkpoints/<file name>` when the file is present
and only downloads otherwise. Fetching through the same call at build time,
with `check_hash=True`, puts each file where the loader looks and checks it
against the SHA-256 prefix in its name.

The set is defined by the weight enums, not by the Hugging Face listing: the
repository also holds 12 files of about 600 bytes whose state dict is empty,
and 6 alternate ViT-S/16 files that no TorchGeo 0.10.0 enum references.
"""

import os
from pathlib import Path

from torchgeo.models import ResNet18_Weights, ResNet50_Weights, ViTSmall16_Weights

# Five sensor/product levels x two SSL methods x three architectures.
EXPECTED_CHECKPOINTS = 30


def ssl4eo_l_weights() -> list:
    """Every SSL4EO-L member of the three TorchGeo weight enums."""
    return [
        w
        for enum in (ResNet18_Weights, ResNet50_Weights, ViTSmall16_Weights)
        for w in enum
        if w.meta.get("dataset") == "SSL4EO-L"
    ]


def main() -> None:
    weights = ssl4eo_l_weights()
    assert len(weights) == EXPECTED_CHECKPOINTS, (
        f"expected {EXPECTED_CHECKPOINTS} SSL4EO-L weights, found {len(weights)}"
    )
    for w in weights:
        w.get_state_dict(progress=False, check_hash=True, weights_only=True)
        print(f"baked {type(w).__name__}.{w.name}", flush=True)

    cache = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"
    files = sorted(cache.glob("*.pth"))
    assert len(files) == EXPECTED_CHECKPOINTS, f"{len(files)} files in {cache}"
    total = sum(f.stat().st_size for f in files)
    print(f"{len(files)} checkpoints, {total:,d} bytes in {cache}")


if __name__ == "__main__":
    main()
