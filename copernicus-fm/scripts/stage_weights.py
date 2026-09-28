"""Stage the Copernicus-FM ViT-B/16 checkpoint where the image looks for it.

Run inside the image with the destination mounted at /stage (SMOKE.md gives the
Compute2 job). The checkpoint lands at
/stage/torch-cache/hub/checkpoints/CopernicusFM_ViT_base_varlang-085350e4.pth,
which the image finds when the same directory is mounted at /weights, because
the image sets TORCH_HOME=/weights/torch-cache.

The weights are staged rather than baked into the public image. They were
pretrained on Llama 3.2 encodings of variable names, and Meta's Llama 3.2
Community License places conditions on distributing a model trained with Llama
outputs that upstream's CC-BY-4.0 release does not address; keeping the file on
lab storage keeps the lab out of redistributing it.
"""

import hashlib
import os
import sys
from pathlib import Path

STAGE_ROOT = Path(os.environ.get("CFM_STAGE_DIR", "/stage"))
TORCH_HOME = STAGE_ROOT / "torch-cache"
FILENAME = "CopernicusFM_ViT_base_varlang-085350e4.pth"
# Hugging Face's LFS hash for torchgeo/copernicus-fm @ f395812, identical to
# upstream's CopernicusFM_ViT_base_varlang_e100.pth.
SHA256 = "085350e4fa0ebc6047a501aed1307394c4fcc23c013dd026db65af06411d6bee"
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    # torch.hub resolves TORCH_HOME at call time, so setting it here redirects
    # TorchGeo's own pinned download into the staging directory.
    os.environ["TORCH_HOME"] = str(TORCH_HOME)
    from torchgeo.models import CopernicusFM_Base_Weights

    CopernicusFM_Base_Weights.CopernicusFM_ViT.get_state_dict(
        progress=False, check_hash=True, weights_only=True
    )
    path = TORCH_HOME / "hub" / "checkpoints" / FILENAME
    digest = sha256_of(path)
    if digest != SHA256:
        print(f"stage_weights: SHA-256 mismatch for {path}: {digest}", file=sys.stderr)
        return 1
    print(f"staged {path} ({path.stat().st_size:,} bytes), SHA-256 ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
