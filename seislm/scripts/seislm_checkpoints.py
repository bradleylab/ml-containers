"""The two pretrained SeisLM checkpoints and the check that gates loading them.

The checkpoints are not in the image. The lab holds its own copy, downloaded
from the Google Drive folder linked in upstream's README, and mounts it at
/weights with the folder layout upstream's notebooks use. Loading a Lightning
checkpoint runs the pickle inside it, so a file is used only after its SHA-256
matches the one recorded here for the lab's copy.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WEIGHTS = Path("/weights")
HASH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class Checkpoint:
    path: str  # relative to the weights mount
    sha256: str
    size_bytes: int
    pretrain_config: str  # file in seisLM/configs/pretrain whose model_config it was trained with


CHECKPOINTS = {
    "base": Checkpoint(
        path="pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt",
        sha256="6e97c53c562e9ba6bc96df85d8d208b89ff84e80a25468825f29bb5bace5eacb",
        size_bytes=136_471_912,
        pretrain_config="pretrain_config_std_norm_single_ax_8_datasets_sample_pick_false.json",
    ),
    "large": Checkpoint(
        path="pretrained_seislm_large/checkpoints/epoch=39-step=701720.ckpt",
        sha256="db44b1a46512de2cdf5c310c7ccfdac0ef04931cc89b7768fad41c978266df4b",
        size_bytes=1_088_481_576,
        pretrain_config="pretrain_config_std_norm_single_ax_8_datasets_32bit_scaleup_samp_false.json",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verified_checkpoint(weights: Path, variant: str) -> Path:
    """Return the checkpoint's path once its size and SHA-256 match the record."""
    ckpt = CHECKPOINTS[variant]
    path = weights / ckpt.path
    if not path.is_file():
        raise SystemExit(f"{variant}: {path} not found; mount the lab's SeisLM weights at {weights}")
    if path.stat().st_size != ckpt.size_bytes:
        raise SystemExit(f"{variant}: {path} is {path.stat().st_size} bytes, expected {ckpt.size_bytes}")
    digest = sha256_file(path)
    if digest != ckpt.sha256:
        raise SystemExit(f"{variant}: SHA-256 {digest} does not match the recorded {ckpt.sha256}")
    return path
