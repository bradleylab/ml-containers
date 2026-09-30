"""Load the baked Panopticon teacher and embed wavelength-labeled images.

Upstream's public hub entrypoint, ``panopticon_vitb14`` in hubconf.py,
downloads the teacher checkpoint from Hugging Face on every call, even when
the file is already on disk. This module builds the same architecture through
upstream's private builder ``_panopticon_vitb14`` and applies the same strict
``load_state_dict`` to the checkpoint baked into the image, so a run needs no
network and cannot silently pick up a different file.

Channel ids follow upstream's contract: an optical channel is its center
wavelength in nm, and a SAR channel is one of the category ids -1 to -12 from
``/opt/panopticon/dinov2/configs/data/satellites/sentinel1.yaml``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch

REPO_DIR = Path(os.environ.get("PANOPTICON_REPO", "/opt/panopticon"))
WEIGHTS_PATH = Path(
    os.environ.get("PANOPTICON_WEIGHTS", "/opt/panopticon-weights/panopticon_vitb14_teacher.pth")
)
EMBED_DIM = 768
PATCH_SIZE = 14
SAR_IDS = range(-12, 0)

# Upstream README example.
RGB_WAVELENGTHS_NM = (664.0, 559.0, 493.0)
# Sentinel-2 MSI L1C band centers (gaussian mu) from upstream
# dinov2/configs/data/satellites/sentinel2.yaml, rounded to 0.1 nm; B10 (cirrus)
# omitted, as in L2A products. The model floors optical ids to whole nm.
S2_12BAND_WAVELENGTHS_NM = (
    442.9,
    493.0,
    559.6,
    664.6,
    704.0,
    740.6,
    782.4,
    827.5,
    864.8,
    945.0,
    1613.9,
    2203.6,
)


class ChannelMismatchError(ValueError):
    """The image and its channel-id list disagree, or an id is not valid."""


def load_teacher(device: str | torch.device = "cpu") -> torch.nn.Module:
    """Build ViT-B/14 with the channel-attention patch embedding and load the teacher."""
    model = torch.hub.load(str(REPO_DIR), "_panopticon_vitb14", source="local")
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def validate_channel_ids(n_channels: int, channel_ids: Sequence[float]) -> None:
    """Refuse a channel-id list that does not describe the image band for band.

    The model adds the channel embeddings to the per-band tokens by
    broadcasting, so a single id would be applied to every band without error.
    """
    if len(channel_ids) != n_channels:
        raise ChannelMismatchError(
            f"image has {n_channels} bands but {len(channel_ids)} channel ids were given"
        )
    for value in channel_ids:
        if value < 0 and value not in SAR_IDS:
            raise ChannelMismatchError(f"negative channel id {value} is not a SAR category (-1 to -12)")
        if value == 0:
            raise ChannelMismatchError("channel id 0 is neither a wavelength nor a SAR category")


def channel_id_tensor(channel_ids: Sequence[float], batch_size: int, device) -> torch.Tensor:
    """(B, C) float tensor of channel ids.

    Upstream's channel embedding floors optical ids in place, so every call gets
    a fresh tensor rather than one the caller may reuse.
    """
    ids = torch.tensor([float(v) for v in channel_ids], dtype=torch.float32, device=device)
    return ids.unsqueeze(0).repeat(batch_size, 1)


@torch.no_grad()
def embed(
    model: torch.nn.Module, images: torch.Tensor, channel_ids: Sequence[float]
) -> dict[str, torch.Tensor]:
    """Normalized class token (B, 768) and patch tokens (B, L, 768) from the last block.

    ``images`` is (B, C, H, W), already standardized per band; H and W must be
    multiples of the 14-pixel patch.
    """
    if images.ndim != 4:
        raise ValueError(f"expected (B, C, H, W), got shape {tuple(images.shape)}")
    batch, n_channels, height, width = images.shape
    validate_channel_ids(n_channels, channel_ids)
    if height % PATCH_SIZE or width % PATCH_SIZE:
        raise ValueError(f"H and W must be multiples of {PATCH_SIZE}, got {height}x{width}")
    x_dict = {
        "imgs": images,
        "chn_ids": channel_id_tensor(channel_ids, batch, images.device),
    }
    out = model.forward_features(x_dict)
    return {"cls": out["x_norm_clstoken"], "patch": out["x_norm_patchtokens"]}
