"""Offline build test: load the baked teacher and embed two synthetic batches.

Every socket connect raises, so a code path that reaches for the network fails
the build instead of a job on a compute node.
"""

from __future__ import annotations

import importlib.metadata as im
import os
import socket
from pathlib import Path

import torch
from panopticon_model import (
    EMBED_DIM,
    RGB_WAVELENGTHS_NM,
    S2_12BAND_WAVELENGTHS_NM,
    embed,
    load_teacher,
)

BATCH = 2
IMAGE_SIZE = 224
N_PATCHES = (IMAGE_SIZE // 14) ** 2


class _NoNetworkSocket(socket.socket):
    def connect(self, *args, **kwargs):
        raise RuntimeError("network access during the offline smoke test")


def main() -> None:
    socket.socket = _NoNetworkSocket
    torch.manual_seed(0)
    model = load_teacher("cpu")
    for name, ids in {"rgb": RGB_WAVELENGTHS_NM, "s2_12band": S2_12BAND_WAVELENGTHS_NM}.items():
        out = embed(model, torch.randn(BATCH, len(ids), IMAGE_SIZE, IMAGE_SIZE), ids)
        assert tuple(out["cls"].shape) == (BATCH, EMBED_DIM), (name, out["cls"].shape)
        assert tuple(out["patch"].shape) == (BATCH, N_PATCHES, EMBED_DIM), (name, out["patch"].shape)
        assert torch.isfinite(out["cls"]).all() and torch.isfinite(out["patch"]).all(), name
        print(f"{name}: {len(ids)} bands -> cls {tuple(out['cls'].shape)}, patch {tuple(out['patch'].shape)}")
    ref = Path(os.environ.get("PANOPTICON_REPO", "/opt/panopticon"), "UPSTREAM_REF")
    print("panopticon", ref.read_text().strip() if ref.exists() else "(no UPSTREAM_REF)")
    versions = {pkg: im.version(pkg) for pkg in ("numpy", "h5py", "rasterio")}
    print(f"torch {torch.__version__} | cuda build {torch.version.cuda} | {versions}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
