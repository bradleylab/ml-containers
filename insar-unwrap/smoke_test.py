"""Offline build-time (and on-node) check for the insar-unwrap image.

Builds all four architectures from the pinned upstream source and asserts
the parameter counts published in `results/standardized/efficiency_metrics.txt`,
plus the 6-channel-in / 1-channel-out 128x128 patch contract that any caller
has to satisfy. An upstream architecture edit that silently changes model
capacity therefore fails the build rather than shipping checkpoints that no
longer load.

No weights are fetched and no network is touched, so the build stays
deterministic. Re-run it inside the container on Compute2 as the smoke test
(see SMOKE.md).
"""

from __future__ import annotations

import importlib.metadata as im

import torch

from train.standardized.base_config import BaseConfig
from train.standardized.train_attention_unet import AttentionInSAR_UNet
from train.standardized.train_enhanced_unet import EnhancedInSAR_UNet
from train.standardized.train_hybrid import HybridMultiScaleUNet
from train.standardized.train_vanilla_unet import VanillaInSAR_UNet

# Parameter counts as published in results/standardized/efficiency_metrics.txt
# at the pinned commit (7.76M / 8.29M / 11.37M / 17.21M there; exact integers
# confirmed by constructing each model).
EXPECTED_PARAMS = {
    VanillaInSAR_UNet: 7_763_905,
    EnhancedInSAR_UNet: 8_287_088,
    AttentionInSAR_UNet: 11_372_820,
    HybridMultiScaleUNet: 17_206_128,
}

PACKAGES = (
    "numpy",
    "scipy",
    "rasterio",
    "scikit-learn",
    "matplotlib",
    "huggingface_hub",
)


def main() -> None:
    print("torch", torch.__version__, "| cuda-build", torch.version.cuda)
    print(" | ".join(f"{pkg} {im.version(pkg)}" for pkg in PACKAGES))

    shape_cfg = (BaseConfig.IN_CHANNELS, BaseConfig.OUT_CHANNELS, BaseConfig.PATCH_SIZE)
    assert shape_cfg == (6, 1, 128), f"input contract drifted: {shape_cfg}"

    x = torch.zeros(
        1, BaseConfig.IN_CHANNELS, BaseConfig.PATCH_SIZE, BaseConfig.PATCH_SIZE
    )
    for model_cls, expected in EXPECTED_PARAMS.items():
        model = model_cls(
            BaseConfig.IN_CHANNELS,
            BaseConfig.OUT_CHANNELS,
            base_channels=BaseConfig.BASE_CHANNELS,
            dropout=0.0,
        ).eval()
        n_params = sum(p.numel() for p in model.parameters())
        with torch.inference_mode():
            out_shape = tuple(model(x).shape)
        print(f"{model_cls.__name__:24s} params={n_params:>10,}  out={out_shape}")
        assert n_params == expected, f"{model_cls.__name__}: {n_params} != {expected}"
        assert out_shape == (1, 1, 128, 128), f"{model_cls.__name__}: {out_shape}"

    print("SMOKE OK")


if __name__ == "__main__":
    main()
