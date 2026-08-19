"""Offline build-time (and on-node) check for the seist image.

Three things are asserted, in order of what actually breaks:

1. The 18 pretrained checkpoints arrived with the pinned source checkout.
   They live in-repo under `pretrained/`; if the tar extraction misses them
   the image is useless, and that is testable at build time.
2. All 15 SeisT model entrypoints are registered (5 task heads x S/M/L).
3. One checkpoint per task family loads with strict=True and produces the
   documented output shape — so a state-dict/architecture mismatch fails the
   build rather than the first real job.

No network, no weight download. Re-run it inside the container on Compute2
as the smoke test (see SMOKE.md).
"""

from __future__ import annotations

import importlib.metadata as im
from pathlib import Path

import torch

from models import create_model, get_model_list

SEIST_HOME = Path("/opt/seist")
PRETRAINED_DIR = SEIST_HOME / "pretrained"

N_CHECKPOINTS = 18
N_SEIST_ENTRYPOINTS = 15

IN_CHANNELS = 3
IN_SAMPLES = 8192  # upstream demo_predict.py window length

# One checkpoint per task family. Parameter counts and output shapes were
# read off the pinned commit; they pin the state-dict contract, not a
# scientific claim.
CASES = [
    # (entrypoint, checkpoint, n_params, output shape)
    (
        "seist_m_dpk",
        "seist_m_dpk_diting.pth",
        380_805,
        (1, 3, IN_SAMPLES),
    ),  # detection + P/S picking
    ("seist_m_pmp", "seist_m_pmp_diting.pth", 312_140, (1, 2)),  # first-motion polarity
    ("seist_m_emg", "seist_m_emg_diting.pth", 312_043, (1, 1)),  # magnitude
    ("seist_m_baz", "seist_m_baz_diting.pth", 312_043, (1, 1)),  # back-azimuth
    ("seist_m_dis", "seist_m_dis_diting.pth", 312_043, (1, 1)),  # epicentral distance
]

PACKAGES = ("timm", "obspy", "h5py", "numpy", "pandas", "matplotlib")


def main() -> None:
    print("torch", torch.__version__, "| cuda-build", torch.version.cuda)
    print(" | ".join(f"{pkg} {im.version(pkg)}" for pkg in PACKAGES))

    checkpoints = sorted(PRETRAINED_DIR.glob("*.pth"))
    total_mb = sum(p.stat().st_size for p in checkpoints) / 1e6
    print(
        f"checkpoints: {len(checkpoints)} files, {total_mb:.1f} MB in {PRETRAINED_DIR}"
    )
    assert len(checkpoints) == N_CHECKPOINTS, [p.name for p in checkpoints]

    entrypoints = sorted(n for n in get_model_list() if n.startswith("seist_"))
    assert len(entrypoints) == N_SEIST_ENTRYPOINTS, entrypoints

    x = torch.zeros(1, IN_CHANNELS, IN_SAMPLES)
    for name, checkpoint_name, expected_params, expected_shape in CASES:
        model = create_model(name, in_channels=IN_CHANNELS)
        state = torch.load(PRETRAINED_DIR / checkpoint_name, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        with torch.inference_mode():
            out_shape = tuple(model(x).shape)
        print(
            f"{name:14s} <- {checkpoint_name:26s} params={n_params:>8,} out={out_shape}"
        )
        assert n_params == expected_params, f"{name}: {n_params} != {expected_params}"
        assert out_shape == expected_shape, f"{name}: {out_shape} != {expected_shape}"

    print("SMOKE OK")


if __name__ == "__main__":
    main()
