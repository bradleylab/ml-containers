"""Pinned ForPT checkpoints, and how the forpt image builds models and reads point clouds.

Every model is built from upstream's own config file, and every input goes through
the test-split transform list in that config, so the commands here run the same
preprocessing upstream evaluated with. The one change is that label keys are
dropped from the transform list, because the inputs here are unlabeled.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import torch
from forpt.datasets.transform import Compose
from forpt.datasets.utils import collate_fn
from forpt.models import build_model
from forpt.utils.config import Config

SRC = Path(os.environ.get("FORPT_SRC", "/opt/forpt/src"))
WEIGHTS_ROOT = Path(os.environ.get("FORPT_WEIGHTS", "/opt/forpt/weights"))
HF_REPO = "prs-eth/ForPT"
HF_REVISION = "6b9047eb9c50689aaf967b894f7c7b08c8b7d026"

# Upstream's test transforms keep one random point per 5 cm voxel
# (GridSample mode="train"), and its configs set shuffle_orders=True, which
# draws the serialization order at random on every forward pass. Seeding before
# each cloud makes a rerun of the same file give the same answer.
SEED = 0

# Keys that hold ground truth in upstream's test transforms.
LABEL_KEYS = {"segment", "origin_segment", "category", "regression"}

PRETRAIN = "pretrain"
SEMSEG = "semseg"
SPECIES = "species"
AGE = "age"


@dataclass(frozen=True)
class Checkpoint:
    """One upstream checkpoint: its config, its file on the Hub and that file's Git LFS SHA-256."""

    config: str
    hf_file: str
    sha256: str

    @property
    def config_path(self) -> Path:
        return SRC / "configs" / f"{self.config}.py"

    @property
    def local_path(self) -> Path:
        """Where the image keeps the checkpoint, reduced to {"state_dict", "epoch"}.

        That is the part of upstream's format its tester reads, so tools/test.py
        runs on these files too."""
        return WEIGHTS_ROOT / f"{self.config.replace('/', '__')}.pth"


CHECKPOINTS = {
    PRETRAIN: Checkpoint(
        "00_ssl_pretrain/pretrain-forpt",
        "00_ssl_pretrain/pretrain-forpt/model/epoch_30.pth",
        "6de5726bc22df0a76e8a921d3afb5dc538f2f63b5255fd19b583794de1b0d00a",
    ),
    SEMSEG: Checkpoint(
        "01_forinstancev2_sem/semseg-forpt-ft-p100",
        "01_forinstancev2_sem/semseg-forpt-ft-p100/model/model_best.pth",
        "7f31d2428793bb873e328f9ce70a34174be3100608044cadc76f75ff068b4de1",
    ),
    SPECIES: Checkpoint(
        "03_forspecies20k/cls-forpt-ft-allpts",
        "03_forspecies20k/cls-forpt-ft-allpts/model/model_last.pth",
        "2d88a698c49ba733b0e148dfc7099a9b2d3c9249b40cc5d7a2d0e689c36192de",
    ),
    AGE: Checkpoint(
        "04_forage/age-forpt-ft-p100",
        "04_forage/age-forpt-ft-p100/model/model_best.pth",
        "cb5c698bba0f54aa5c5cd098d52093eb69a3ed16bf77f89ff0ff4ae7f6de9256",
    ),
}

# The pretraining checkpoint holds a student and a teacher; upstream's
# fine-tuning configs initialize from the student's backbone.
PRETRAIN_BACKBONE_PREFIX = "module.student.backbone."
DDP_PREFIX = "module."


def load_config(task: str) -> Config:
    return Config.fromfile(str(CHECKPOINTS[task].config_path))


def class_names(task: str) -> list[str]:
    return list(load_config(task).data.names)


def load_state_dict(task: str) -> dict[str, torch.Tensor]:
    return torch.load(CHECKPOINTS[task].local_path, map_location="cpu", weights_only=True)["state_dict"]


def build_task_model(task: str, device: str) -> torch.nn.Module:
    """Build the semseg, species or age model from its config and load its weights strictly."""
    model_config = load_config(task).model
    # Losses hold no weights and are used only in training; building the age
    # loss reads the FORage training metadata CSV, which the image does not have.
    model_config.criteria = None
    model = build_model(model_config)
    model.load_state_dict(load_state_dict(task))
    return model.to(device).eval()


def build_pretrained_encoder(device: str) -> torch.nn.Module:
    """Build the self-supervised LitePT-S encoder and load the student's weights strictly."""
    model = build_model(load_config(PRETRAIN).model.backbone)
    model.load_state_dict(load_state_dict(PRETRAIN))
    return model.to(device).eval()


def strip_checkpoint(task: str, checkpoint: dict) -> dict[str, torch.Tensor]:
    """Reduce an upstream training checkpoint to the state dict its model loads."""
    state = checkpoint["state_dict"]
    if task == PRETRAIN:
        return {
            k.removeprefix(PRETRAIN_BACKBONE_PREFIX): v
            for k, v in state.items()
            if k.startswith(PRETRAIN_BACKBONE_PREFIX)
        }
    return {k.removeprefix(DDP_PREFIX): v for k, v in state.items()}


def _unlabeled(step: dict) -> dict | None:
    if step["type"] == "Copy":
        return None
    if step["type"] == "Collect":
        step = dict(step, keys=tuple(k for k in step["keys"] if k not in LABEL_KEYS))
    return step


def test_transform(task: str) -> Compose:
    """Upstream's test-split transforms for a task, without the steps that need labels."""
    steps = [_unlabeled(dict(s)) for s in load_config(task).data.test.transform]
    return Compose([s for s in steps if s is not None])


def model_input(coord: np.ndarray, transform: Compose, device: str) -> dict:
    """Turn one cloud's (N, 3) coordinates in meters into a batch of one."""
    batch = collate_fn([transform({"coord": coord.astype(np.float64)})])
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def read_points(path: Path) -> np.ndarray:
    """x, y, z in float64 from LAS/LAZ, .npy, or comma-separated text (FOR-species20K's format)."""
    suffix = path.suffix.lower()
    if suffix in {".las", ".laz"}:
        las = laspy.read(path)
        return np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    if suffix == ".npy":
        return np.load(path)[:, :3].astype(np.float64)
    if suffix in {".txt", ".csv", ".xyz"}:
        return np.loadtxt(path, delimiter=",", usecols=(0, 1, 2), dtype=np.float64)
    raise ValueError(f"{path}: expected .las, .laz, .npy, .txt, .csv or .xyz")


def require_cuda() -> str:
    """LitePT's attention runs only through FlashAttention and the PointROPE CUDA kernel."""
    if not torch.cuda.is_available():
        raise SystemExit("ForPT inference needs a CUDA GPU: LitePT attention runs only through FlashAttention")
    return "cuda"
