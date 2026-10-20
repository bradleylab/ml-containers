"""Pinned SPECTER2 checkpoints: where they come from and where the image keeps them.

Revisions and SHA-256 values are the Hugging Face commit and the Git LFS object
id of each weight file at that commit. The adapter names follow the model card's
table of task formats.
"""

import os
from dataclasses import dataclass
from pathlib import Path

WEIGHTS_ROOT = Path(os.environ.get("SPECTER2_WEIGHTS", "/opt/specter2/weights"))
BASE = "base"


@dataclass(frozen=True)
class Checkpoint:
    repo_id: str
    revision: str
    weights_file: str
    sha256: str

    @property
    def local_dir(self) -> Path:
        return WEIGHTS_ROOT / self.repo_id.split("/")[1]


CHECKPOINTS = {
    BASE: Checkpoint(
        "allenai/specter2_base",
        "3447645e1def9117997203454fa4495937bfbd83",
        "pytorch_model.bin",
        "801eda968fad1752fe846a8e572bfdc25202be85544680dda4c99f4589646ebc",
    ),
    "proximity": Checkpoint(
        "allenai/specter2",
        "2081559630a80fc5851d8f798a05ba81e9468089",
        "pytorch_adapter.bin",
        "5b6105714f78b294daa7e2d181134df12f0acb7a4ec7c1e3c13e512c908db94a",
    ),
    "adhoc_query": Checkpoint(
        "allenai/specter2_adhoc_query",
        "3f4448817028388648a74349ece07af4518ec5bd",
        "pytorch_adapter.bin",
        "c369a7a0d08c82d694f6d1dbdd824415e8c6c98e4317d87e65080fd9a73da0a1",
    ),
    "classification": Checkpoint(
        "allenai/specter2_classification",
        "d843816b414a856d9a5268d5865f188bb589e6e1",
        "pytorch_adapter.bin",
        "bf733988173916c54e731618a6f1cfdfd0ca3f911fe2a7b3c3085aa76405c9e5",
    ),
    "regression": Checkpoint(
        "allenai/specter2_regression",
        "a9d105def7a081c8f7c5a88ab4145646901c6003",
        "pytorch_adapter.bin",
        "aa2676e2c90489c435ed201df0e25a9691765211f7b51a0e738dad4ceceac723",
    ),
}
ADAPTERS = [name for name in CHECKPOINTS if name != BASE]
