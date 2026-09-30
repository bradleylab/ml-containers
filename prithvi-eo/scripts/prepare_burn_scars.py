"""Stage the Prithvi-EO 2.0 burn-scar model for offline inference.

Downloads ``Prithvi_EO_V2_300M_BurnScars.pt`` and ``burn_scars_config.yaml`` from
``ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars`` at a pinned revision,
checks both files' SHA-256, and writes the config with one change:
``backbone_pretrained: false``. As published, the config has TerraTorch fetch
the Prithvi backbone from the Hub before the checkpoint loads; the checkpoint
holds every weight of the model, backbone included, so the fetch only costs a
network call and a download. A ``provenance.json`` records the source and the
change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
CHECKPOINT = "Prithvi_EO_V2_300M_BurnScars.pt"
CONFIG = "burn_scars_config.yaml"
PRETRAINED_LINE = "backbone_pretrained: true"
OFFLINE_LINE = "backbone_pretrained: false"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(filename: str, revision: str, expected: str, cache: Path) -> Path:
    path = Path(hf_hub_download(REPO, filename, revision=revision, local_dir=str(cache)))
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{filename}: sha256 {actual}, expected {expected}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="directory to write the model files into")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--cache", type=Path, required=True, help="download directory, removable")
    args = parser.parse_args(argv)

    checkpoint = fetch(CHECKPOINT, args.revision, args.checkpoint_sha256, args.cache)
    config = fetch(CONFIG, args.revision, args.config_sha256, args.cache)

    text = config.read_text()
    if text.count(PRETRAINED_LINE) != 1:
        raise SystemExit(f"{CONFIG}: expected one '{PRETRAINED_LINE}' line")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / CONFIG).write_text(text.replace(PRETRAINED_LINE, OFFLINE_LINE))
    shutil.move(checkpoint, args.out / CHECKPOINT)
    provenance = {
        "source": f"huggingface.co/{REPO}@{args.revision}",
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": args.checkpoint_sha256,
        "config": CONFIG,
        "config_sha256": args.config_sha256,
        "config_change": f"'{PRETRAINED_LINE}' written as '{OFFLINE_LINE}'",
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance))
    return 0


if __name__ == "__main__":
    sys.exit(main())
