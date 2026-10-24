"""Build-time check of the contract entrypoint, run inside the image with no network.

Runs /usr/local/bin/dinov3-sat-contract the way the executor does, under an
empty environment (a Slurm job step carries the host's environment, not the
image's), on two small synthetic images: one at the input size and one wider
than it is tall. input_size 256 keeps the CPU forward pass short. The check
reads the manifest back against the files, the feature grid geometry and the
preview size, confirms that bad parameters are refused before anything is
written, and confirms that the entrypoint and extract_features.py agree on
which files are images.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

import numpy as np
from PIL import Image

ENTRYPOINT = Path("/usr/local/bin/dinov3-sat-contract")
EXTRACTOR_DIR = Path("/opt/dinov3-sat")
SEED = 0
INPUT_SIZE = 256
GRID = INPUT_SIZE // 16  # the patch size the smoke test asserts
CHANNELS = 1024  # ViT-L feature width, also asserted by the smoke test
PREFIX_TOKENS = 5  # CLS + 4 registers (README, "Three things that bite")
IMAGES = {"square.png": (INPUT_SIZE, INPUT_SIZE), "wide.png": (300, 200)}


def fail(message: str) -> None:
    print(f"contract check FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def load_entrypoint():
    loader = SourceFileLoader("dinov3_sat_contract", str(ENTRYPOINT))
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    sys.modules[loader.name] = module  # dataclasses look their module up here
    loader.exec_module(module)
    return module


def check_suffixes_agree() -> None:
    sys.path.insert(0, str(EXTRACTOR_DIR))
    import extract_features

    ours, theirs = load_entrypoint().IMAGE_SUFFIXES, extract_features.IMAGE_SUFFIXES
    if ours != theirs:
        fail(f"entrypoint reads {sorted(ours)}, extractor {sorted(theirs)}")
    print("ok: entrypoint and extractor agree on image suffixes")


def make_case(root: Path, params: dict) -> Path:
    rng = np.random.default_rng(SEED)
    primary = root / "input" / "primary"
    primary.mkdir(parents=True)
    for name, (width, height) in IMAGES.items():
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(primary / name)
    (root / "params.json").write_text(json.dumps(params))
    return root


def run_case(case: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            str(ENTRYPOINT),
            "--input-dir",
            str(case / "input"),
            "--output-dir",
            str(case / "output"),
            "--params-json",
            str(case / "params.json"),
        ],
        env={},
        capture_output=True,
        text=True,
        check=False,  # the caller judges the exit status
    )


def check_success(case: Path) -> None:
    result = run_case(case)
    print(result.stdout, end="")
    if result.returncode != 0:
        fail(f"exit {result.returncode}: {result.stderr.strip()}")
    out = case / "output"
    manifest = json.loads((out / "run.json").read_text())
    if manifest.get("contract_version") != 1:
        fail(f"contract_version is {manifest.get('contract_version')!r}")
    roles = Counter(entry["role"] for entry in manifest["outputs"])
    if roles != {"features": 2, "preview": 2, "feature_metadata": 1}:
        fail(f"manifest roles are {dict(roles)}")
    for entry in manifest["outputs"]:
        if hashlib.sha256((out / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            fail(f"sha256 of {entry['path']} does not match the manifest")

    for name in IMAGES:
        stem = Path(name).stem
        with np.load(out / "features" / f"{stem}.npz") as archive:
            features = archive["features"]
        if features.shape != (GRID, GRID, CHANNELS) or features.dtype != np.float16:
            fail(f"{stem}.npz holds {features.dtype} {features.shape}")
        with Image.open(out / "previews" / f"{stem}.png") as preview:
            if (preview.size, preview.mode) != ((GRID, GRID), "RGB"):
                fail(f"{stem}.png is {preview.mode} {preview.size}")
    meta = json.loads((out / "features" / "_features_meta.json").read_text())
    if (meta["grid"], meta["num_prefix_tokens"]) != (GRID, PREFIX_TOKENS):
        fail(f"_features_meta.json records grid {meta['grid']}, prefix {meta['num_prefix_tokens']}")
    if not any("not square" in warning for warning in manifest.get("warnings", [])):
        fail(f"no stretching warning for wide.png: {manifest.get('warnings')}")
    print(f"ok: full run under an empty environment; warnings {manifest['warnings']}")


def check_refused(case: Path, expected: str) -> None:
    result = run_case(case)
    if result.returncode != 2 or expected not in result.stderr:
        fail(f"expected a refusal naming {expected!r}, got {result.returncode}: {result.stderr}")
    if (case / "output" / "run.json").exists():
        fail("a refused run wrote a manifest")
    print(f"ok: refused ({expected})")


def main() -> int:
    check_suffixes_agree()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        check_success(make_case(root / "run", {"input_size": str(INPUT_SIZE)}))
        check_refused(make_case(root / "size", {"input_size": "500"}), "input_size must be one of")
        check_refused(make_case(root / "name", {"batch_size": "4"}), "unknown parameters")
    print("dinov3-sat contract check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
