"""Tests for bin/dinov3-sat-contract that need neither the model nor the image.

The extractor and the device probe are replaced by stand-ins, so what is
tested is the entrypoint's own work: parameters, input discovery, warnings,
the preview and the manifest. Run from the repository root:

    uv run --python 3.11 --with numpy --with pillow --with pytest \
      pytest dinov3-sat/tests/test_contract.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "dinov3-sat-contract"
SEED = 0
CHANNELS = 1024  # ViT-L/16 feature width
PATCH = 16


def _load_contract():
    loader = SourceFileLoader("dinov3_sat_contract", str(SCRIPT))
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    sys.modules[loader.name] = module  # dataclasses look their module up here
    loader.exec_module(module)
    return module


contract = _load_contract()


def write_image(path: Path, size: tuple[int, int], mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).convert(mode).save(path)


def write_params(path: Path, params: dict) -> Path:
    path.write_text(json.dumps(params))
    return path


@pytest.fixture
def fake_extractor(monkeypatch):
    """Stand in for extract_features.py: its file names, a random grid per image."""
    calls = []

    def run_extractor(primary, features_dir, size, device):
        calls.append((primary, features_dir, size, device))
        features_dir.mkdir(parents=True, exist_ok=True)
        (features_dir / contract.FEATURES_META).write_text(json.dumps({"input_size": size}))
        rng = np.random.default_rng(SEED)
        grid = size // PATCH
        for image in contract.find_images(primary):
            features = rng.normal(size=(grid, grid, CHANNELS)).astype(np.float16)
            np.savez_compressed(features_dir / f"{image.stem}.npz", features=features)

    monkeypatch.setattr(contract, "run_extractor", run_extractor)
    monkeypatch.setattr(contract, "visible_device", lambda: "cuda")
    return calls


# Parameters


def test_params_default_is_the_extractors_default(tmp_path):
    assert contract.read_params(write_params(tmp_path / "p.json", {})).input_size == 512


@pytest.mark.parametrize("value", ["1024", 1024])
def test_params_accept_a_listed_size_as_text_or_whole_number(tmp_path, value):
    params = contract.read_params(write_params(tmp_path / "p.json", {"input_size": value}))
    assert params.input_size == 1024


@pytest.mark.parametrize("value", ["500", "512 ", 512.0, True, None, ["512"]])
def test_params_refuse_an_unlisted_size(tmp_path, value):
    with pytest.raises(contract.Refused, match="input_size must be one of"):
        contract.read_params(write_params(tmp_path / "p.json", {"input_size": value}))


def test_params_refuse_an_unknown_name(tmp_path):
    with pytest.raises(contract.Refused, match="unknown parameters"):
        contract.read_params(write_params(tmp_path / "p.json", {"batch_size": "4"}))


@pytest.mark.parametrize("text", ["[]", "not json"])
def test_params_refuse_anything_but_an_object(tmp_path, text):
    path = tmp_path / "p.json"
    path.write_text(text)
    with pytest.raises(contract.Refused):
        contract.read_params(path)


# Inputs


def test_find_images_walks_primary_like_the_extractor(tmp_path):
    primary = tmp_path / "primary"
    write_image(primary / "chips" / "b.PNG", (32, 32))
    write_image(primary / "a.jpg", (32, 32))
    (primary / "list.txt").write_text("ignored")
    found = [p.relative_to(primary).as_posix() for p in contract.find_images(primary)]
    assert found == ["a.jpg", "chips/b.PNG"]


def test_find_images_refuses_an_empty_input(tmp_path):
    (tmp_path / "primary").mkdir()
    with pytest.raises(contract.Refused, match="no image under"):
        contract.find_images(tmp_path / "primary")


def test_find_images_refuses_inputs_whose_outputs_would_collide(tmp_path):
    primary = tmp_path / "primary"
    write_image(primary / "one" / "tile.png", (32, 32))
    write_image(primary / "two" / "tile.tif", (32, 32))
    with pytest.raises(contract.Refused, match="overwrite each other"):
        contract.find_images(primary)


def test_inspect_images_is_quiet_for_square_rgb_at_the_input_size(tmp_path):
    write_image(tmp_path / "a.png", (256, 256))
    assert contract.inspect_images([tmp_path / "a.png"], tmp_path, 256) == []


def test_inspect_images_warns_about_resizing_stretching_and_conversion(tmp_path):
    write_image(tmp_path / "wide.png", (300, 200))
    write_image(tmp_path / "gray.png", (512, 512), mode="L")
    images = sorted(tmp_path.glob("*.png"))
    warnings = contract.inspect_images(images, tmp_path, 256)
    assert "2 of 2 images were resized to 256x256 px" in warnings[0]
    assert "the largest from 512x512 px" in warnings[0]
    assert "1 of them were not square" in warnings[0]
    assert "Pillow modes L" in warnings[1]


def test_inspect_images_refuses_a_file_that_is_not_an_image(tmp_path):
    (tmp_path / "._a.png").write_bytes(b"not an image")
    with pytest.raises(contract.Refused, match="cannot read ._a.png"):
        contract.inspect_images([tmp_path / "._a.png"], tmp_path, 256)


# Preview


def test_preview_separates_two_kinds_of_patch_on_the_first_component():
    rng = np.random.default_rng(SEED)
    features = rng.normal(scale=0.01, size=(8, 8, CHANNELS))
    features[:, :4] += rng.normal(size=CHANNELS)  # left half is one surface, right another
    rgb = contract.principal_components_rgb(features)
    assert rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8
    red = rgb[..., 0].astype(int)
    assert abs(red[:, :4].mean() - red[:, 4:].mean()) > 200
    for band in range(3):
        assert rgb[..., band].min() == 0 and rgb[..., band].max() == 255


def test_preview_colors_do_not_depend_on_the_svd_sign():
    rng = np.random.default_rng(SEED)
    features = rng.normal(size=(6, 6, 32))
    # Negating every feature negates every component, which the sign convention
    # undoes; LAPACK may round the two decompositions differently by a hair.
    plus = contract.principal_components_rgb(features).astype(int)
    minus = contract.principal_components_rgb(-features).astype(int)
    assert np.abs(plus - minus).max() <= 1


def test_preview_of_uniform_features_is_black():
    rgb = contract.principal_components_rgb(np.ones((4, 4, CHANNELS), dtype=np.float16))
    assert not rgb.any()


# The whole run, with the extractor replaced


def make_job(tmp_path: Path, params: dict) -> tuple[Path, Path, Path]:
    write_image(tmp_path / "input" / "primary" / "survey" / "chip_a.png", (256, 256))
    write_image(tmp_path / "input" / "primary" / "survey" / "chip_b.tif", (300, 200))
    return tmp_path / "input", tmp_path / "output", write_params(tmp_path / "p.json", params)


def run_main(input_dir: Path, output_dir: Path, params: Path) -> int:
    return contract.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--params-json",
            str(params),
        ]
    )


def test_run_writes_features_previews_and_a_manifest_that_matches_them(tmp_path, fake_extractor):
    input_dir, output_dir, params = make_job(tmp_path, {"input_size": "256"})
    assert run_main(input_dir, output_dir, params) == 0
    assert fake_extractor[0][2:] == (256, "cuda")

    manifest = json.loads((output_dir / "run.json").read_text())
    assert manifest["contract_version"] == 1
    by_path = {entry["path"]: entry for entry in manifest["outputs"]}
    assert {path: entry["role"] for path, entry in by_path.items()} == {
        "features/_features_meta.json": "feature_metadata",
        "features/chip_a.npz": "features",
        "features/chip_b.npz": "features",
        "previews/chip_a.png": "preview",
        "previews/chip_b.png": "preview",
    }
    for path, entry in by_path.items():
        assert entry["sha256"] == hashlib.sha256((output_dir / path).read_bytes()).hexdigest()
    with Image.open(output_dir / "previews" / "chip_a.png") as preview:
        assert (preview.size, preview.mode) == ((16, 16), "RGB")
    assert len(manifest["warnings"]) == 1 and "not square" in manifest["warnings"][0]


def test_run_warns_when_no_gpu_is_visible(tmp_path, fake_extractor, monkeypatch):
    monkeypatch.setattr(contract, "visible_device", lambda: "cpu")
    input_dir, output_dir, params = make_job(tmp_path, {})
    assert run_main(input_dir, output_dir, params) == 0
    warnings = json.loads((output_dir / "run.json").read_text())["warnings"]
    assert any("ran on the CPU" in warning for warning in warnings)


def test_a_refused_run_exits_2_before_the_extractor_and_writes_no_manifest(
    tmp_path, fake_extractor, capsys
):
    input_dir, output_dir, params = make_job(tmp_path, {"input_size": "768"})
    assert run_main(input_dir, output_dir, params) == 2
    assert fake_extractor == []
    assert not (output_dir / "run.json").exists()
    assert "input_size must be one of" in capsys.readouterr().err


def test_a_missing_feature_file_fails_the_run_without_a_manifest(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(contract, "visible_device", lambda: "cuda")

    def writes_only_the_sidecar(primary, features_dir, size, device):
        features_dir.mkdir(parents=True)
        (features_dir / contract.FEATURES_META).write_text("{}")

    monkeypatch.setattr(contract, "run_extractor", writes_only_the_sidecar)
    input_dir, output_dir, params = make_job(tmp_path, {})
    assert run_main(input_dir, output_dir, params) == 1
    assert not (output_dir / "run.json").exists()
    assert "wrote no features/chip_a.npz" in capsys.readouterr().err


def test_help_needs_no_environment():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], env={}, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "--params-json" in result.stdout
