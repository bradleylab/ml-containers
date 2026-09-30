"""Open a fraXteX ONNX model and hold it to the input contract it declares.

Each published model records its contract in its own ``metadata_props``: band
count and order, value range, tile size, and whether normalization happens in
the graph. The two modality directories disagree on almost all of it (``rgb/``
takes raw 0-255 and normalizes internally; ``rgbdem/`` takes 0-1 and does
not), and feeding either the other's input gives plausible masks rather than an
error. So nothing here is hard-coded per model: the contract is read from the
file, cross-checked against the graph's own input and output shapes, and any
value this code does not know how to honor stops the run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_DIR = Path(os.environ.get("FRAXTEX_MODEL_DIR", "/opt/fraxtex/models"))
MODELS = {
    "unet-rgb": "rgb/unet-rgb.onnx",
    "sam2-rgb": "rgb/sam2-rgb.onnx",
    "unet-rgbdem": "rgbdem/unet-rgbdem.onnx",
    "segformer-rgbdem": "rgbdem/segformer-rgbdem.onnx",
}

# The 15 keys every published model carries (model card, "Input contracts").
CONTRACT_KEYS = (
    "model.name", "preprocessing.in_graph", "preprocessing.normalize",
    "input.dtype", "input.layout", "input.channel_order", "input.channels",
    "input.range", "input.dynamic_hw", "input.fixed_size",
    "output.dtype", "output.channels", "output.type", "output.range",
    "citation.doi",
)
RANGE_MAX = {"0-1": 1.0, "0-255": 255.0}
CHANNEL_ORDER = {3: "RGB", 4: "RGB+DEM"}
# The model card states the dynamic-size U-Net accepts "any multiple of 16";
# its exported output shape, 16*(height//16), silently crops anything else.
DYNAMIC_TILE_MULTIPLE = 16
# A sigmoid output that has been through float32 arithmetic can sit a few ulp
# outside [0, 1]; anything further out is not a probability.
OUTPUT_RANGE_TOLERANCE = 1e-5

CUDA_EP = "CUDAExecutionProvider"
CPU_EP = "CPUExecutionProvider"


class ContractError(ValueError):
    """The model's declared contract is missing, inconsistent, or unsupported."""


@dataclass(frozen=True)
class Contract:
    name: str
    channels: int
    value_max: float  # inputs are fed in [0, value_max]
    normalize: str  # applied inside the graph; the caller never normalizes
    tile: int | None  # fixed square tile, or None for any multiple of 16
    doi: str
    metadata: dict[str, str]

    @property
    def uses_dem(self) -> bool:
        return self.channels == 4


def resolve_model(name_or_path: str) -> Path:
    """Map a model name (``unet-rgb`` ...) or a path to an existing .onnx file."""
    path = MODEL_DIR / MODELS[name_or_path] if name_or_path in MODELS else Path(name_or_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; choose one of {sorted(MODELS)} or give a path to an .onnx file"
        )
    return path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def read_contract(session: ort.InferenceSession) -> Contract:
    """Parse and validate the contract in ``metadata_props`` against the graph."""
    meta = dict(session.get_modelmeta().custom_metadata_map or {})
    missing = [k for k in CONTRACT_KEYS if k not in meta]
    _require(not missing, f"model metadata lacks contract keys {missing}")

    (graph_in,) = session.get_inputs()
    (graph_out,) = session.get_outputs()
    in_shape, out_shape = graph_in.shape, graph_out.shape

    _require(meta["input.dtype"] == "float32" and graph_in.type == "tensor(float)",
             f"input dtype {meta['input.dtype']} / {graph_in.type} is not float32")
    _require(meta["input.layout"] == "NCHW" and len(in_shape) == 4,
             f"input layout {meta['input.layout']} with rank {len(in_shape)} is not NCHW")
    channels = int(meta["input.channels"])
    _require(in_shape[1] == channels,
             f"metadata declares {channels} channels, graph input has {in_shape[1]}")
    _require(CHANNEL_ORDER.get(channels) == meta["input.channel_order"],
             f"channel order {meta['input.channel_order']} for {channels} channels is unsupported")

    _require(meta["input.range"] in RANGE_MAX, f"input range {meta['input.range']} is unsupported")
    in_graph = meta["preprocessing.in_graph"]
    _require(in_graph in ("true", "false"), f"preprocessing.in_graph={in_graph} is not a boolean")
    # In-graph preprocessing means raw 0-255 in; none means the caller scales
    # to 0-1 and applies no mean/std (the repo's CONTEXT.md glossary). Any other
    # pairing would need statistics this code does not have.
    _require((in_graph == "true") == (meta["input.range"] == "0-255"),
             f"in_graph={in_graph} contradicts input.range={meta['input.range']}")
    _require(in_graph == "true" or meta["preprocessing.normalize"] == "none",
             f"caller-side normalization '{meta['preprocessing.normalize']}' is not supported")

    tile = _read_tile(meta, in_shape)

    _require(meta["output.dtype"] == "float32" and graph_out.type == "tensor(float)",
             "output is not float32")
    _require(meta["output.channels"] == "1" and out_shape[1] == 1, "output is not single-channel")
    _require(meta["output.type"] == "sigmoid_probability" and meta["output.range"] == "0-1",
             f"output {meta['output.type']} in {meta['output.range']} is not a probability")

    return Contract(
        name=meta["model.name"],
        channels=channels,
        value_max=RANGE_MAX[meta["input.range"]],
        normalize=meta["preprocessing.normalize"],
        tile=tile,
        doi=meta["citation.doi"],
        metadata=meta,
    )


def _read_tile(meta: dict[str, str], in_shape: list) -> int | None:
    dims = in_shape[2:]
    if meta["input.dynamic_hw"] == "true":
        _require(meta["input.fixed_size"] == "none", "dynamic model also declares a fixed size")
        _require(all(isinstance(d, str) for d in dims), f"dynamic model has fixed graph dims {dims}")
        return None
    _require(meta["input.dynamic_hw"] == "false", f"input.dynamic_hw={meta['input.dynamic_hw']}")
    height, width = (int(v) for v in meta["input.fixed_size"].split("x"))
    _require(list(dims) == [height, width], f"fixed_size {height}x{width} but graph dims {dims}")
    _require(height == width, f"non-square fixed size {height}x{width} is unsupported")
    return height


def check_tile(contract: Contract, tile: int) -> None:
    """Refuse a tile size the model cannot take unchanged."""
    if contract.tile is not None and tile != contract.tile:
        raise ContractError(f"{contract.name} takes {contract.tile}x{contract.tile} tiles only, not {tile}")
    if contract.tile is None and tile % DYNAMIC_TILE_MULTIPLE:
        raise ContractError(f"tile {tile} is not a multiple of {DYNAMIC_TILE_MULTIPLE}")


def open_session(path: Path, device: str) -> tuple[ort.InferenceSession, str]:
    """Create a session on CUDA when asked or available, and report which ran.

    ``device`` is ``auto`` (CUDA if it initializes, else CPU), ``cuda`` (fail
    rather than fall back) or ``cpu``. The CUDA and cuDNN libraries are pip
    wheels, so they are preloaded from the nvidia site-packages first
    (onnxruntime.preload_dlls, available since 1.21). ONNX Runtime falls back to
    CPU on its own when the CUDA provider cannot start, e.g. on a node with no
    GPU, so the provider actually in use is read back rather than assumed.
    """
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device must be auto, cuda or cpu, not {device}")
    has_cuda_build = CUDA_EP in ort.get_available_providers()
    if device == "cuda" and not has_cuda_build:
        raise RuntimeError(f"this onnxruntime build has no {CUDA_EP}")
    providers = [CPU_EP]
    if device != "cpu" and has_cuda_build:
        ort.preload_dlls(cuda=True, cudnn=True, msvc=False, directory="")
        providers = [CUDA_EP, CPU_EP]
    session = ort.InferenceSession(str(path), providers=providers)
    used = session.get_providers()[0]
    if device == "cuda" and used != CUDA_EP:
        raise RuntimeError(f"CUDA was required but the session runs on {used}")
    return session, used


def predict(session: ort.InferenceSession, batch: np.ndarray) -> np.ndarray:
    """Run an (N, C, H, W) float32 batch; return (N, H, W) probabilities."""
    (graph_in,) = session.get_inputs()
    (out,) = session.run(None, {graph_in.name: batch})
    if out.shape[0] != batch.shape[0] or out.shape[2:] != batch.shape[2:]:
        raise ContractError(f"output shape {out.shape} does not match input {batch.shape}")
    lo, hi = float(out.min()), float(out.max())
    if lo < -OUTPUT_RANGE_TOLERANCE or hi > 1 + OUTPUT_RANGE_TOLERANCE:
        raise ContractError(f"output spans [{lo}, {hi}], outside the declared 0-1")
    return np.clip(out[:, 0], 0.0, 1.0)


def describe(contract: Contract) -> str:
    tile = f"{contract.tile}x{contract.tile} fixed" if contract.tile else \
        f"any multiple of {DYNAMIC_TILE_MULTIPLE}"
    return (f"{contract.name}: {contract.channels} ch {contract.metadata['input.channel_order']}, "
            f"feed 0-{contract.value_max:g}, normalization in graph: "
            f"{contract.metadata['preprocessing.in_graph']} ({contract.normalize}), tile {tile}, "
            f"doi {contract.doi}")
