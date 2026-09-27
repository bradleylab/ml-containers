"""Stage the NASA-IBM Lunar Foundation Model weights and SomBench downstream
benchmarks, each pinned to the Hugging Face revision recorded here.

Run inside any image that has huggingface_hub, with the destination mounted at
/stage; SMOKE.md gives the Compute2 job. The layout it produces is the one
`lfm-link` expects when the destination is later mounted at /weights.

The nine VQ-VAE tokenizers (10.3 GB) are skipped: they serve any-to-any
generation, not fine-tuning or downstream inference.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path("/stage")
ORG = "nasa-ibm-ai4science"

MODELS = {
    "NASA-IBM-Lunar-Foundation-Model": "b657bdfe6fc4c0445d918f77a2d7b919805f3bce",
    "Crater-Detection-NASA-IBM-Lunar-Foundation-Model": "f09ebeae2099c28f09bf4060f56d3e327a6192de",
    "IMP-Segmentation-NASA-IBM-Lunar-Foundation-Model": "10a09ffae35ef5712c40c2a1fc41ec82d4881ea6",
    "Ice-Prospectivity-NASA-IBM-Lunar-Foundation-Model": "fb0510d0d676ae7bfae9acd88219b05449c769ee",
}
# Directory names are the ones the upstream configs expect under data/.
DATASETS = {
    "Sombench-Ice-Prospectivity-Regression": ("78040503ceaa68107829f48114774112e8ee940d", "prospectivity_dataset"),
    "Sombench-IMP-Segmentation": ("1418851011f4b1d4fd8140a8d765b17fda86ebc0", "imp_dataset"),
    "Sombench-NAC-Crater-Detection": ("79419eb3486b2dd9ad09a283802e0bb8d75c767d", "nac_craters_dataset"),
    "Sombench-WAC-Crater-Detection": ("20f800becb64a0c7ad314652a683f97f0281fe67", "wac_craters_dataset"),
}

for name, rev in MODELS.items():
    dest = ROOT / "hf" / name
    patterns = None
    if name == "NASA-IBM-Lunar-Foundation-Model":
        patterns = ["backbone/*", "config.json", "README.md", "NI_LFM_Technical_Report.pdf"]
    snapshot_download(f"{ORG}/{name}", revision=rev, local_dir=dest, allow_patterns=patterns)
    print(f"model   {name} @ {rev[:7]} -> {dest}", flush=True)

for name, (rev, dirname) in DATASETS.items():
    dest = ROOT / "downstream_dataset" / dirname
    snapshot_download(f"{ORG}/{name}", repo_type="dataset", revision=rev, local_dir=dest)
    print(f"dataset {name} @ {rev[:7]} -> {dest}", flush=True)

total = 0
for p in sorted(ROOT.rglob("*")):
    if p.is_file() and ".cache" not in p.parts:
        total += p.stat().st_size
        if p.stat().st_size > 100e6:
            print(f"  {p.stat().st_size:>14,d}  {p.relative_to(ROOT)}")
print(f"staged bytes (excluding .cache): {total:,d}")
