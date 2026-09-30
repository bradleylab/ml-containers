"""Stage the Aurora weather checkpoints, their static fields and upstream's
small-model reference batch from microsoft/aurora, pinned to one Hugging Face
revision.

Run inside the aurora image with the destination mounted at /stage; SMOKE.md
gives the Compute2 job. The files land flat in /stage, which is the layout the
verification scripts expect when the same directory is mounted at /weights.

The revision is the one microsoft-aurora 2.0.1 itself pins for these three
checkpoints (default_checkpoint_revision in aurora/model/aurora.py), so a
staged file is byte-identical to what model.load_checkpoint() would fetch.

Not staged, and why:
  aurora-0.25-12h-pretrained.ckpt  12-hour step; the package pins it to a different
                                    revision, and the weather check is 6-hourly
  aurora-0.1-finetuned.ckpt        needs IFS HRES analysis at 0.1°, 1801 x 3600
  aurora-0.25-v1.5*.ckpt           needs 19 surface inputs and 36 static fields;
                                    WeatherBench2's HRES T0 store carries 5 of the
                                    inputs and none of the static fields
  aurora-0.25-wave.ckpt            needs HRES-WAM wave analysis
  aurora-0.4-air-pollution.ckpt    needs CAMS analysis, which requires a
                                    Copernicus Atmosphere Data Store account
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path("/stage")
REPO = "microsoft/aurora"
REVISION = "0be7e57c685dac86b78c4a19a3ab149d13c6a3dd"

FILES = [
    "aurora-0.25-pretrained.ckpt",  # ERA5 and any source without a fine-tuned model
    "aurora-0.25-finetuned.ckpt",  # IFS HRES T0
    "aurora-0.25-small-pretrained.ckpt",  # upstream's debugging model
    "aurora-0.25-static.pickle",  # lsm, slt, z for all three
    "aurora-0.25-small-pretrained-test-input.pickle",  # upstream's reference batch
    "aurora-0.25-small-pretrained-test-output.pickle",  # and its expected prediction
]

for name in FILES:
    path = Path(hf_hub_download(REPO, name, revision=REVISION, local_dir=ROOT))
    print(f"{path.stat().st_size:>15,d}  {name} @ {REVISION[:7]}", flush=True)

total = sum((ROOT / name).stat().st_size for name in FILES)
print(f"staged bytes: {total:,d}")
