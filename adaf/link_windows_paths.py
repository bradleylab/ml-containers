"""Make upstream's Windows model paths resolve on Linux, without patching source.

adaf/adaf_inference.py hardcodes its model paths as Windows raw strings:

    "barrow": r".\\ml_models\\OD_barrow.tar"

and then joins them with Path(__file__).parent. On Windows that resolves. On
Linux a backslash is an ordinary filename character, so the join produces a
SINGLE file called

    /opt/adaf/adaf/.\\ml_models\\OD_barrow.tar

which does not exist, and every stock code path fails. This is in the source,
not just the install docs.

Two ways to run this on Linux, and this image supports both:

  1. THE CLEAN ROUTE, and what our wrapper uses. Both entry points take a
     `custom_model` argument, and `models["custom"]` is that value. Because
     Path(a) / Path(b) returns b when b is absolute, passing an absolute POSIX
     path works exactly as intended:

         run_aitlas_segmentation(["custom"], tiles, custom_model="/opt/adaf-weights/...")

     This is a documented parameter, not a workaround, so it survives upstream
     updates.

  2. THE COMPATIBILITY ROUTE, which this script builds. The stock notebooks and
     main_routine() do NOT go through custom_model -- they index the dict by
     label. So we also create symlinks whose FILENAMES literally contain
     backslashes, letting that code find the weights unmodified.

Deliberately NOT a fork: no upstream file is edited, so `git pull` stays clean.
When upstream fixes the separators these symlinks become inert rather than
breaking, and the POSIX-named copies in ml_models/ are what it will look for.
"""

import os
from pathlib import Path

WEIGHTS = Path("/opt/adaf-weights")
PKG = Path("/opt/adaf/adaf")

MODELS = [
    "OD_barrow.tar",
    "OD_enclosure.tar",
    "OD_ringfort.tar",
    "OD_AO.tar",
    "barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
    "enclosure_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
    "ringfort_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
    "AO_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
]

missing = [m for m in MODELS if not (WEIGHTS / m).is_file()]
if missing:
    raise SystemExit(f"weights not staged: {missing}")

# 1. The layout upstream intends, and will use once the separators are fixed.
posix_dir = PKG / "ml_models"
posix_dir.mkdir(parents=True, exist_ok=True)
for name in MODELS:
    link = posix_dir / name
    if not link.exists():
        link.symlink_to(WEIGHTS / name)

# 2. The literal backslash filenames today's source actually asks for.
for name in MODELS:
    weird = PKG / f".\\ml_models\\{name}"
    if not os.path.lexists(weird):
        weird.symlink_to(WEIGHTS / name)

resolved = sum(1 for name in MODELS if (posix_dir / name).resolve().is_file())
weird_ok = sum(1 for name in MODELS if Path(os.path.realpath(PKG / f".\\ml_models\\{name}")).is_file())
print(f"posix-named links resolving   : {resolved}/{len(MODELS)}")
print(f"backslash-named links resolving: {weird_ok}/{len(MODELS)}")
if resolved != len(MODELS) or weird_ok != len(MODELS):
    raise SystemExit("not every model path resolves")
print("model paths wired for both calling conventions")
