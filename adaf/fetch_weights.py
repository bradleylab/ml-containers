"""Fetch the ADAF model weights from Zenodo at build time, with retries.

Eight TAR files, 5.52 GB total, record 15848663 (concept DOI 10.5281/zenodo.15848662).
They are NOT extracted -- AiTLAS loads the .tar directly, and upstream is explicit
that extracting them breaks loading.

Licence: CC-BY-SA-4.0, which is NOT the Apache-2.0 of the code. See LICENSE.weights.md
in this directory, which is shipped inside the image because CC-BY-SA requires the
notice to travel with the work.
"""

import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RECORD = "15848663"
BASE = f"https://zenodo.org/records/{RECORD}/files"
DEST = Path("/opt/adaf-weights")
ATTEMPTS = 5

# Sizes are from the Zenodo record and are asserted after download: a truncated
# transfer that still exits 0 is the failure mode a bare urlretrieve allows.
FILES = {
    "OD_barrow.tar": 517_754_880,
    "OD_enclosure.tar": 517_754_880,
    "OD_ringfort.tar": 517_754_880,
    "OD_AO.tar": 517_754_880,
    "barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar": 861_460_480,
    "enclosure_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar": 861_460_480,
    "ringfort_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar": 861_460_480,
    "AO_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar": 861_460_480,
}

# Zenodo reports sizes rounded in its own UI; allow a small tolerance rather than
# asserting an exact byte count we have not independently confirmed.
SIZE_TOLERANCE = 0.02


def fetch(name: str, expected: int) -> None:
    target = DEST / name
    url = f"{BASE}/{name}"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            urllib.request.urlretrieve(url, target)
            got = target.stat().st_size
            if abs(got - expected) / expected > SIZE_TOLERANCE:
                raise OSError(f"size {got} differs from expected {expected} by >2%")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            print(f"  {name}: {got / 1e6:.1f} MB  sha256={digest[:16]}")
            return
        except (urllib.error.URLError, OSError) as exc:
            print(f"  attempt {attempt} for {name} failed: {type(exc).__name__}: {exc}"[:300])
            target.unlink(missing_ok=True)
            if attempt == ATTEMPTS:
                sys.exit(f"failed to fetch {name} after {ATTEMPTS} attempts")
            time.sleep(20 * attempt)


DEST.mkdir(parents=True, exist_ok=True)
for filename, size in FILES.items():
    fetch(filename, size)
print(f"all {len(FILES)} weight files staged in {DEST}")
