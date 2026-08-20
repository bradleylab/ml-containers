"""Cache the TRELLIS.2 checkpoint at build time, with retries.

A 4 B checkpoint pulled from a CI runner hits Hugging Face rate limiting often
enough that a single unguarded `snapshot_download` is not a reliable build step
— that is what failed the first attempt at this image (HTTP 429).
`snapshot_download` resumes, so each retry continues rather than restarting.
"""

import sys
import time

from huggingface_hub import snapshot_download

REPO = "microsoft/TRELLIS.2-4B"
ATTEMPTS = 6

for attempt in range(1, ATTEMPTS + 1):
    try:
        snapshot_download(REPO, max_workers=2)
        print(f"weights cached on attempt {attempt}")
        break
    except Exception as exc:  # noqa: BLE001 — any transport failure is retryable here
        print(f"attempt {attempt} failed: {type(exc).__name__}: {exc}"[:300])
        if attempt == ATTEMPTS:
            sys.exit(f"weight download failed after {ATTEMPTS} attempts")
        time.sleep(30 * attempt)
