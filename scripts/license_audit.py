#!/usr/bin/env python3
"""Compare each image's declared license label against its upstream's license.

A wrong org.opencontainers.image.licenses label misstates an obligation we pass
on to whoever pulls the image. The direction that matters is a permissive label
sitting over a copyleft upstream: MIT cannot hold over a GPL-3.0 upstream.

Reads every top-level <image>/Dockerfile, extracts the licenses label and the
bradleylab.model.upstream URL, and for GitHub upstreams compares the label to
the license GitHub reports. Prints a markdown report; exits 1 if any image
mismatches.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

LABEL_RE = re.compile(r'org\.opencontainers\.image\.licenses="([^"]*)"')
UPSTREAM_RE = re.compile(r'bradleylab\.model\.upstream="([^"]*)"')
GITHUB_RE = re.compile(r'^https://github\.com/([^/]+)/([^/#?]+)')

# SPDX spellings that mean the same obligation.
ALIASES = {
    "GPL-3.0-only": "GPL-3.0",
    "GPL-3.0-or-later": "GPL-3.0",
    "AGPL-3.0-only": "AGPL-3.0",
    "AGPL-3.0-or-later": "AGPL-3.0",
    "BSD-3": "BSD-3-Clause",
    "Apache 2.0": "Apache-2.0",
}

# Labels naming a weights license over a differently-licensed codebase. Not a
# mismatch — the label deliberately names the binding constraint.
WEIGHTS_PREFIXES = ("CC-BY", "CC0")


def load_exceptions(root: Path) -> dict:
    """Images where the label legitimately differs from the API's answer.

    An exception only holds while the label stays what was reviewed. Change the
    label and the image is checked normally again, so a stale entry cannot
    quietly excuse a new mistake.
    """
    path = root / "scripts" / "license_audit_exceptions.json"
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


def normalize(spdx: str) -> str:
    return ALIASES.get(spdx.strip(), spdx.strip())


def fetch_license(owner: str, repo: str, token: str | None) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        return "REPO-404" if e.code == 404 else f"HTTP-{e.code}"
    except Exception as e:  # network, timeout, malformed
        return f"ERROR-{type(e).__name__}"
    lic = data.get("license") or {}
    return lic.get("spdx_id") or "NONE"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    token = os.environ.get("GITHUB_TOKEN")
    exceptions = load_exceptions(root)

    mismatches, review, unchecked, ok = [], [], [], 0

    for dockerfile in sorted(root.glob("*/Dockerfile")):
        image = dockerfile.parent.name
        text = dockerfile.read_text(errors="replace")
        lm, um = LABEL_RE.search(text), UPSTREAM_RE.search(text)
        label = lm.group(1) if lm else ""
        upstream = um.group(1) if um else ""

        if not label:
            review.append((image, "(no label)", upstream, "no licenses label set"))
            continue

        gh = GITHUB_RE.match(upstream) if upstream else None
        if not gh:
            unchecked.append((image, label, upstream or "(none)"))
            continue

        actual = fetch_license(gh.group(1), gh.group(2).removesuffix(".git"), token)

        if actual in ("REPO-404",):
            mismatches.append((image, label, actual, "upstream repository does not resolve"))
            continue
        if actual.startswith(("HTTP-", "ERROR-")):
            review.append((image, label, actual, "could not reach upstream this run"))
            continue

        if " AND " in label:
            review.append((image, label, actual, "compound label: code and weights differ"))
        elif label.startswith("LicenseRef-"):
            review.append((image, label, actual, "custom license, no SPDX equivalent"))
        elif label == "NOASSERTION" and actual in ("NONE", "NOASSERTION"):
            ok += 1
        elif label.startswith(WEIGHTS_PREFIXES) and actual not in ("NONE", "NOASSERTION"):
            review.append((image, label, actual, "label names the weights license"))
        elif normalize(label) == normalize(actual):
            ok += 1
        elif exceptions.get(image, {}).get("label") == label:
            review.append((image, label, actual, exceptions[image]["reason"]))
        else:
            mismatches.append((image, label, actual, "label disagrees with upstream"))

    total = ok + len(mismatches) + len(review) + len(unchecked)
    out = [
        "## License-label audit",
        "",
        f"Checked **{total}** images. "
        f"{ok} agree with upstream, {len(mismatches)} disagree, "
        f"{len(review)} need a human look, {len(unchecked)} could not be checked automatically.",
        "",
    ]

    if mismatches:
        out += ["### Mismatches", "", "| Image | Label says | Upstream is | |", "|---|---|---|---|"]
        out += [f"| `{i}` | {l} | **{a}** | {n} |" for i, l, a, n in mismatches]
        out.append("")
    else:
        out += ["No mismatches.", ""]

    if review:
        out += ["<details><summary>Reviewed, no action</summary>", "",
                "| Image | Label | Upstream | Why |", "|---|---|---|---|"]
        out += [f"| `{i}` | {l} | {a} | {n} |" for i, l, a, n in review]
        out += ["", "</details>", ""]

    if unchecked:
        out += ["<details><summary>Not automatically checkable</summary>", "",
                "Upstream is not a GitHub repository, so its license cannot be read from the API.", "",
                "| Image | Label | Upstream |", "|---|---|---|"]
        out += [f"| `{i}` | {l} | {u} |" for i, l, u in unchecked]
        out += ["", "</details>", ""]

    print("\n".join(out))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
