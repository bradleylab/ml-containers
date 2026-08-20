# hylite

`hylite` + `hklearn` — hyperspectral mineral mapping for drillcore and outcrop.
HIF Freiberg. MIT.

## This one ships no weights

Every other image in this catalog answers a question out of the box. This one
does not. Minimum-wavelength mapping is **fitted per scan**, and `hklearn`'s
classifiers train on your own labelled core. "Running it" means fitting on your
data.

That is the state of the field, not an oversight: the 2026-08 coverage triage
found no pretrained hyperspectral mineral model with released weights anywhere,
and filed the area under genuine voids.

## What it closes

It does not duplicate `raman-classifier` or `xrd-classifier`. Those identify a
mineral from a single point spectrum. This maps mineralogy **spatially** across
a hyperspectral scan — correction, hull removal, minimum-wavelength mapping,
mineral indices, then ML on top.

## Usage

```bash
docker run --rm -v "$PWD":/work ghcr.io/bradleylab/hylite:v1 python - <<'PY'
import hylite
from hylite import io
img = io.load('/work/corescan.hdr')
img.set_wavelengths(img.get_wavelengths())
# hull-correct, then fit minimum wavelength over the SWIR mineral window
PY
```

CPU only — neither package uses CUDA.

## Pins

`hylite==1.36` from PyPI; `hklearn` is not on PyPI and is installed from a
pinned commit (`4f79fb75577d`). numpy is held below 2 because hylite's
spectral/scipy path has not been validated against the numpy 2 ABI break.

## Verification

The build smoke test constructs a synthetic hyperspectral cube with a known
absorption feature and exercises the fitting path, rather than importing the
package — an import check would not catch a broken scipy/numpy pairing, which
is where this stack actually breaks.

**Not yet executed on Compute2.**
