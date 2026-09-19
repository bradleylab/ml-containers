# tabfm

Zero-shot classification and regression on tabular data.

Google Research (2026). Upstream: [google-research/tabfm](https://github.com/google-research/tabfm).
Weights: [google/tabfm-1.0.0-pytorch](https://huggingface.co/google/tabfm-1.0.0-pytorch).

## What this is

TabFM reads your training rows as *context* and predicts on new rows in a single
forward pass. There is no per-dataset training and no hyperparameter search: you
hand it a table and it answers. It was pretrained on synthetic datasets generated
from structural causal models, and it takes mixed numeric and categorical columns
as they come.

The API is scikit-learn shaped, so it drops into an existing tabular workflow as
an alternative to a tuned gradient-boosted tree.

| Checkpoint | Task |
|---|---|
| `classification/` | classification, up to 10 classes |
| `regression/` | regression |

## The license, first

**Non-commercial and non-production.** The TabFM Non-Commercial License v1.0
permits academic research, internal benchmarking and experimentation. Three
restrictions matter before anyone runs this:

1. **Weights may not be redistributed** (restriction 3b). That is why this image
   does not bake them, and why you must not copy a populated cache into a shared
   or published artifact.
2. **Outputs are restricted too** (restriction 3a). Predictions may not be used
   "in commercial decision-making, client deliverables, or paid products/
   services". Contract and consulting work is out, not only selling the model.
3. **Derivatives inherit everything.** A fine-tuned TabFM carries the same terms.

Full text ships at `/opt/licenses/LICENSE.tabfm.md`. The source code is
Apache-2.0; the weights are not, and the weights are what bind you.

## Weights are not baked

The image carries the environment only. Fetch the checkpoints at run time into a
mounted cache — the repository is ungated, so no token is needed.

```bash
export HF_HOME=/path/to/cache
python -c "
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
tabfm_v1_0_0.load(model_type='classification')
"
```

That pulls 13.1 GB: 6.56 GB for classification, 6.59 GB for regression. On
Compute2, stage the cache from a login node first and then run offline with
`HF_HUB_OFFLINE=1`; see [`SMOKE.md`](SMOKE.md).

## Usage

```python
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

model = tabfm_v1_0_0.load(model_type="classification")
clf = TabFMClassifier(model=model)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)
```

Regression is the same shape:

```python
from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

model = tabfm_v1_0_0.load(model_type="regression")
reg = TabFMRegressor(model=model)
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
```

`fit` stores the context rather than training parameters, which is why it
returns immediately and why the size of the training table drives inference cost.

## Stack

- Base `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` (Python 3.11; tabfm
  requires 3.11 or newer).
- `tabfm[pytorch]==1.0.1` installed under a constraint pinning torch to the
  base image's CUDA build, so the resolve cannot swap in a CPU wheel.
- GPU is optional. The model runs on CPU, but a 6.5 GB checkpoint doing
  in-context learning over a large table is what the H100 is for.

Pull: `ghcr.io/bradleylab/tabfm:v1`
