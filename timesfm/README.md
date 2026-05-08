# timesfm

[TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
(Das et al., ICML 2024) is a decoder-only foundation model for
univariate time-series forecasting from Google Research. The 2.5
release (Sept 2025) is 200M parameters, supports up to 16k context,
and ships an optional 30M continuous-quantile head.

This container ships the upstream `timesfm` package from
[`google-research/timesfm`](https://github.com/google-research/timesfm)
at a pinned commit. Weights are NOT baked — pulled lazily from HF Hub
on first call.

CPU-primary. The 200M model + Apache-2.0 weights mean short-horizon
forecasts run comfortably on a laptop. Add a CUDA variant later if a
batch-forecasting workload at scale lands.

## Image tag

`ghcr.io/bradleylab/timesfm:v1` (also `:latest`, `:torch2.5-cpu`)

## Stack

- Base: `python:3.11-slim`
- PyTorch 2.5.1 CPU wheels (manylinux + manylinux_aarch64)
- `timesfm` from GitHub at pinned commit `d720daa67865`
  (2026-04-15) — installed via `pip install ".[torch]"`. PyPI's
  `timesfm 1.3.0` is the v1/v2 archive; TimesFM 2.5 is GitHub-only
  as of this image.
- `numpy`, `safetensors`, `huggingface_hub` (transitive)

`WANDB_MODE=offline` set in ENV so the import path doesn't try to
phone home.

## Weights

Not baked. Pull from HF Hub on first call:

| Variant | HF Hub repo | Notes |
|---|---|---|
| 2.5 200M (PyTorch native) | [`google/timesfm-2.5-200m-pytorch`](https://huggingface.co/google/timesfm-2.5-200m-pytorch) | Loaded via `timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)` — the canonical interface |
| 2.5 200M (HF Transformers) | [`google/timesfm-2.5-200m-transformers`](https://huggingface.co/google/timesfm-2.5-200m-transformers) | Loaded via `transformers.AutoModel.from_pretrained(...)` for LoRA fine-tuning with PEFT |

`HF_HOME=/opt/hf-cache` — bind-mount that path for cross-run weight
persistence.

## Quickstart

```bash
docker run --rm \
  -v /path/to/data:/work \
  -v /shared/hf-cache:/opt/hf-cache \
  ghcr.io/bradleylab/timesfm:v1 bash
```

Inside the container:

```python
import torch
import timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(timesfm.ForecastConfig(
    max_context=1024,
    max_horizon=256,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    force_flip_invariance=True,
    infer_is_positive=True,
    fix_quantile_crossing=True,
))

# Forecast 12 steps ahead from one or more univariate series
import numpy as np
series = np.sin(np.linspace(0, 6 * np.pi, 1024)).astype(np.float32)
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[series.tolist()],
)
print("point shape:", point_forecast.shape)         # (1, 12)
print("quantile shape:", quantile_forecast.shape)   # (1, 12, 9)
```

For LoRA fine-tuning via HuggingFace Transformers + PEFT, see
upstream
[`timesfm-forecasting/examples/finetuning/`](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning).

## Lab use cases

- **Hydrology / streamflow forecasting** — TimesFM is the natural
  zero-shot fallback for sites without enough history to fine-tune
  a CAMELS-style LSTM (cf. `bradleylab/neuralhydrology` for the
  full LSTM toolkit).
- **Climate reanalysis pixel-time-series** — quick "next 12 months"
  predictions on any ERA5 / MERRA-2 single-pixel signal without
  per-pixel training.
- **Eddy-covariance / soil-moisture / met-station gap-filling**
  short-horizon imputation using observed history.

For panels of thousands of series, expect minutes-to-tens-of-minutes
on CPU; a GPU variant of this container is a sensible follow-up
when the batch size makes that meaningful.

## License

Code: Apache-2.0 (this Dockerfile, `google-research/timesfm`).
Weights: Apache-2.0 (per the HF model card on
[`google/timesfm-2.5-200m-pytorch`](https://huggingface.co/google/timesfm-2.5-200m-pytorch)).

## References

- Das et al. (2024) "A decoder-only foundation model for time-series
  forecasting", ICML 2024
  [arXiv:2310.10688](https://arxiv.org/abs/2310.10688).
- Upstream code:
  [google-research/timesfm](https://github.com/google-research/timesfm)
  (model code, fine-tuning examples, agent skill).
- HF Collection:
  [TimesFM Hugging Face Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6).
