# geoclip

Worldwide image geolocalization via CLIP-aligned GPS coordinates —
[Vivanco Cepeda, Nayak, Shah (NeurIPS 2023)](https://arxiv.org/abs/2309.16020).
Given an RGB photo, returns the top-k predicted `(lat, lon)` locations
and their probabilities. SOTA on Im2GPS3k, YFCC26k, GWS15k, NUS-Wide.

## Image tag

`ghcr.io/bradleylab/geoclip:v1` (also `:latest`, `:torch2.5-cpu`).

Multi-arch: `linux/amd64` + `linux/arm64`. Apple-Silicon Macs pull
native arm64; Compute2 / EC2 pull amd64.

## Stack

- Base: `python:3.11-slim`
- PyTorch 2.5.1 + torchvision 0.20.1 (CPU wheels)
- `geoclip>=1.2` ([Vivanco Cepeda et al. 2023](https://arxiv.org/abs/2309.16020), MIT)
- `huggingface_hub>=0.25`, `Pillow>=10`

## Architecture

- **Image encoder.** CLIP ViT-L/14 (~890 MB) — produces an image
  embedding from the input photo.
- **Location encoder.** Small MLP that projects GPS `(lat, lon)`
  pairs into the same embedding space as the image encoder.
- **GPS gallery.** A fixed 100,000-point set of candidate locations
  worldwide (`coordinates_100K.csv`), bundled with the pip package.
  At inference time, the image embedding is compared against the
  encoded gallery; top-k by softmax over cosine similarity.

## Weights

Baked at build time. The `geoclip` pip package does **not** ship the
backbone weights — `GeoCLIP(from_pretrained=True)` fetches them from
Hugging Face Hub on instantiation. To make the container offline-
capable, we instantiate `GeoCLIP()` during the build, which populates
`HF_HOME=/opt/hf-cache` with the full set of weights. The runtime
container is therefore self-contained — no network access required
to call `predict()`.

This is **different from `remoteclip:v2`** (which keeps weights at
runtime via mounted HF cache). The choice is pragmatic: GeoCLIP's
weights are smaller and the model is intended for one-shot photo
QA, not large batch processing where a shared cache is useful.

## Inference

```bash
docker run --rm \
  -v "$PWD:/work" \
  ghcr.io/bradleylab/geoclip:v1 \
  python /opt/scripts/geoclip_predict.py \
    --image /work/photo.jpg \
    --top-k 5 \
    --out /work/preds.csv
```

Output CSV columns: `rank, lat, lon, prob`. Probabilities are softmax
over the GPS gallery — they sum to 1.0 across the full 100K-point
gallery (so even the top-1 may be small, e.g. 0.001-0.05, when the
photo's likely region covers many gallery points).

## Inputs

- Any RGB image readable by Pillow (JPEG, PNG, GeoTIFF via Pillow +
  rasterio if you add it). The image encoder uses standard CLIP
  preprocessing (224×224 center crop, normalize with CLIP's
  ImageNet stats).
- Photos with EXIF GPS will have that GPS *ignored* — predictions
  come purely from image content.

## Use cases

- **Geo-tagged dataset QA.** Detect photos with implausible EXIF
  GPS (e.g. "this photo claims Missouri but the model predicts
  coastal Oregon"). Quick sniff test for ingestion pipelines.
- **Locating photos with missing/stripped EXIF.** Order-of-magnitude
  geolocation when no other metadata exists.
- **Provenance / deduplication.** Photos likely to come from the
  same general area cluster in the model's location embedding space.

GeoCLIP is *not* designed for sub-meter or even sub-kilometer accuracy.
The original paper reports country/continent accuracy at ~1km tolerance
for street-level scenes, lower for landscape/wilderness. Treat
predictions as a hypothesis, not a measurement.

## Run on Compute2

CPU inference is sufficient — single image takes ~5-10 s on a CPU
node after the model loads (~2-5 s warm; longer on first call due
to torch JIT). For batch geolocalization across many images,
submit a CPU job array on `general-cpu`:

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-cpu \
       --cpus-per-task=4 \
       --mem=8G \
       --time=02:00:00 \
       --array=0-99 \
       --wrap='srun --container-image=$IMG \
         --container-mounts=/scratch2/fs1/$USER:/scratch2/fs1/$USER \
         --container-workdir=/work \
         bash -lc "export PYTHONNOUSERSITE=1; \
                   python /opt/scripts/geoclip_predict.py \
                     --image /scratch2/fs1/$USER/photos/${SLURM_ARRAY_TASK_ID}.jpg \
                     --out /scratch2/fs1/$USER/results/${SLURM_ARRAY_TASK_ID}.csv"'
```

## Caveats

- **100K-point gallery is a coarse spatial discretisation.** The
  model can only return locations from this fixed set. For finer
  resolution, the upstream offers `top_k_radius` mode that softmax-
  weights nearby gallery points; not exposed in v1.
- **Training corpus bias.** MP-16 is heavily biased toward populated,
  photographed regions (Europe, North America, East Asia). Photos
  from underrepresented geographies (interior Africa, Pacific
  islands, polar regions) will see degraded accuracy.
- **No SAR or multispectral input.** RGB only. For multispectral
  satellite work see the `dofa` / `remoteclip` containers.
