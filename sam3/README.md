# sam3

SAM 3 — promptable concept segmentation. Meta (2025). SAM License.

## What changed from SAM 2

SAM 2 segments **what you point at**: a click, a box, a mask prompt, one object
at a time. SAM 3 segments **what you name**: give it `"tree"` and it finds and
masks every tree in the image or video, with no exemplar and no training.

For "every crown", "every crater", "every fracture", that is the difference
between writing a prompt and running a labelling campaign.

`sam2` stays in the catalog and is not deprecated. Point-prompt segmentation is
still right when you know where the object is and want one mask, and existing
work references the published image.

## Weights are not baked — you supply them

`facebook/sam3` is **manually gated**: Meta grants access per account. The SAM
License does permit redistribution provided the Agreement travels with the
materials, so baking the weights would be licence-compliant. This image does not
bake them anyway, because publishing them on a public registry hands them to
anyone who pulls, bypassing the gate Meta deliberately put in front of them.
Compliant is not the same as appropriate.

So: accept the gate at <https://huggingface.co/facebook/sam3> under your own
account, then supply your token at run time.

```bash
docker run --rm --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_HOME=/cache \
  -v "$PWD/hf-cache":/cache -v "$PWD":/work \
  ghcr.io/bradleylab/sam3:v1 python your_script.py
```

### Compute2 — stage first, then run offline

Compute nodes have no outbound network, so fetch the weights on a login node
into a Storage3 cache, then run against it offline.

```bash
HF_TOKEN="$HF_TOKEN" \
HF_HOME=/storage3/fs1/alexander.s.bradley/Active/hf-cache \
  python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam3')"
```

```bash
srun -A compute2-alexander.s.bradley -p general-gpu \
     --gpus=1 --cpus-per-task=8 --mem=48G --time=00:30:00 \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+sam3+v1.sqsh \
     --container-mounts=/storage3/fs1/alexander.s.bradley/Active/hf-cache:/cache \
     bash -lc 'export PYTHONNOUSERSITE=1 HF_HOME=/cache HF_HUB_OFFLINE=1; python your_script.py'
```

## Licence obligations that bind you, not just us

The Agreement is at `/opt/licenses/LICENSE.sam3.md` inside the image. Two terms
matter in practice:

1. **Publications must acknowledge SAM Materials.** A condition of use, not a
   courtesy.
2. **Trade controls.** The licence forbids use for military or warfare purposes,
   nuclear applications, espionage, and weapons development, and requires
   compliance with ITAR and sanctions law.

## Verification

The build smoke test runs with **no weights, no network and no token**: it
asserts this transformers build actually registers the `sam3_video`
architecture and resolves `Sam3VideoModel`, which is what breaks on a
transformers bump. Weight loading cannot be proven at build time by design — it
is proven in `SMOKE.md` on Compute2 instead.

**Not yet executed on Compute2.**
