# forpt — smoke test

## 1. Build (GitHub Actions)

`forpt-fetch-weights` downloads the four checkpoints from `prs-eth/ForPT` at
revision `6b9047e` and stops the build if any file's SHA-256 differs from
`scripts/forpt_lib.py`. The build then ends with `forpt-selftest`, run on the
CPU with the network disabled. It:

- builds the pretrained encoder and the semseg, species and age models from
  upstream's configs and loads the baked weights strictly;
- checks there are 3 semseg classes and 33 species;
- runs each task's test transforms on a synthetic tree (a ground square, a
  trunk and a spherical crown), with the semseg and species copies shifted to
  UTM-sized coordinates, and checks that the semseg voxel map covers every
  input point and that the coordinates come out centered.

A forward pass needs FlashAttention and so a GPU; on a GPU node the same command
also runs every model on the synthetic tree and checks the outputs are finite.

## 2. Compute2, H100

Import the image, then run three checks in one GPU job: the self-test with its
forward passes, upstream's own tester on the FOR-instanceV2 test split, and the
image's `forpt_lib` path on the same split. The test split is 1.7 GB, so it is
downloaded inside the job, not on the login node.

```bash
JOB=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/forpt-h100
srun -A compute2-alexander.s.bradley \
  -p general-gpu \
  --gpus=1 \
  --container-image=$JOB/bradleylab+forpt+v1.sqsh \
  --container-mounts=$JOB:$JOB \
  bash -c "
    forpt-selftest &&
    HF_HUB_OFFLINE=0 HF_HOME=$JOB/hf-home \
      hf download prs-eth/forpt-forinstancev2-sem test.tar --repo-type dataset --local-dir $JOB/data &&
    tar -xf $JOB/data/test.tar -C $JOB/data &&
    cd /opt/forpt/src &&
    python tools/test.py \
      --config-file configs/01_forinstancev2_sem/semseg-forpt-ft-p100.py \
      --num-gpus 1 \
      --options weight=/opt/forpt/weights/01_forinstancev2_sem__semseg-forpt-ft-p100.pth \
        data.test.data_root=$JOB/data save_path=$JOB/exp enable_wandb=False &&
    python $JOB/forpt_lib_miou.py $JOB/data/test"
```

`forpt_lib_miou.py` is `smoke/forpt_lib_miou.py` from this folder, copied into
the job folder. The image sets `HF_HUB_OFFLINE=1`, so the download line turns
it off for that one command.

What to check:

- `forpt-selftest` prints the H100 as its device, a forward pass for each model
  and `SELFTEST OK`.
- Upstream's tester ends with `Val result: mIoU/mAcc/allAcc`. Table 1 of the
  paper gives 88.3 / 92.4 / 95.6 for this checkpoint, with IoU of 99.1 for
  ground, 94.9 for leaf and 71.0 for wood. The tester draws its own random seed,
  so its numbers can differ from the paper's in the last digit.
- `forpt_lib_miou.py` prints mIoU, overall accuracy and per-class IoU through
  the code path `forpt-semseg` uses. They should match the tester's to within
  the same run-to-run variation; a larger gap means the image's preprocessing
  has drifted from upstream's.
