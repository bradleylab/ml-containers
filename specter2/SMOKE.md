# specter2 — smoke test

## 1. Build (GitHub Actions)

`specter2-fetch-weights` downloads the base model and the four adapters at
their pinned revisions and stops the build if any file's SHA-256 differs from
`scripts/specter2_weights.py`. The build then ends with `specter2-selftest`,
run on the CPU with the network disabled. It embeds the model card's two
example papers ("BERT" and "Attention is all you need") with each adapter, and
the query "language representation model" with `adhoc_query`, and checks:

- every embedding is float32, 768 wide and finite;
- the paper ids come back in input order;
- each adapter's vectors differ from the `proximity` adapter's, so the adapter
  was applied.

It also prints the cosine similarity of the query to each paper. Run on an
Apple-silicon Mac with the recipe's pinned versions (torch 2.6.0, adapters
1.2.0, transformers 4.51.3) before the image was built, it printed:

| Paper | Cosine to the query |
|---|---|
| BERT | 0.8322 |
| Attention is all you need | 0.8248 |

## 2. Compute2, H100

Import, then run the self-test on a GPU node:

```bash
srun -A compute2-alexander.s.bradley \
  -p general-gpu \
  --gpus=1 \
  --container-image=ghcr.io#bradleylab/specter2:v1 \
  bash -c 'python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))" && specter2-selftest'
```

The self-test forces the CPU, so the first command is the GPU check. Check
that PyTorch sees the H100 and that the self-test prints the same cosines as
the build.
