# specter2

Embeddings of scientific papers from their title and abstract (Singh,
D'Arcy, Cohan, Downey and Feldman, 2023). SPECTER2 is a BERT encoder
initialized from SciBERT and trained on over 6 million citation triplets: a
query paper, a paper it cites, and one it does not. Small adapter modules on
top of it were then trained for four task formats on the SciRepEval tasks.
Papers that cite each other land close together, which makes the embeddings
useful for finding related papers, ranking a corpus against a query, and as
features for classifiers.

| Adapter | Model card's use |
|---|---|
| `proximity` | Papers as queries and candidates: nearest-neighbor search, link prediction. The card's default for tasks not listed below |
| `adhoc_query` | Short free-text search queries; embed the candidate papers with `proximity` |
| `classification` | Paper features for linear classifiers |
| `regression` | Paper features for linear regressors |

The model card reports these scores on SciRepEval (averages over its tasks)
and on MDCR, a citation-recommendation benchmark:

| Model | SciRepEval in-train | SciRepEval out-of-train | SciRepEval average | MDCR MAP | MDCR Recall@5 |
|---|---|---|---|---|---|
| SPECTER2 base, no adapter | 56.3 | 73.6 | 69.1 | 38.0 | 32.4 |
| SPECTER2 with adapters | 62.3 | 74.1 | 71.1 | 38.4 | 33.0 |
| SPECTER, the predecessor | 54.7 | 72.0 | 67.5 | 30.6 | 25.5 |

Paper: [SciRepEval, EMNLP 2023](https://doi.org/10.18653/v1/2023.emnlp-main.338) ·
Code: [allenai/SPECTER2](https://github.com/allenai/SPECTER2) ·
Weights: [allenai/specter2_base](https://huggingface.co/allenai/specter2_base) and its four adapters ·
License: Apache-2.0 (code and weights)

Pull: `ghcr.io/bradleylab/specter2:v1`

## Run it

Papers are JSON Lines with an `id`, a `title` and an optional `abstract`:

```json
{"id": "paper-001", "title": "…", "abstract": "…"}
```

```bash
specter2-embed --papers papers.jsonl --out papers.npz
specter2-embed --queries queries.txt --out queries.npz
```

The output `.npz` holds `ids` (the input ids, or the line order for queries)
and `embeddings`, float32, one 768-wide row per input. For search, compare the
query rows with the paper rows by cosine similarity.

On Compute2, from a folder under Storage3 that holds the input:

```bash
srun -A compute2-alexander.s.bradley \
  -p general-gpu \
  --gpus=1 \
  --container-image=ghcr.io#bradleylab/specter2:v1 \
  --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
  --container-workdir="$PWD" \
  specter2-embed --papers papers.jsonl --out papers.npz
```

The encoder is BERT-base sized, so a CPU partition also works for modest
corpora: add `--export=ALL,NVIDIA_VISIBLE_DEVICES=void` and drop `--gpus`.

| Option | Default | Meaning |
|---|---|---|
| `--papers` / `--queries` | | The input, one of the two |
| `--adapter` | `proximity` for papers, `adhoc_query` for queries | `proximity`, `adhoc_query`, `classification` or `regression` |
| `--device` | `cuda` when a GPU is visible, else `cpu` | |

## What the image does to a paper

As in the model card's example: the title and abstract are joined with the
tokenizer's `[SEP]` token, truncated to 512 tokens, run through the encoder
with the chosen adapter active, and the last layer's `[CLS]` vector is the
embedding. A paper with no abstract is embedded from its title alone.

## Choices made in this recipe

- **Weights baked.** The base model (439,740,465 bytes) and the four adapters
  (3,593,365 bytes each) are downloaded at build from pinned Hugging Face
  revisions and checked against their SHA-256 values
  (`scripts/specter2_weights.py`). The image then runs with
  `HF_HUB_OFFLINE=1`.
- **The 2023 release, not the refresh.** AllenAI also publishes
  `specter2_aug2023refresh`, pretrained on papers published after 2018, and
  the model card asks that the original be used for benchmarking. This image
  carries the original.
- **PyTorch 2.6.** transformers 4.50 and later refuse to load `.bin`
  checkpoints under older PyTorch, and SPECTER2 ships no other format. The
  adapters library 1.2.0 requires transformers 4.51.
- **An adapter is required to be active.** The adapters library prints "none
  are activated" while it loads the base model. `specter2-embed` checks that
  the requested adapter is active before embedding, because without it the
  output would silently be the base model's.

## Build-time check

With the network disabled, the build embeds the model card's two example
papers with every adapter and one query with `adhoc_query`, checks that the
vectors are finite and 768 wide, and that different adapters give different
vectors. See `SMOKE.md`.
