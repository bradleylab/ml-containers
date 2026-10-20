# geobert

Named-entity recognition for geoscience text. GeoBERT is a BERT token
classifier that its model card describes as fine-tuned from SciBERT on a
"Labeled Geoscientific Corpus" of about one million sentences. It tags four
entity types:

| Label | Model card's definition |
|---|---|
| GeoPetro | any term in the geosciences |
| GeoMeth | tools or methods associated with the geosciences |
| GeoLoc | geological locations |
| GeoTime | geological time-scale entities |

The model card reports these seqeval scores, without saying what data they
were measured on:

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| GeoLoc | 0.9727 | 0.9591 | 0.9658 |
| GeoMeth | 0.9433 | 0.9447 | 0.9445 |
| GeoPetro | 0.9767 | 0.9745 | 0.9756 |
| GeoTime | 0.9695 | 0.9666 | 0.9680 |

Weights: [botryan96/GeoBERT](https://huggingface.co/botryan96/GeoBERT) @
`308126d` (TensorFlow `tf_model.h5`, 437,612,588 bytes) · No license · No paper
or code repository

Pull: `ghcr.io/bradleylab/geobert:v1`

This is not the GeoBERT of Liu et al. (2022, *Earth Science Informatics*,
[doi:10.1007/s12145-022-00775-x](https://doi.org/10.1007/s12145-022-00775-x))
or of Denli et al. (2021, ADIPEC,
[doi:10.2118/207766-ms](https://doi.org/10.2118/207766-ms)). Those are
separate models that share the name.

## WashU-internal: no license

The model repository states no license, and neither do any of its author's
other Hugging Face repositories. Absent a license, no permission to
redistribute has been granted, so:

- **The weights are not in the image.** `geobert-ner` downloads them into a
  cache you name, from the pinned revision, and refuses to load
  `tf_model.h5` unless its SHA-256 matches the value published for that
  revision (`67c1b568…cf227ce`).
- The image holds only the lab's wrapper and public libraries.
- Use it for WashU research only, and do not republish the weights or present
  the model as the lab's.

There is no description of the training corpus, its sources or its labeling,
and no held-out test set is named, so the scores in the table cannot be
checked. Treat them as the uploader's claim.

## Run it

Input is one or more UTF-8 text files with one passage per line. Output is
JSON Lines, one object per non-empty input line. This record, cut to one of
its eight entities, is the real output for the example sentence in the model
card:

```json
{"file": "card.txt", "line": 1, "entities": [{"label": "GeoTime", "text": "Paleozoic", "start": 445, "end": 454, "score": 0.9999}]}
```

`start` and `end` are character offsets into the line, and `text` is that
slice of the original line. Tokens are grouped into spans with the
transformers `aggregation_strategy="simple"`, as in the model card's example.

On Compute2, from a folder under Storage3 that holds the input, with the
weights cached in the lab's Hugging Face cache:

```bash
srun -A compute2-alexander.s.bradley \
  -p general-gpu \
  --gpus=1 \
  --container-image=ghcr.io#bradleylab/geobert:v1 \
  --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
  --container-workdir="$PWD" \
  geobert-ner abstracts.txt \
    --out entities.jsonl \
    --cache-dir /storage3/fs1/alexander.s.bradley/Active/hf-cache
```

The model is BERT-base sized (12 layers, 768 wide, per its `config.json`) and
runs on the CPU too; on a CPU partition add
`--export=ALL,NVIDIA_VISIBLE_DEVICES=void` and drop `--gpus`.

| Option | Meaning |
|---|---|
| `--out` | Output JSON Lines file |
| `--cache-dir` | Hugging Face cache to download into, or reuse; the first run fetches about 438 MB |
| `--model-dir` | A local model folder to use instead, with no download and no hash check |

## Limits

- **512 tokens per passage.** The model's position embeddings stop at 512
  tokens, and its tokenizer files do not set a limit. A longer line is not
  truncated: it is written to the output with a `"skipped"` field, and
  counted on stderr. Split long passages, for example into sentences.
- **Uncased.** The tokenizer lowercases its input, so "Sandstone" and
  "sandstone" are the same to the model.

## Choices made in this recipe

- **TensorFlow 2.15 and transformers 4.44.** The checkpoint exists only as
  TensorFlow weights, transformers 5 removed TensorFlow support, and 4.44
  accepts TensorFlow 2.15, the last release on Keras 2. The model card's
  framework versions are transformers 4.22.1 and TensorFlow 2.10.0.
- **The revision is pinned.** A later push to the model repository is not
  used until the pin and the hash in `scripts/geobert-ner` are changed.

## Build-time check

With the network disabled, the build saves a tiny randomly initialized BERT
token classifier with GeoBERT's nine labels and runs `geobert-ner` on it end to
end: one passage tagged, one over-length passage skipped. This proves the
TensorFlow stack, the pipeline and the output format, not accuracy. See
`SMOKE.md` for the run with the real weights.
