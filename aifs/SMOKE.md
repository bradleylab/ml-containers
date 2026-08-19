# aifs — Compute2 smoke test

Smallest run that proves the image works end to end on an H100:

- One single-GPU job (`--gpus=1 --mem=64G`, cache mounted at `/root/.cache`) running the checkpoint repo's shipped config at the shortest possible lead time — `anemoi-inference run inference.yaml lead_time=6`, i.e. one 6-hour model step from the latest ECMWF Open Data date — which passes if `output.grib` is written and its `2t` field has a global mean in the 250–300 K range (a wrong-checkpoint or wrong-input failure lands far outside that); budget `--time=01:00:00` and replace this with the measured wall time after the first successful run, since nothing here has been timed on C2 yet.
