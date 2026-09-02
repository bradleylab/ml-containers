# ADAF model weights — licence notice

The eight model files baked into this image at `/opt/adaf-weights` are **not**
covered by the Apache-2.0 licence that covers the ADAF source code. They are
licensed **CC-BY-SA-4.0**, and this notice ships inside the image because
CC-BY-SA requires the licence notice to travel with the work.

**Work:** ADAF machine learning model weights, version 1.0.0
**Source:** https://doi.org/10.5281/zenodo.15848663
**Licence:** Creative Commons Attribution-ShareAlike 4.0 International
(CC-BY-SA-4.0) — https://creativecommons.org/licenses/by-sa/4.0/

**Attribution.** ADAF was developed by Nejc Čož, Žiga Kokalj, Anthony Corns,
Susan Curran, Dragi Kocev, and Ana Kostovska, through collaboration between ZRC
SAZU, Bias Variance Labs, and The Discovery Programme. Funded by Transport
Infrastructure Ireland Open Research Call 2021, the Slovenian Research and
Innovation Agency (P2-0406), and European Research Council project STONE
(GAP-101089123).

**Cite:** Čož, N., Corns, A., Curran, S., Kocev, D., & Kokalj, Ž. (2026).
*Journal of Archaeological Science: Reports* 71:105733.
https://doi.org/10.1016/j.jasrep.2026.105733

## What share-alike means for work done with this image

- **Running inference and publishing the detections** — unencumbered. Results
  are not a derivative of the weights.
- **Fine-tuning these weights** — the resulting weights are an adapted work. You
  may use them freely in-house, but **distributing** them obliges you to
  distribute under CC-BY-SA-4.0.
- **Needing permissively-licensed weights for a downstream** — do not start from
  these. Train from ImageNet initialisation inside the same Apache-2.0 ADAF and
  AiTLAS code. That forfeits the archaeological pretraining and buys licence
  freedom; it is a real trade, not a formality.
- **Redistributing the weights unmodified** (which this image does) — permitted,
  with attribution and this notice, both of which are satisfied here.
