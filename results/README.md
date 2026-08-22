# Pilot outputs

These JSON files are raw CPU outputs from `experiments/pythia_cpu_smoke.py` using `EleutherAI/pythia-14m`.

- `pythia_14m_step0.json`: initialization checkpoint.
- `pythia_14m_intermediate.json`: `step1000` and `step10000`.
- `pythia_14m_step143000.json`: final checkpoint.
- `pythia_14m_pilot.json`: combined four-checkpoint run used in the Markdown summary.

The pilot uses four repeated-measure checkpoints from one pretrained model run,
three hand-written prompt factorials, and eight Fisher-Haar planes per context.
The files predate result-schema versioning and use ambiguous `random_seed`
fields for control draws; those values are not independent model-training
seeds. They also predate the implemented block-spectrum-matched control. Eight
controls give a minimum plus-one one-sided tail rank of \(1/9\); that rank is
not automatically a calibrated p-value for a hand-selected semantic plane. The
semantic-minus-Haar contrast is also confounded by Fisher-spectrum alignment.
The pilot validates execution and numerical identities only; it is not a
population estimate or evidence for a semantic or training conclusion.

New outputs from the driver use schema `pythia-smoke-2`. They store immutable
model and tokenizer snapshot commits, tokenizer/outcome/design and environment
lock hashes, repository commit and dirty state, runtime versions, and distinct
model-run/base-control/per-control seed fields. The legacy files above cannot
be upgraded to full v2 provenance because the missing facts were not recorded.
The driver refuses dirty or unidentified checkouts by default; its explicit
dirty-run override is for smoke testing and is recorded as nonreproducible.

The model weights and Hugging Face cache are intentionally not committed.
