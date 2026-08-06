# Pilot outputs

These JSON files are raw CPU outputs from `experiments/pythia_cpu_smoke.py` using `EleutherAI/pythia-14m`.

- `pythia_14m_step0.json`: initialization checkpoint.
- `pythia_14m_intermediate.json`: `step1000` and `step10000`.
- `pythia_14m_step143000.json`: final checkpoint.
- `pythia_14m_pilot.json`: combined four-checkpoint run used in the Markdown summary.

The pilot uses one model seed, three hand-written prompt factorials, and eight Fisher-Haar planes per context. It predates the implemented block-spectrum-matched control. Eight controls give a minimum plus-one one-sided tail rank of \(1/9\); that rank is not automatically a calibrated p-value for a hand-selected semantic plane. The semantic-minus-Haar contrast is also confounded by Fisher-spectrum alignment. The pilot validates execution and numerical identities only; it is not a population estimate or evidence for a semantic or training conclusion.

The model weights and Hugging Face cache are intentionally not committed.
