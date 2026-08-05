# Pilot outputs

These JSON files are raw CPU outputs from `experiments/pythia_cpu_smoke.py` using `EleutherAI/pythia-14m`.

- `pythia_14m_step0.json`: initialization checkpoint.
- `pythia_14m_intermediate.json`: `step1000` and `step10000`.
- `pythia_14m_step143000.json`: final checkpoint.
- `pythia_14m_pilot.json`: combined four-checkpoint run used in the Markdown summary.

The pilot uses one model seed, three hand-written prompt factorials, and eight random Fisher-orthonormal planes per context. It validates execution and numerical identities only. It is not a population estimate or evidence for a semantic/training conclusion.

The model weights and Hugging Face cache are intentionally not committed.
