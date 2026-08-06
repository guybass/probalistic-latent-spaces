# Predictive Fisher Geometry

This project studies a precise version of the question “are language-model representations really linear?”

The current answer is:

> At a linear softmax head, hidden addition is exactly affine for the flat exponential connection. It is mathematically coherent, but it does not preserve the Fisher metric. Fisher--Levi-Civita transport preserves predictive lengths and angles, but can be curved, path dependent, and only locally feasible.

The research task is to determine which tradeoff actual semantic composition follows.

## Theory paper

The arXiv-style manuscript is in [`paper/main.tex`](paper/main.tex), with a verified PDF at [`output/pdf/predictive_geometric_agreement.pdf`](output/pdf/predictive_geometric_agreement.pdf).

Its central result is a quantitative stability ladder from predictive-map agreement to Fisher metric, Amari alpha-connection, parallel-transport, and curvature agreement. It also proves that cross-entropy convergence alone is insufficient, gives a loss-to-transport theorem under Sobolev regularity, and separates predictive semantic insufficiency from connection mismatch through a conditional-variance decomposition.

A key audit result is that “final-layer flatness” depends on the metric. Pulling back
Euclidean distance between centered logits gives a constant flat metric, while pulling
back local KL/cross-entropy gives the probability-dependent Fisher metric studied here.

## Start here

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md): plain-language explanation of what is proved, what the pilot observed, and what remains open.
- [BROAD_HYPOTHESIS_PROGRAM.md](BROAD_HYPOTHESIS_PROGRAM.md): corrected broad hypothesis, connection-relative linearity, proof/disproof table, and expanded evidence program.
- [MSC_NOVELTY_PROPOSAL.md](MSC_NOVELTY_PROPOSAL.md): selected thesis-sized novelty, exact local results, literature boundary, and CPU-only experiment.
- [TRAINING_REGIMES.md](TRAINING_REGIMES.md): matched CPU training objectives for testing whether a chosen connection causally improves semantic transfer.
- [PROOFS_AND_NOVELTY.md](PROOFS_AND_NOVELTY.md): audited theorems, counterexamples, closest prior art, and the defensible novelty boundary.
- [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md): pre-registered CPU program, controls, falsification criteria, and the Pythia-14M pilot.
- [RESEARCH_NOTES.md](RESEARCH_NOTES.md): the original developing notebook and book-flatness explanation.

## Implemented code

- `src/predictive_geometry/simplex.py`: exact mixture, exponential, and Fisher--LC operations on the open categorical simplex.
- `src/predictive_geometry/softmax.py`: exact decoder Fisher metric, cubic tensor contractions/operators, curvature commutators, and sectional curvature from vocabulary cumulants.
- `src/predictive_geometry/field.py`: alpha-connection transport, local transport defect, and semantic connection fitting.
- `src/predictive_geometry/benchmark.py`: held-out semantic quadrilateral benchmark with feasibility, Fisher error, KL/JS error, and transport metric distortion.
- `experiments/synthetic_validation.py`: Fisher-sphere and holonomy checks.
- `experiments/synthetic_connection_recovery.py`: recovers synthetic mixture-, exponential-, and Fisher-generated compositions.
- `experiments/pythia_cpu_smoke.py`: real Pythia checkpoint inference and curvature on CPU.
- `experiments/synthetic_semantic_alpha.py`: recovers known synthetic semantic connections with NumPy only.

## Verification status

- 28 automated CPU tests pass.
- Saturated three-category curvature recovers \(K=1/4\), including a near-boundary stress test.
- Curvature is invariant under invertible hidden-coordinate changes.
- The curvature-operator commutator and hidden-translation metric-derivative identities are checked numerically.
- Synthetic connection identities are recovered exactly.
- A real Pythia-14M pilot ran at four checkpoints with no GPU.
- Small raw pilot JSON outputs are included under `results/`; downloaded model weights are excluded.

The pilot is an implementation result, not evidence for a scientific conclusion: it uses one seed and three prompt families.

## Run tests

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The model-free geometry requires only NumPy. The Pythia experiment additionally requires CPU-only PyTorch, Transformers, and Safetensors; see `requirements-model.txt`.

## Selected MSc contribution

The broad “probabilistic embeddings instead of vectors” thesis is already occupied by information geometry, manifold embeddings, and recent 2026 work on softmax geometry and representation holonomy.

The selected question is:

> Which information-geometric connection makes an observed context-dependent semantic vector field closest to parallel?

The proposed semantic alpha-index selects among exponential, Levi--Civita, mixture, or an intermediate alpha-connection using local held-out transport residuals. The previous curvature and three-connection work remains the mathematical foundation and a secondary analysis.
