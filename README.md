# Predictive Fisher Geometry

This project studies a precise version of the question “are language-model representations really linear?”

The current answer is:

> At a linear softmax head, hidden addition is exactly affine for the flat exponential connection. It is mathematically coherent, but it does not preserve the Fisher metric. Fisher--Levi-Civita transport preserves predictive lengths and angles, but can be curved, path dependent, and only locally feasible.

The research task is to determine which tradeoff actual semantic composition follows.

## Theory paper

The arXiv-style manuscript is in [`paper/main.tex`](paper/main.tex), with a verified PDF at [`output/pdf/predictive_geometric_agreement.pdf`](output/pdf/predictive_geometric_agreement.pdf).

Its central result is a quantitative stability ladder from predictive-map agreement to Fisher metric, Amari alpha-connection, parallel-transport, and curvature agreement. The revised paper adds a square-root/Hellinger theorem whose Levi--Civita constants have no explicit vocabulary-size or minimum-token-probability dependence, and score-moment bounds for affine heads. It also proves that cross-entropy convergence alone is insufficient and states exactly when conditional variance is informative: only for a genuinely noninjective predictive map. A declared coarsening is handled by replacing the map, effect, and image geometry with their coarsened counterparts, not by conditioning the original tangent field on coarse labels.

A key audit result is that “final-layer flatness” depends on the metric. Pulling back
Euclidean distance between centered logits gives a constant flat metric, while pulling
back local KL/cross-entropy gives the probability-dependent Fisher metric studied here.

## Start here

The manuscript and experiment protocol are authoritative. The older proof and proposal files are retained as development history and can contain superseded conjectures or terminology.

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md): plain-language explanation of what is proved, what the pilot observed, and what remains open.
- [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md): current CPU protocol, spectrum-matched controls, numerical gates, finite-edge safeguards, falsification criteria, and the execution-only Pythia-14M pilot.
- [RED_TEAM_RESOLUTION.md](RED_TEAM_RESOLUTION.md): disposition of every substantive theoretical and empirical audit finding.
- [TRAINING_REGIMES.md](TRAINING_REGIMES.md): proposed matched training objectives; these remain future experiments rather than supported conclusions.
- [PREDICTIVE_CONNECTION_DISTILLATION.md](PREDICTIVE_CONNECTION_DISTILLATION.md): unified teacher--student protocol combining the KL--Sobolev Levi--Civita guarantee with direct \((G,L,C)\) packet distillation, derivative controls, CPU curriculum, and falsification criteria.
- [BROAD_HYPOTHESIS_PROGRAM.md](BROAD_HYPOTHESIS_PROGRAM.md), [MSC_NOVELTY_PROPOSAL.md](MSC_NOVELTY_PROPOSAL.md), [PROOFS_AND_NOVELTY.md](PROOFS_AND_NOVELTY.md), and [RESEARCH_NOTES.md](RESEARCH_NOTES.md): historical development notes, superseded wherever they conflict with the manuscript or protocol.

## Implemented code

- `src/predictive_geometry/simplex.py`: exact mixture, exponential, and Fisher--LC operations on the open categorical simplex.
- `src/predictive_geometry/softmax.py`: exact decoder Fisher metric, cubic tensor contractions/operators, curvature commutators, and sectional curvature from vocabulary cumulants.
- `src/predictive_geometry/stability.py`: square-root cubic diagnostics and executable finite-scale Levi--Civita stability bounds, including the metric-compatible transport refinement.
- `src/predictive_geometry/field.py`: alpha-connection transport, local transport defect, and semantic connection fitting.
- `src/predictive_geometry/benchmark.py`: held-out semantic quadrilateral benchmark with feasibility, Fisher error, KL/JS error, and transport metric distortion.
- `src/predictive_geometry/distillation.py`: shared-chart connection packets with score-moment cubics, distillation losses, acceptance gates, checksummed serialization, and the packet-to-connection and packet-to-transport bounds.
- `experiments/synthetic_validation.py`: Fisher-sphere and holonomy checks.
- `experiments/synthetic_connection_recovery.py`: recovers synthetic mixture-, exponential-, and Fisher-generated compositions.
- `experiments/pythia_cpu_smoke.py`: real Pythia checkpoint inference and curvature on CPU.
- `experiments/synthetic_semantic_alpha.py`: recovers known synthetic semantic connections with NumPy only.

## Verification status

- The complete automated CPU suite passes.
- Saturated three-category curvature recovers \(K=1/4\), including a near-boundary stress test.
- Curvature is invariant under invertible hidden-coordinate changes.
- The curvature-operator commutator and hidden-translation metric-derivative identities are checked numerically.
- Synthetic connection identities are recovered exactly.
- The distillation packet core recovers exact affine-head and Bernoulli closed forms, numerically verifies the packet-to-connection and packet-to-transport bounds, and demonstrates the KL-only oscillatory escape.
- A real Pythia-14M pilot ran at four checkpoints with no GPU.
- Small raw pilot JSON outputs are included under `results/`; downloaded model weights are excluded.

The pilot is an implementation result, not evidence for a scientific conclusion: it uses one seed, three prompt families, and eight Fisher-Haar controls. Those controls are not spectrum matched and cannot resolve a one-sided Monte Carlo p-value below \(1/9\).

## Run tests

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The model-free geometry requires only NumPy. The Pythia experiment additionally requires CPU-only PyTorch, Transformers, and Safetensors; see `requirements-model.txt`.

## Current MSc research target

The broad “probabilistic embeddings instead of vectors” thesis is already occupied by information geometry, manifold embeddings, and recent work on softmax geometry and representation holonomy.

The mathematical target is a vocabulary-independent stability theory:

> Levi--Civita transport is controlled by square-root/Hellinger regularity, while a general nonzero Amari alpha-connection additionally requires a stable raised third score moment. This distinction is sharp: paired Bernoulli maps can converge in square-root C² while every fixed nonzero-alpha connection defect diverges. A tokenwise probability floor is sufficient but not necessary; affine softmax heads provide floor-free second-, third-, and fourth-score-moment formulas.

The paired empirical question is which of exponential, Levi--Civita, mixture, or an intermediate alpha-connection predicts held-out semantic tangent fields and behavioral transfer. Ordinary vector reuse is the exponential arm, not a non-geometric baseline. A non-exponential winner must survive spectrum-matched controls, length-to-zero or integrated transport, multiple seeds, and behavioral evaluation.

The cross-model extension asks whether a large teacher can transfer this local
predictive structure to a smaller student through compact \((G,L,C)\) packets on a
shared intervention chart. The packet reconstructs the full Amari connection
family without aligning hidden dimensions. Output-only, metric-only,
centered-logit Jacobian, square-root Jacobian, Levi--Civita, full-cubic, and
connection-scrambled controls are compared at matched student NLL and compute.
The separate KL-plus-\(H^s\) route tests the manuscript's vocabulary-independent
LC transport guarantee while treating lower-order roughness only as a
heuristic. No distillation result has yet been run.
