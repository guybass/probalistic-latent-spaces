# Project overview

## The 30-second version

Language-model hidden states are ordinary vectors, but those vectors also parameterize probability distributions over the next token. These two facts give the same representation two geometries:

1. a flat affine geometry, where reusing the same vector is natural;
2. a predictive Fisher geometry, where lengths measure changes in model behavior and Levi--Civita transport can be curved.

The research question is therefore not whether vectors or manifolds “exist.” Both do. The question is:

> **Which connection makes a semantic transformation remain the same when the context changes?**

Ordinary vector addition chooses the flat exponential connection. Fisher--Levi-Civita transport chooses metric compatibility. Mixture transport keeps dual/expectation-coordinate information constant. The project measures which choice predicts real semantic transformations and interventions.

```mermaid
flowchart LR
    H["Hidden state h in R^d"] --> P["Next-token distribution p(.|h)"]
    P --> G["Predictive Fisher metric G(h)"]
    G --> C["Choose transport: e, LC, m, or alpha"]
    C --> T["Transport a semantic effect across contexts"]
    T --> E["Held-out prediction and causal intervention"]
```

## What we originally believed

The initial hypothesis was that latent spaces are probabilistic manifolds rather than vector spaces, that vector sums are invalid, and that parallel transport should replace addition.

The useful core was right, but the literal statement was not:

- hidden states really are vectors;
- every finite final hidden state maps through a linear softmax head to a valid distribution;
- constant-vector reuse is already parallel transport under the exponential connection;
- a global isometric embedding can still be intrinsically curved.

The corrected hypothesis is:

> A global context-independent vector may fail to be the correct **semantic invariant**, even though vector addition is mathematically valid. Semantic transformations should be modeled as local vector fields and tested for covariant constancy under competing connections.

For a semantic operation $T$, write

\[
X_T(h)=h(Tc)-h(c).
\]

The operation is connection-linear when

\[
\nabla X_T\approx0.
\]

This is called **connection-relative linearity**.

## What is proved, observed, and still open

### Proved mathematically

- A linear softmax decoder induces the exact pullback Fisher metric

  \[
  G(h)=W^\top\bigl(\operatorname{diag}p_h-p_hp_h^\top\bigr)W.
  \]

- Hidden coordinates are flat for the exponential connection.
- Mixture coordinates are flat for the dual mixture connection.
- Levi--Civita transport preserves Fisher lengths and angles but may be curved.
- In a dually flat family,

  \[
  R^{(\alpha)}=(1-\alpha^2)R^{LC}.
  \]

- Nonzero LC curvature obstructs a global vector-addition law whose translations also preserve the Fisher metric.
- A local held-out transport residual estimates $\nabla^{(\alpha)}X_T$, so connection-relative linearity is empirically testable.

### Verified computationally

- 26 model-free CPU tests pass.
- Exact categorical curvature recovers $1/4$.
- LC transport preserves Fisher length numerically.
- Synthetic $e$, LC, $m$, and intermediate-alpha fields are recovered by the estimator.
- The Pythia-14M decoder experiment runs on CPU with full-vocabulary geometry.

### Preliminary Pythia-14M observation

Across four checkpoints, one seed, three hand-written semantic factorials, and eight matched random planes per context:

- metric condition numbers changed from about $1.27$ at initialization to roughly $3.1\times10^3$--$4.2\times10^3$ at the final checkpoint;
- semantic sectional curvature changed from roughly $0.001$ at initialization to values between $-0.0445$ and $0.0211$ at the final checkpoint;
- LC Fisher-length distortion stayed below $1.3\times10^{-16}$, while final-checkpoint mean absolute distortion was $0.0877$ for exponential and $0.4389$ for mixture transport;
- unscaled finite mixture and ambient LC analogies left their valid domains, while exponential composition remained feasible.

These are engineering results, not semantic evidence. The sample is far too small to establish a trend, and entropy and conditioning changed simultaneously.

### Still open

- Which connection best predicts held-out semantic fields?
- Does transporting a source vector improve an actual target intervention?
- Does curvature/holonomy predict operation-order or composition failure?
- Does the preferred connection develop reproducibly during pretraining?
- Can a training objective deliberately create more transferable connection-linear representations?

## How the hypothesis can lose

If exponential transport consistently wins on held-out contexts and connection-aware corrections do not improve interventions, the strong hypothesis is false for the tested operations. The Fisher manifold still exists, but it is not operationally necessary.

If LC wins, the original intuition receives its strongest support: predictive metric compatibility is more useful than constant hidden coordinates.

If mixture or an intermediate alpha wins, probabilistic connection-aware semantics is supported, but LC is not privileged.

If no connection generalizes, predictive Fisher geometry is insufficient and context dependence must be modeled by a richer field, support geometry, or learned connection.

## Why training regimes matter

Checkpoint observation establishes correlation at best. Training with a controlled geometric objective gives a causal experiment:

> If two models have matched language-model loss but one is trained to make semantic fields parallel under a declared connection, does it obtain better held-out semantic transfer and composition?

The first CPU-feasible training study compares:

1. cross-entropy only;
2. Euclidean/exponential vector-consistency regularization;
3. Fisher-weighted exponential consistency;
4. first-order LC parallel-consistency regularization;
5. mixture consistency;
6. a fitted-alpha diagnostic evaluated after training.

A meaningful positive result must improve behavior—not merely make the regularizer smaller or curvature flatter. The primary outcomes are held-out compositional accuracy, intervention target log-odds, off-target KL, and transfer error at matched language-model loss.

See [TRAINING_REGIMES.md](TRAINING_REGIMES.md) for the complete design.

## Recommended reading order

1. This overview.
2. [BROAD_HYPOTHESIS_PROGRAM.md](BROAD_HYPOTHESIS_PROGRAM.md) for the proof/disproof logic.
3. [MSC_NOVELTY_PROPOSAL.md](MSC_NOVELTY_PROPOSAL.md) for the first thesis-sized experiment.
4. [TRAINING_REGIMES.md](TRAINING_REGIMES.md) for causal training studies.
5. [PROOFS_AND_NOVELTY.md](PROOFS_AND_NOVELTY.md) for the full mathematics and literature boundary.
6. [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for the original checkpoint/composition protocol.
