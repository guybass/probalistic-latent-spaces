# MSc thesis proposal: selecting the connection of a semantic vector field

> **Historical proposal (superseded where inconsistent).** See [paper/main.tex](paper/main.tex), [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), and [RED_TEAM_RESOLUTION.md](RED_TEAM_RESOLUTION.md). The closed-form alpha estimator below is first order and has generic \(O(\ell)\) finite-edge parameter bias; current use requires length-to-zero extrapolation or integrated transport and a connection-neutral tangent-field construction.

**Literature audit date:** 4 August 2026
**Status:** mathematically specified; synthetic implementation verified; language-model experiment not yet run
**Compute assumption:** CPU only

This document is the minimal decisive study. The larger hypothesis and the exact conditions under which this study supports or falsifies it are developed in [BROAD_HYPOTHESIS_PROGRAM.md](BROAD_HYPOTHESIS_PROGRAM.md).

## The selected idea

> **Which information-geometric connection makes a context-dependent semantic transformation field closest to parallel?**

For a controlled semantic operation (T), observe its effect at many contexts,

\[
V_T(h_i)=h(Tc_i)-h(c_i),
\qquad h_i=h(c_i),
\]

at the representation immediately entering a trained linear softmax head. Compare the same local field using the exponential (`e`), Fisher--Levi-Civita (LC), mixture (`m`), and fitted Amari alpha-connections.

The output of the thesis is a **semantic connection profile**:

\[
\widehat\alpha_T
\quad\text{and}\quad
\bigl(E_e(T),E_{LC}(T),E_m(T)\bigr),
\]

estimated on held-out local context edges. Here

- (α=1) is the flat (e)-connection: ordinary hidden-vector components stay constant;
- (α=0) is LC: Fisher lengths and angles are preserved;
- (α=-1) is the flat (m)-connection: expectation-coordinate covectors stay constant.

This is the right MSc-sized question. It has one exact model family, one local theorem, one new estimator, one controlled experiment, and useful positive and negative outcomes. It does **not** attempt to invent a new manifold for all transformer layers.

## Novelty verdict and closest work

The broad ideas around the proposal are occupied:

- Hu, Niu, and Varma formalize contextual transformations as vector fields and show that no single displacement direction dominates ([paper](https://arxiv.org/abs/2607.04525)). They compare point clouds and fields using kernels, CKA, PCA, and Grassmann geometry, not predictive Fisher transport.
- Steering Vector Fields learns a context-dependent direction (v(h)=\nabla f(h)) because a fixed vector can be locally misaligned ([paper](https://arxiv.org/abs/2602.01654)). It does not ask whether the apparent variation is explained by a connection.
- Park et al. establish the exact dually flat (e/m) geometry of softmax and use it for interpolation and dual steering ([paper](https://arxiv.org/abs/2602.15293)). They explicitly focus on the dual connections rather than LC transport.
- FishBack derives pullback Fisher metrics and locally optimal Fisher steering directions ([paper](https://arxiv.org/abs/2605.17231)). It does not study LC or (α)-parallel transport of an observed semantic field.
- Natural Alpha Embeddings varies (α)-geometry for static conditional item/word embeddings ([paper](https://arxiv.org/abs/1912.02280)). It does not estimate an (α)-connection from contextual transformation fields.
- A 2018 workshop abstract by Volpi and Malagò compares static word relations after information-geometric parallel transport ([abstract](https://files-www.mis.mpg.de/mpi-typo3/events-files/abstract_876.pdf)). It is the closest conceptual predecessor, but has no contextual LM field, predictive connection selection, or steering-transfer test.
- Manifold Steering fits activation and behavior splines and uses Hellinger/Fisher sphere geometry for outputs ([paper](https://arxiv.org/abs/2605.05115)). It selects paths/metrics, not Amari connections for transporting a semantic field.

The gap found in this audit is narrow and defensible:

> Existing work has (i) context-dependent semantic vector fields, (ii) exact softmax information geometry, and (iii) Fisher-aware steering, but I found no work that estimates which (e/LC/m/α) connection makes an empirical contextual semantic field closest to parallel.

The novelty confidence is about **0.8**, not certainty. This is a **novel application and diagnostic**, not a claim that the underlying differential geometry is new. Before a publication-level priority claim, refresh backward/forward citations and search Semantic Scholar/OpenAlex for work appearing after the audit date. Use “to our knowledge,” not “the first ever.”

Unsafe novelty claims are:

- context-dependent vector fields are new;
- parallel transport for embeddings is new;
- Fisher geometry for steering is new;
- choosing (α) for an NLP representation is new;
- curvature alone explains steering failure.

The likely unoccupied intersection is

\[
\boxed{
\text{contextual field}
+\text{predictive softmax geometry}
+\text{α-connection selection by parallelism}.
}
\]

## Exact mathematical setting

Let (h\in H\subseteq\mathbb R^d) be the vector immediately entering the LM head and

\[
p_h(y)
=
\frac{\exp(b_y+w_y^\top h)}
{\sum_z\exp(b_z+w_z^\top h)}.
\]

Write

\[
\psi(h)=\log\sum_y\exp(b_y+w_y^\top h),
\qquad
G(h)=\nabla^2\psi(h).
\]

With μ the probability-weighted mean decoder row and (\widetilde w_y=w_y-\mu),

\[
G(h)
=
\sum_y p_h(y)\widetilde w_y\widetilde w_y^\top
=
W^\top(\operatorname{diag}p-pp^\top)W.
\]

This is the pullback Fisher metric of next-token distributions. Define its cubic tensor

\[
C_h(a,v,z)
=
\nabla^3\psi(h)[a,v,z]
=
\sum_y p_h(y)
(\widetilde w_y^\top a)
(\widetilde w_y^\top v)
(\widetilde w_y^\top z).
\]

Assume the decoder is minimal, equivalently (Wu\in\operatorname{span}\{\mathbf1\}) only for (u=0). Then (G) is positive definite. Otherwise, every statement is made on the identifiable quotient/eigenspace, and that restriction must be reported.

Define (A_{h,a}) by

\[
G(h)(A_{h,a}v,z)=C_h(a,v,z),
\qquad
A_{h,a}v=G(h)^{-1}C_h(a,v,\cdot).
\]

In the natural (h)-coordinates, the Amari (α)-connection is

\[
\boxed{
\nabla^{(\alpha)}_aV
=
DV[h][a]
+\frac{1-\alpha}{2}A_{h,a}V(h).
}
\]

The three canonical cases are (e: α=1), (LC: α=0), and (m: α=-1). Geometry alone does not privilege LC: the right connection depends on what “the same transformation” is meant to preserve.

## Result 1: a local edge residual estimates a covariant derivative

Let (V) be a (C^1) vector field and (h'=h+\varepsilon a). Along the declared straight natural-coordinate segment, (α)-parallel transport satisfies

\[
P^{(\alpha)}_{h\to h'}V(h)
=
V(h)
-\varepsilon\frac{1-\alpha}{2}A_{h,a}V(h)
+O(\varepsilon^2).
\]

Taylor expansion gives (V(h')=V(h)+\varepsilon DV[h][a]+O(\varepsilon^2)). Subtracting,

\[
\boxed{
V(h')-P^{(\alpha)}_{h\to h'}V(h)
=
\varepsilon\nabla^{(\alpha)}_aV(h)+O(\varepsilon^2).
}
\]

Therefore the transport residual on a local context graph is a consistent finite-difference estimator of connection-parallelness. This supplies the missing bridge between “the arrows vary with context” and “the arrows may be one intrinsic object transported through a nonconstant metric.”

For finite data, use the dimensionless endpoint residual

\[
E_{\alpha}(i,j)
=
\frac{
\left\|V_j-P^{(\alpha)}_{i\to j}V_i\right\|_{G_j}
}{
\|V_j\|_{G_j}+\|P^{(\alpha)}_{i\to j}V_i\|_{G_j}+\epsilon
}.
\]

The exact flat transports are

\[
P^e_{i\to j}v=v,
\qquad
P^m_{i\to j}v=G(h_j)^{-1}G(h_i)v.
\]

LC transport solves

\[
\dot U(t)=-\tfrac12A_{\gamma(t),\dot\gamma(t)}U(t),
\qquad
\gamma(t)=h_i+t(h_j-h_i),
\]

and can be integrated by RK4. The path is part of the definition: LC transport is path dependent when curvature is nonzero.

## Result 2: a closed-form semantic α-index

For a local directed edge (i\to j), set

\[
a_{ij}=h_j-h_i,
\qquad
\Delta V_{ij}=V_j-V_i,
\qquad
B_{ij}=A_{h_i,a_{ij}}V_i,
\qquad
\beta=\frac{1-\alpha}{2}.
\]

The first-order residual is (ΔV_{ij}+βB_{ij}). On a frozen set of local edges, minimize

\[
L(\beta)
=
\sum_{(i,j)}q_{ij}
\|\Delta V_{ij}+\beta B_{ij}\|_{G_i}^2,
\]

where (q_{ij}) are fixed nonnegative weights. If at least one (B_{ij}\ne0), the objective is strictly convex and

\[
\boxed{
\widehat\beta
=
-\frac{
\sum q_{ij}\langle\Delta V_{ij},B_{ij}\rangle_{G_i}
}{
\sum q_{ij}\|B_{ij}\|_{G_i}^2
},
\qquad
\widehat\alpha=1-2\widehat\beta.
}
\]

This scalar is the proposed **semantic α-index**. Do not clip the primary estimate:

- (α̂\approx1): a global hidden vector is locally adequate;
- (α̂\approx0): Fisher-metric-compatible transport explains the variation;
- (α̂\approx-1): expectation-coordinate constancy explains it;
- (α̂) far outside ([-1,1]), or low held-out improvement: none of the canonical transports explains the field.

This has a clean identifiability condition: if every cubic correction (B_{ij}) vanishes, the samples cannot distinguish the connections locally. The estimator is coordinate invariant in its infinitesimal limit because it comes from covariant derivatives and Fisher inner products. On finite edges, charts differ at second order; use local neighbors and report distance-stratified results.

## Result 3: a global obstruction

The attractive phrase “one global semantic vector under LC transport” is generally impossible, not merely empirically doubtful.

In a dually flat family let (D=\nabla^e) and (K=\nabla^{LC}-D). Since (\nabla^m=D+2K) is flat,

\[
d_DK=-2K\wedge K.
\]

Because (\nabla^{(\alpha)}=D+(1-\alpha)K), its curvature is

\[
\boxed{
R^{(\alpha)}=(1-\alpha^2)R^{LC}.
}
\]

Proof: expand the curvature of (D+sK) as

\[
R(D+sK)=s\,d_DK+s^2K\wedge K,
\]

substitute (s=1-\alpha) and (d_DK=-2K\wedge K), then compare with (R^{LC}=-K\wedge K).

If (V) is parallel on an open set, then (R^{(\alpha)}(X,Y)V=0) for all (X,Y). Therefore, wherever the curvature operators have trivial common kernel, no nonzero global (α)-parallel field exists for (|α|<1). On the saturated categorical simplex of dimension at least two, the Fisher--LC metric has constant sectional curvature (1/4), so this obstruction applies to every interior (α)-connection.

Interpretation:

- (e) and (m) can support global affine representations because they are flat;
- LC can support meaningful **local** comparison, but curvature obstructs a path-independent global semantic vector;
- fitting (α) is a local empirical law, not permission to ignore holonomy globally.

The curvature identity is classical information/Hessian geometry. Its role here is to delimit exactly what the semantic connection profile can and cannot mean.

## A secondary diagnostic: local e-versus-LC defect

For context direction (a) and semantic vector (v), define

\[
\boxed{
\chi_h(a,v)
=
\frac12
\frac{
\sqrt{C(a,v,\cdot)^\top G^{-1}C(a,v,\cdot)}
}{
\sqrt{(a^\top Ga)(v^\top Gv)}
}.
}
\]

It is the leading relative (e)-versus-LC transport discrepancy per unit Fisher context distance:

\[
\chi_h(a,v)
=
\lim_{\varepsilon\to0}
\frac{
\|P^e v-P^{LC}v\|_G
}{
d_G(h,h+\varepsilon a)\|v\|_G
}.
\]

Use (χ) as a secondary predictor of where constant-vector transfer fails. This avoids the circular claim that LC must be correct: first estimate which connection fits, then ask whether the predicted (e)-LC discrepancy explains the observed difference.

All required contractions are CPU-computable without storing a (d^3) tensor. If (z_y=w_y-\mu), then

\[
C(a,v,\cdot)
=
\sum_y p_y(z_y^\top a)(z_y^\top v)z_y.
\]

Form (G) once at each source, reuse its factorization over outgoing graph edges, and process vocabulary rows in chunks.

## Empirical claims

### Primary null

\[
H_0:\quad \alpha_T=1
\]

for every operation (T). This is the precise constant-global-vector hypothesis in natural hidden coordinates.

### Primary alternative

At least one controlled operation has a stable (α_T\ne1), and transport under its fitted connection reduces held-out field residual relative to (e)-transport.

### Stronger structured alternative

Different transformations select reproducibly different connections. For example, a grammatical transformation might be close to (e), while contextual framing might be closer to LC. Such a result would show that “the geometry of representations” is not one universal curvature label; the operative connection depends on the invariant being compared.

### Secondary predictions

1. The local defect (χ_h(a,V_i)) predicts ordinary-vector transfer error beyond Fisher distance, Euclidean distance, vector norm, and predictive entropy.
2. The fitted (α_T) changes from random initialization toward a stable operation-specific value during pretraining.
3. If LC wins locally, its advantage decays with edge length unless integrated transport is used.

## CPU experiment

There are two sensible data designs. Freeze one as primary before seeing the result.

### Design A, recommended: reproduce a small contextual field

Adapt Hu et al.'s object using 200--500 single-token nouns, a neutral template, and two or three target frames chosen from categorization, perceptual, situational, affective, and world-knowledge contexts. For transformation (T), define

\[
V_T(h_i)=h_i^T-h_i^{neutral}.
\]

This is closest to the literature gap and supplies many base points for a local graph. Use Pythia-14M for engineering. If the fields pass manipulation and stability checks, confirm on Pythia-70M.

### Design B: tightly controlled grammatical pairs

Use at least 100 minimal prompt pairs per operation across distinct lexical items and template families:

1. subject-number change, checked by singular/plural auxiliary or verb log odds;
2. present/past cue change, checked by inflection log odds.

For pair (i), extract the **post-final-normalization vector entering the LM head**:

\[
h_i^- = h(c_i^-),
\qquad
h_i^+ = h(c_i^+),
\qquad
V_i=h_i^+-h_i^-.
\]

This (V_i) is the initial velocity of the explicitly declared intervention path (h_i^-+s(h_i^+-h_i^-)). Do not claim that the line is a naturally generated prompt path.

Design B has stronger linguistic control but may give a less densely sampled base manifold. It is an excellent robustness dataset if Design A is primary.

### Frozen pilot

Use only the pilot split to freeze:

- viable transformations;
- tokenizer and prompt rules;
- manipulation checks;
- graph neighborhood (k\in\{5,10\});
- the metric spectral threshold;
- whether full hidden space or a declared resolution subspace is analyzed.

Do not remove confirmation cases after inspecting their connection residuals.

### Local graph and transport

For each transformation and checkpoint:

1. Build a (k)-nearest-neighbor graph among neutral/base representations. Use a symmetrized local Fisher approximation for primary neighbor distance and Euclidean neighbors as a robustness graph.
2. Split by **template family and lexical item**, never by individual edge.
3. On training edges, estimate (α̂_T) using the closed form.
4. On held-out edges, compare integrated (e), LC, (m), and fitted-(α̂_T) transports using (E_α(i,j)).
5. Bootstrap model seed, template family, and lexical item. Graph edges are not independent observations.

The full-vocabulary calculation is primary for 14M. If 70M is slow, a fixed vocabulary truncation or 8--32-dimensional resolution subspace may be used only after validating it against a full-vocabulary subset; label the result as approximation-restricted.

### Training dynamics and confirmation

Run PolyPythia-14M seeds 1--5 at

\[
\texttt{step0},\quad
\texttt{step512},\quad
\texttt{step10000},\quad
\texttt{step50000},\quad
\texttt{step143000}.
\]

If the manipulation checks and conditioning are acceptable, confirm at `step0` and `step143000` on three Pythia-70M seeds. Pythia/PolyPythia supplies the required multiple seeds and checkpoint history ([model collection](https://huggingface.co/collections/EleutherAI/polypythias), [repository](https://github.com/EleutherAI/pythia)). Load one model/checkpoint at a time and cache only hidden states and derived geometry.

### Primary statistic

For each transformation, report the paired held-out improvement

\[
S_T
=
\operatorname{median}_{(i,j)}
\log\frac{E_e(i,j)+\epsilon}
{E_{\widehat\alpha_T}(i,j)+\epsilon}.
\]

Success requires:

- a positive hierarchical-bootstrap 95% interval for (S_T);
- replication in at least two trained seeds;
- improvement over a freely fitted scalar rescaling of (V_i);
- improvement over norm-matched random and opposite-sign corrections;
- stability over predeclared spectral thresholds;
- convergence from 8 to 16 RK4 steps.

Report unrestricted alpha-hat, its interval, the three canonical residuals, and fitted-connection held-out residual. A fitted alpha without held-out improvement is overfitting, not a discovery.

## Essential controls

1. **Random initialization:** any trained-model claim must differ from `step0`.
2. **Distance matching:** connection terms grow with edge size; compare within Fisher-distance strata.
3. **Euclidean baseline:** raw cosine and residual of the same fields.
4. **Scalar rescaling:** distinguishes transport rotation from a simple norm change.
5. **Matched random correction:** same target Fisher norm as the geometric correction.
6. **Opposite sign:** detects benefits caused only by an extra degree of freedom.
7. **Shuffled field:** permute field vectors among base points within a transformation.
8. **Full vocabulary:** dynamic top-k changes the manifold and is not the 14M primary analysis.
9. **Float64 and spectral audit:** no unreported ridge in the primary geometry.
10. **Path audit:** e/m are path independent; for LC/fitted alpha, compare the straight path with a two-segment local path on a subset.
11. **Manipulation check:** report failures rather than filtering the confirmation set after observing geometry.
12. **Template-level split:** prevents nearly duplicated sentences on both sides of the split.

## Falsification and useful negative theses

The proposal is falsified as a claim against ordinary vector arithmetic if:

- e-transport is best on held-out transformations;
- alpha-hat is unstable across seeds or graph scales;
- fitted transport does not beat scalar and random corrections;
- any benefit vanishes after distance matching;
- results depend on a single spectral cutoff;
- the small model never passes the semantic or linguistic manipulation checks.

Those outcomes still support valid MSc conclusions:

- **e wins:** “Context-dependent semantic fields are nonuniform in Euclidean plots, but predictive information geometry does not explain the variation; ordinary natural-coordinate vectors transfer best locally.”
- **None wins:** “The variation is genuine semantic heterogeneity rather than a connection artifact, supporting learned vector fields such as SVF.”
- **The model fails the manipulation:** the exact estimator and numerical study remain, but the thesis must not pretend to settle semantic geometry.

## What counts as a nontrivial result

“The Fisher metric is non-Euclidean” is not enough; that is already known. A strong result is one of:

1. a reproducible alpha-hat different from 1 with held-out transport improvement and controls;
2. transformation-specific connection selection that evolves with training;
3. a well-powered null result showing that predictive Fisher transport does **not** explain field variation despite measurable curvature or nonmetricity.

Any of these distinguishes hypotheses that current vector-field work leaves conflated:

\[
\text{coordinate variation}
\quad\text{vs}\quad
\text{geometric transport}
\quad\text{vs}\quad
\text{irreducible semantic heterogeneity}.
\]

## Scope and schedule

- **Weeks 1--2:** lock theorems, synthetic tests, stimulus generator, and manipulation checks.
- **Weeks 3--4:** frozen 14M pilot; choose viable transformations and graph scale.
- **Weeks 5--6:** five 14M seeds and five checkpoints; exact e/LC/m/alpha residuals.
- **Week 7:** controls, distance strata, spectral and path sensitivity.
- **Weeks 8--9:** 70M confirmation if the pilot is viable.
- **Weeks 10--12:** bootstrap analysis, figures, literature refresh, and writing.

Stop expanding scope after the pilot. Intermediate transformer layers require Jacobian pullbacks and are a PhD-scale extension, not part of the primary MSc claim.

## Implemented proof of concept

The repository now contains:

- `src/predictive_geometry/field.py`: exact e/m transports, RK4 alpha-transport, the local defect scalar, and closed-form semantic alpha fitting;
- `tests/test_semantic_field.py`: tests covering exact flat transports, LC metric preservation, second-order local error, scale invariance, identifiability, and alpha recovery;
- `experiments/synthetic_semantic_alpha.py`: deterministic synthetic recovery for e, LC, m, and an intermediate connection, with numerical-integration refinement reported only for the RK4 cases.

These validate computation only. They are not evidence about language models.

The deterministic noisy synthetic run produced:

| Generating connection | True alpha | Estimated alpha | Absolute error |
|---|---:|---:|---:|
| Exponential | 1.00 | 0.999939 | 0.000061 |
| Levi--Civita | 0.00 | 0.000033 | 0.000033 |
| Mixture | -1.00 | -1.000026 | 0.000026 |
| Intermediate | 0.35 | 0.349811 | 0.000189 |

The exponential case correctly estimates alpha but has essentially no explainable field variation: its generated vectors are constant apart from measurement noise. This is a useful warning that parameter recovery and explanatory power must both be reported.

## Thesis wording

Recommended title:

> **Which Connection Carries a Concept? Information-Geometric Transport of Semantic Vector Fields in Language Models**

Defensible abstract claim:

> We test whether contextual semantic transformations in pretrained language models are approximately parallel under a distinguished Amari alpha-connection of the model's predictive softmax manifold. We derive a local closed-form estimator of the connection parameter from Fisher-weighted transformation-field variation and evaluate whether the selected connection predicts held-out transformations and activation-steering transfer.

More conservative title:

> **Testing Exponential, Levi--Civita, and Mixture Transport of Contextual Transformations in Small Language Models**
