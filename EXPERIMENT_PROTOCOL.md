# CPU-Only Experiment Protocol

Date: 2026-08-17
Primary objects: ambient decoded intervention geometry and declared support-relative charts

## 1. Research question

For controlled semantic operations in an autoregressive language model, which geometry best predicts held-out composition?

1. Exponential connection: ordinary addition in natural hidden/logit coordinates.
2. Mixture connection: ordinary addition in probability coordinates.
3. Fisher--Levi-Civita connection: metric-compatible parallel transport.

The experiment also asks whether exact Fisher--LC sectional curvature of semantic planes changes during pretraining and predicts failures of flat composition, and whether a compact teacher connection packet improves large-to-small model distillation beyond output and metric matching.

The goal is not to demonstrate that the full categorical simplex is curved; that curvature is already known to be \(1/4\).

## 2. Pre-registered objects and claims

Claims are reported in three noninterchangeable classes. **Mathematical
identities** (for example LC metric compatibility and flat-connection closure
on their natural domains) are proved properties and are tested only as
implementation oracles. **Numerical validation claims** establish that a
specific implementation attains declared residual and refinement tolerances.
**Empirical scientific claims** concern learned models and require the sampling,
controls, uncertainty, and multiplicity rules below. Passing an identity check
does not count as empirical support.

### Object A: decoded intervention manifold

\[
\mathcal M_{\mathrm{dec}}
=\{\operatorname{softmax}(Wh+b):h\in\mathbb R^d\}.
\]

At a natural context \(c\), the base point is \(h(c)\), but the tangent space consists of arbitrary infinitesimal final-state interventions. The exact Fisher metric is

\[
G=W^\top(\operatorname{diag}p-pp^\top)W.
\]

Its sectional curvature is calculated analytically from the third cumulants of the rows of \(W\); see the authoritative derivation in [paper/main.tex](paper/main.tex) and the historical development notes in [PROOFS_AND_NOVELTY.md](PROOFS_AND_NOVELTY.md).

This is an **ambient intervention manifold**, not automatically the manifold of naturally reachable activations. For a post-LayerNorm state, natural activations satisfy at least the affine constraint

\[
\sum_i\frac{h_i-\beta_i}{\gamma_i}=0
\]

when all gains are nonzero, and are nearly fixed-radius when the normalization epsilon is negligible. Support-relative claims must restrict directions and controls to the reachable tangent chart or explicitly pull the decoder geometry back through the final normalization map.

### Object B: hard-prompt output quadrilateral

For two controlled operations \(A,B\), collect

\[
p_{00},\quad p_{10},\quad p_{01},\quad p_{11}.
\]

This is a held-out prediction task in the full output simplex. It is not by itself an intrinsic-curvature estimate.

### Object C: explicit intervention surface

For fixed-length, disjoint token substitutions, interpolate input embeddings or final hidden states with a declared map

\[
F(a,b)=p_\theta(\cdot\mid z(a,b)).
\]

Its Gaussian curvature belongs to this selected surface. It cannot automatically be called the sectional curvature of a larger empirical hidden manifold.

### Object D: aligned teacher--student intervention chart

For frozen teacher \(T\), student \(S\), and a shared low-dimensional coordinate
\(z\in U\subset\mathbb R^m\), define

\[
F_T(c,z)=Q_Tp_T(\cdot\mid I_T(c,z)),
\qquad
F_S(c,z)=Q_Sp_S(\cdot\mid I_S(c,z)).
\]

The intervention coordinate and outcome space are shared; hidden dimensions and
hidden coordinates need not align. The primary Pythia study uses a shared
tokenizer and \(Q_T=Q_S=I\). Connection comparisons are valid only after this
common-chart identification. See
[PREDICTIVE_CONNECTION_DISTILLATION.md](PREDICTIVE_CONNECTION_DISTILLATION.md)
for the complete packet and training contract.

## 3. Hypotheses

### H1: training effect

For matched contexts and semantic planes,

\[
K_{\mathrm{final}}-K_{\mathrm{step0}}
\]

has a reproducible nonzero distribution across templates and random seeds.

### H2: semantic-plane structure

Semantic-plane sectional curvature differs from spectrum-matched control planes at the same context. The primary null preserves the semantic plane's leverage across Fisher eigendirections; Fisher-Haar planes are a secondary isotropic null only.

### H3: connection selection

Held-out semantic composition errors differ systematically across the exponential, mixture, and Levi--Civita connections.

### H4: geometric relevance

Curvature or integrated holonomy predicts composition error after controlling for entropy, softmax-entropy cumulant-profile baselines, Fisher edge lengths, operation magnitude, metric conditioning, and random-plane curvature.

### H5: reproducibility

The sign and ranking of the effects replicate across training seeds and prompt templates.

### H6: predictive connection distillation

At matched student NLL, capacity, data order, and training compute, transferring
the teacher's predictive \((G,L,C)\) packet improves held-out semantic transport
and at least one behavioral transfer outcome beyond output-only,
centered-logit Jacobian, square-root Jacobian, and metric-only distillation.

The null result is meaningful: exponential/logit transport may remain the best rule even when Fisher--LC curvature is nonzero.

## 4. Experiment A: numerical validation

This stage must pass before any model result is interpreted.

1. Fisher square-root sphere distance, log, exponential, and transport identities.
2. Mixture and exponential log/exp round trips.
3. Exact duality of mixture and exponential transports.
4. Zero loop holonomy for the two flat connections.
5. Spherical triangle LC holonomy equals spherical excess.
6. Saturated three-category softmax gives \(K=1/4\).
7. Near-boundary saturated tests with
   \[
   p=(1-2\varepsilon,\varepsilon,\varepsilon),
   \quad \varepsilon\in\{10^{-1},10^{-3},10^{-6},10^{-9}\}.
   \]
8. Curvature invariance under an arbitrary invertible hidden-coordinate transformation.
9. Rank-deficient decoders and collinear planes are rejected rather than regularized silently.
10. Synthetic quadrilaterals generated by each connection recover their generating rule.
11. Vocabulary-block accumulation reproduces the full-vocabulary curvature calculation.
12. Direct cubic actions agree with materialized Amari--Chentsov operators.
13. Relative-log spectral controls preserve the declared per-band leverage exactly.
14. Paired Bernoulli square-root jets converge while their nonzero-alpha connection defect grows.

Current status: the complete automated CPU suite passes.

## 5. Experiment B: exact decoded-manifold curvature

### 5.1 Data

Discovery run:

- Pythia-14M for code and CPU smoke tests;
- Pythia-70M for the first substantive run;
- checkpoints \(\texttt{step0}\), \(\texttt{step512}\), \(\texttt{step10000}\), \(\texttt{step50000}\), and \(\texttt{step143000}\);
- 12--24 base factorial prompts;
- 20 control planes per base point for execution-only smoke tests;
- 999 block-spectrum-matched controls per base point for any inferential run, giving minimum one-sided plus-one tail rank \(0.001\); this rank is a p-value only under the preregistered conditional group-invariance null in Section 5.3.

Confirmation run:

- the number of independently trained Pythia-70M runs returned by the
  Section 9 simulation-based 80%-power calculation, not a fixed five-seed
  rule;
- five to seven fixed checkpoints, treated as repeated measurements;
- a versioned prompt/operation manifest sized by the same power and
  interval-width analysis, with target frame, inclusion rules, and selection
  mechanism recorded; and
- the frozen manifest and split committed before inspecting curvature or
  behavioral outcomes.

Pythia is appropriate because the [official project](https://github.com/EleutherAI/pythia) provides 154 checkpoints and multiple small model sizes under a consistent training setup.

### 5.2 Semantic directions

For a base context and two controlled continuous interventions \(A_t,B_t\), the primary semantic directions are tangent derivatives

\[
u=\left.\frac{d}{dt}\right|_{t=0}h(A_t c),
\qquad
v=\left.\frac{d}{dt}\right|_{t=0}h(B_t c).
\]

Estimate them with symmetric differences at step sizes \(s,s/2,s/4\) and require a convergence plateau. The finite chords \(h(Ac)-h(c)\) and \(h(Bc)-h(c)\) remain a secondary analysis of conventional exponential-coordinate interventions; they are not treated as connection-neutral tangent fields.

Reject a plane when

\[
\frac{
(u^\top Gu)(v^\top Gv)-(u^\top Gv)^2
}{(u^\top Gu)(v^\top Gv)}<10^{-4}.
\]

This is a plane-degeneracy check, not a significance test.

### 5.3 Matched and isotropic controls

Fisher-orthonormalize the semantic plane and write it in whitened metric-eigenvector coordinates. Partition the Fisher spectrum into contiguous bands, joining adjacent eigenvalues when their relative gap is at most \(10^{-6}\). The primary control applies a Haar orthogonal rotation inside every multi-axis band and a random sign in every singleton band. This preserves each band's \(2\times2\) leverage contribution while randomizing orientation relative to the decoder's third moments, and it is invariant to the arbitrary eigenbasis inside a repeated eigenspace.

The relative-gap rule can produce mostly singleton bands and therefore a narrow sign-flip orbit. Report every band size and preregister a required coarser sensitivity analysis that groups axes by decades of \(\lambda_{\max}/\lambda_i\) (relative-log mode with width 1.0), plus nearby widths. The coarser orbit preserves leverage only at its declared band resolution, so it tests robustness to the conditioning choice rather than replacing the primary null silently. Report conclusions that change with the band rule as control-sensitive.

As a secondary null, sample Gaussian directions with covariance \(G^{-1}\), then Fisher-orthonormalize them. This defines an isotropic Fisher-Haar plane, but it does not match the semantic plane's spectral alignment and cannot by itself support a semantics-specific interpretation.

Reuse the same computed \(G\) for every plane at that context.

The inferential null for this control is explicit: conditional on the semantic plane's leverage in each preregistered spectral band, its curvature statistic is invariant under the product of those within-band orthogonal groups. For \(B\) controls, report the one-sided plus-one control-tail ranks

\[
q_+=\frac{1+\#\{K_b\ge K_{\mathrm{sem}}\}}{B+1},
\qquad
q_-=\frac{1+\#\{K_b\le K_{\mathrm{sem}}\}}{B+1}.
\]

These ranks are calibrated randomization p-values only under the stated conditional group-invariance null; otherwise they are descriptive reference-tail ranks. The curvature-family alternative is two-sided, \(\min(1,2\min(q_+,q_-))\), with Holm correction across the preregistered context-by-checkpoint family. This is a separate mechanistic family: it neither enters nor establishes the Section 8.3 distillation claim. If curvature and distillation are promoted to co-primary program-level claims, freeze an outer multiplicity procedure across their global hypotheses before data inspection. Do not convert fewer than 30 control draws into Gaussian z-tests. The completed eight-plane pilot has minimum attainable tail rank \(1/9\) and is execution-only.

### 5.4 Exact computation

Let \(z_y=w_y-\mathbb E_pw_y\). Compute

\[
G=\sum_yp_yz_yz_y^\top
\]

in float64. For each plane, compute

\[
c_{uv}=\sum_yp_y(z_y^\top u)(z_y^\top v)z_y
\]

and the analogous \(c_{uu},c_{vv}\). Then

\[
K(u,v)=
\frac{
\tfrac14(c_{uv}^\top G^{-1}c_{uv}-c_{uu}^\top G^{-1}c_{vv})
}{
(u^\top Gu)(v^\top Gv)-(u^\top Gv)^2
}.
\]

Do not add a ridge in the primary analysis. A ridge changes the metric. If \(G\) is numerically rank deficient, report a resolution-restricted pseudoinverse sensitivity analysis separately; do not call it the full-manifold result.

### 5.5 Numerical acceptance

- compute probabilities from float64 logits;
- form centered covariance, then symmetrize \(G\);
- report all eigenvalues or at least extrema and condition number;
- use the preregistered metric-rank gate \(\lambda_{\min}/\lambda_{\max}>10^{-12}\) in the primary implementation and report sensitivity to stricter thresholds;
- require normalized plane Gram determinant at least \(10^{-4}\), separately from the metric-rank gate;
- require relative linear-solve residual below \(10^{-10}\);
- require condition-number times that residual below \(10^{-4}\), as a conservative forward-error sensitivity gate;
- require curvature to change by at most \(10^{-10}\max(1,|K|)\) after Fisher-orthonormalizing the same plane;
- independently recompute every reported semantic-plane curvature in vocabulary blocks of 4096 rows and require agreement within \(10^{-7}\max(1,|K_{\rm full}|,|K_{\rm chunked}|)\).

### 5.6 Complexity

For vocabulary size \(V\) and hidden size \(d\):

- metric construction: \(O(Vd^2)\);
- factorization: \(O(d^3)\);
- each additional plane: \(O(Vd+d^2)\).

This is realistic on CPU for Pythia-14M and Pythia-70M. No model training is required.

## 6. Experiment C: held-out connection transfer

### 6.1 Primary prediction

Use the observed effect \(p_{00}\to p_{10}\) to predict \(p_{11}\) at base \(p_{01}\):

\[
\widehat p_{11}^\nabla
=\operatorname{Exp}_{p_{01}}^\nabla
\left(P_{p_{00}\to p_{01}}^\nabla
\operatorname{Log}_{p_{00}}^\nabla p_{10}\right).
\]

Also reverse the orientation by transporting \(p_{00}\to p_{01}\) to \(p_{10}\). Report both; do not average away LC path dependence.

### 6.2 Closed forms

Mixture:

\[
\widehat p^m=p_{01}+p_{10}-p_{00}.
\]

Exponential:

\[
\widehat p^e_i
\propto\frac{p_{01,i}p_{10,i}}{p_{00,i}}.
\]

Levi--Civita: use the radius-two Fisher sphere log, minimal-geodesic transport, and exponential.

These three closed forms are **ambient full-simplex** rules. Only the exponential
rule is guaranteed to remain inside a lower-dimensional decoded softmax family.
Ambient mixture and LC paths generally leave \(\mathcal M_{\mathrm{dec}}\), even
when their endpoints remain valid categorical distributions.

The later intrinsic decoder-family comparison must instead use:

- exponential coordinates \(h\);
- mixture/expectation coordinates
  \(\eta(h)=\nabla\psi(h)=\mathbb E_{p_h}W_Y\), with numerical inversion of the
  moment map for finite endpoints;
- LC transport obtained from
  \(\Gamma^k_{ij}=\tfrac12G^{k\ell}C_{ij\ell}\) along paths inside decoded
  hidden space.

Results from these two comparison levels must not be pooled.

### 6.3 Primary outcome

\[
e_\nabla=d_{FR}(\widehat p_{11}^\nabla,p_{11}).
\]

This is an error, so lower values are better. In the confirmatory analysis the
corresponding behavioral score is \(S_{\rm beh}=-e_\nabla\), so every
superiority contrast is oriented with higher values better. A different
behavioral endpoint may replace it only if one scalar endpoint, its direction,
transformation, and aggregation are frozen in the versioned analysis manifest
before confirmatory training; no best-looking endpoint may be selected later.

Secondary outcomes:

- \(D_{KL}(p_{11}\|\widehat p)\);
- reverse KL;
- Jensen--Shannon divergence;
- top-token agreement;
- task-token log-odds error.

### 6.4 Feasibility is an outcome

Mixture translation can leave the simplex. LC sphere exponential can leave the positive square-root orthant. Do not clip, square negative coordinates, or silently project. Report feasibility rate separately.

Exponential transport is globally feasible in the open simplex and, for a linear softmax family, exactly equals hidden/logit arithmetic. This is a structural advantage, not a numerical accident.

If hard operations are too large for a three-way comparison, use one of these pre-specified alternatives:

1. smaller continuous interventions with observed intermediate targets;
2. a fixed vocabulary partition selected once, including a fixed “other” bin;
3. intrinsic transport on an explicit continuous decoder or soft-prompt surface.

Never use a prompt-dependent top-\(k\) support: changing support makes the geometry discontinuous and comparisons unfair.

### 6.5 Nonmetricity outcome

For the transported operation \(u\), record

\[
\delta_\nabla
=\log\frac{\|P^\nabla u\|_{FR}}{\|u\|_{FR}}.
\]

LC should give zero up to numerical error. Exponential and mixture transport generally do not. This directly measures the price paid for flatness.

### 6.6 Fitted-alpha estimator

The closed-form fitted-alpha statistic uses the first-order residual

\[
\Delta V+\frac{1-\alpha}{2}G^{-1}C(\Delta h,V,\cdot).
\]

It is an infinitesimal estimator, not an exact finite-edge likelihood. Integrated transport has an \(O(\ell^2)\) local truncation relative to an \(O(\ell)\) connection signal, so the fitted parameter has generic \(O(\ell)\) finite-edge bias, whose sign is decoder dependent. Every fitted-alpha result must therefore:

1. report the maximum and distribution of source-Fisher edge lengths \(\ell\);
2. repeat the fit on at least three nested length scales and extrapolate to \(\ell=0\), or fit \(\alpha\) by integrated transport directly;
3. generate synthetic validation targets with integrated transport, rather than the same first-order formula used by the estimator;
4. fit on one split and evaluate the selected \(\alpha\) on held-out edges and behavioral endpoints;
5. report unrestricted \(\widehat\alpha\), fixed \(e\), LC, and \(m\) residuals.

The closed-form fit is accepted only when the dimensionless excitation ratio

\[
\rho_{\rm exc}
=\frac{\sum_i\|A_{\Delta h_i}V_i\|_{G_i}^2}
{\sum_i\|\Delta h_i\|_{G_i}^2\|V_i\|_{G_i}^2}
\]

is at least \(10^{-8}\). Otherwise alpha is weakly identified and no point estimate is reported.

RK4 step convergence applies only to connections evaluated by numerical integration. The exact exponential and mixture closed forms have no step-size error.

### 6.7 Theory-coverage diagnostic

Whenever the square-root Levi--Civita stability theorem is used to interpret a
cross-model or approximation comparison, estimate its constants on the entire
declared continuous path, not only at the endpoint contexts. Record

\[
B_1=\sup\|D\psi\|_{\mathrm{op}},\qquad
B_2=\sup\|D^2\psi\|_{\mathrm{op}},\qquad
\lambda=\inf\lambda_{\min}(g),
\]

for both aligned maps using common envelopes, together with

\[
\delta_1=\sup\|D\psi-D\widetilde\psi\|_{\mathrm{op}},\qquad
\delta_2=\sup\|D^2\psi-D^2\widetilde\psi\|_{\mathrm{op}},
\qquad L_\gamma.
\]

For an affine softmax chart and directions \(u,v\), the exact directional
second-derivative diagnostic is

\[
\|D^2\psi[u,v]\|_2
=\frac12\sqrt{\mathbb E_p[S_Y(u)^2S_Y(v)^2]}.
\]

Consequently the pointwise bilinear operator norm is exactly

\[
\|D^2\psi_x\|_{\mathrm{op}}
=\frac12\sqrt{
  \sup_{\|u\|_2\le1}\mathbb E_{p(x)}[S_Y(u)^4]
}.
\]

Directional samples provide lower bounds and convergence diagnostics for
\(B_2\); they are not a proof of the operator supremum unless the directional
optimization is certified.

Compute

\[
D_{LC}
=\lambda^{-1}(B_1\delta_2+B_2\delta_1)
+2B_1^2B_2\lambda^{-2}\delta_1
\]

and report the certified transport defect

\[
L_\gamma D_{LC}
\min\left\{
\exp\left(\frac{B_1B_2}{\lambda}L_\gamma\right),
\frac{B_1^2}{\lambda}
\right\}.
\]

The polynomial branch follows from Levi--Civita metric compatibility and means
that a large generic Gronwall exponent does not establish an intrinsic
short-path barrier. Pointwise metric spectra from the historical pilot do not
measure \(B_2,\delta_1,\delta_2\), or pathwise extrema. Report the actual
transport defect beside the certificate, the background-coordinate and Fisher
path lengths, the chart and its scaling, and any refinement study used for the
suprema. The helper
`predictive_geometry.levi_civita_stability_bounds` implements these formulas.

When comparing two Levi--Civita transports, also report the coordinate-invariant
mixed-norm diagnostic when feasible. For
\(A=\nabla^{g,LC}-\nabla^{\widetilde g,LC}\),

\[
\|P^{g,LC}_{1\leftarrow0}-P^{\widetilde g,LC}_{1\leftarrow0}\|_{
\widetilde g_0\to g_1}
\le
\int_0^1
\|A_{\gamma(t)}\|_{\widetilde g,\widetilde g\to g}
\|\dot\gamma(t)\|_{\widetilde g}\,dt.
\]

This form has no separate chart-condition-number multiplier, but it does not
eliminate rank sensitivity: raising-index instability is contained in the mixed
norm of \(A\).

## 7. Experiment D: intrinsic semantic surface and holonomy

Use factorial prompts with two disjoint, fixed-position token substitutions. Define a declared continuous interpolation whose four corners reproduce the four hard prompts exactly.

With

\[
x(a,b)=2\sqrt{p(a,b)},
\]

estimate first and second derivatives. Let

\[
J=[x_a,x_b],\qquad g=J^\top J,
\]

and project second derivatives onto the normal space:

\[
B_{ij}=x_{ij}-Jg^{-1}J^\top x_{ij}.
\]

Then the selected surface has Gaussian curvature

\[
K_\Sigma
=\frac{
\langle B_{aa},B_{bb}\rangle-\|B_{ab}\|^2
}{\det g}.
\]

This uses only second derivatives of \(x\), avoiding noisy numerical differentiation of Christoffel symbols.

Because the surface lies in \(S^{V-1}(2)\), also report \(K_\Sigma-1/4\) to separate the ambient Fisher-sphere contribution from bending inside the sphere.

Numerical QA identities are

\[
\|x\|=2,
\qquad
\langle x,x_i\rangle=0,
\qquad
\langle x,x_{ij}\rangle=-g_{ij}.
\]

Use fourth-order stencils and spacings \(h,h/2,h/4\). Freeze the primary
absolute tolerance at zero before inspecting results and accept only when

\[
\|T_{h/2}-T_{h/4}\|
\le a_{\rm ref}+r_{\rm ref}\max\{\|T_{h/2}\|,\|T_{h/4}\|\}.
\]

Store both errors and the tolerance ratio; a visually selected plateau is not
an acceptance rule. A nonzero absolute floor is a separately labeled
sensitivity analysis and requires a tensor-specific propagated roundoff bound.

For an independent holonomy check, integrate intrinsic LC transport around a small loop and compare its angle to

\[
\Omega=\int_DK_\Sigma\,dA.
\]

Repeat with alternative interpolation charts through the same corners. Strong chart dependence is evidence that the result belongs to the interpolation choice rather than a robust natural-language structure.

## 8. Experiment E: predictive connection distillation

The model-free packet core and its synthetic tests are implemented; real-model
packet generation, student training, and evaluation remain prospective. The
authoritative training and packet specification is
[PREDICTIVE_CONNECTION_DISTILLATION.md](PREDICTIVE_CONNECTION_DISTILLATION.md).

### 8.1 Primary setup

- first validate all estimators on a synthetic saturated-softmax
  teacher--student pair with analytic geometry and an oscillatory KL-only
  counterexample;
- frozen Pythia-70M teacher and Pythia-14M student;
- shared tokenizer and full outcome vocabulary;
- one-dimensional charts for the initial \(H^3\) audit and two-dimensional
  soft-token charts for packet transfer, all with frozen anchor tokens;
- frozen chart-sampling measure with a reported density and quadrature audit;
- offline float64 teacher packets, float32 deterministic serialization of the
  geometric sidecar, and float64 output anchors unless an error-bounded output
  compression is declared;
- adapter or final-block student training with the LM head frozen;
- four independent training seeds for the engineering pilot, paired by
  initialization and frozen example order across arms; the confirmatory seed
  count is chosen by the simulation-based power rule in Section 9;
- NLL-matched selection or a preregistered behavior--NLL Pareto frontier.

The primary training comparison fixes total student-training FLOPs or wall
time, including forward, backward, JVP, and ODE work. Each arm consumes the
longest prefix it can process from the frozen order, and processed-example
counts are reported. A secondary fixed-exposure run uses identical examples
and steps and reports its unequal compute. NLL matching is a model-selection
constraint, not a substitute for compute matching. Shared offline teacher
packet generation is accounted for separately.

At each accepted point, store

\[
\mathcal P_T=(q_T,G_T,L_T,C_T),
\]

where \(L_{ij,k}=\langle\partial_{ij}(2\sqrt q),
\partial_k(2\sqrt q)\rangle\). The packet reconstructs every Amari
connection through

\[
(\Gamma_T^{(\alpha)})^\ell{}_{ij}
=(G_T^{-1})^{\ell k}
\left((L_T)_{ij,k}-\frac\alpha2(C_T)_{ijk}\right).
\]

### 8.2 Matched arms

| Arm | Student objective beyond data NLL |
|---|---|
| D0 | none |
| D1 | output KD |
| Jz | D1 plus centered-logit Jacobian matching |
| Jpsi | D1 plus square-root-output Jacobian matching |
| D2 | output KD plus Fisher metric matching |
| D3 | D2 plus Levi--Civita connection matching |
| D4 | D3 plus raised cubic matching |
| D5 | D3 plus a context-shuffled teacher cubic target replacing \(C_T\); \(L\) is never scrambled |
| D6 | D4 plus integrated transport matching; secondary only |

For D5, keep the \(G_T\) and \(L_T\) targets and replace each context block's
cubic field with the teacher's field from one preregistered donor context at
matching chart coordinates, shuffled within strata of matched Fisher
conditioning and metric scale and held fixed per seed. Each donor \(C_T\) is a
real teacher tensor, but its combination with the recipient \((G_T,L_T)\) is
not guaranteed to be jointly realizable by one predictive map; D5 is therefore
a coherent donor-field control, not a realizability control. \(L\) is never
scrambled: it is a pointwise function of
the metric derivatives, so a scrambled-\(L\) target is jointly infeasible
with the retained \(G_T\) target at dense sampling. Per-packet
Fisher-orthogonal scrambles of \(C\) are discontinuous across packets and act
as shrinkage pressure rather than an orientation control; they are secondary
stress tests. Achieved geometric-loss floors must be reported per arm, and an
elevated D5 floor makes the D4--D5 contrast infeasible rather than positive
evidence. The
chart output-KD term is evaluated at the identical full stencil-point set in
every arm, making D1 the stencil-matched exposure control.
[PREDICTIVE_CONNECTION_DISTILLATION.md](PREDICTIVE_CONNECTION_DISTILLATION.md)
Section 7 is authoritative for arm definitions.
The preregistered mechanistic contrasts are D2--D1, D3--D2, D4--D3, and
D4--D5, with the additional comparisons Jpsi--Jz and D3--Jpsi. This list does
not make every contrast a separate confirmatory claim: Section 9 defines the
global candidate-arm hypotheses and multiplicity procedure.
The Jacobian arms separate generic derivative distillation, oriented
square-root tangent matching, and intrinsic Gram-metric matching. D6 is run only
after finite-difference packets and pointwise connection training pass their
held-out numerical checks.

### 8.3 Distillation outcomes and confirmatory success

Report held-out NLL, output KL, centered-logit and square-root Jacobian defects,
metric defect, measured \(H^{s_*}\) envelope, LC connection defect, raised cubic
defect, integrated \(e\)/LC/\(m\)/fitted-\(\alpha\) transport errors, semantic-field
commutator error, behavioral composition and intervention transfer, off-target
KL, rank, conditioning, packet size, and compute. On a verified noninjective
map or complete declared coarsening only, additionally report the
fixed-representation conditional-variance obstruction and cross-model
conditional-mean-field mismatch in the student metric.

A positive result requires D3 or D4 to reject its Section 9 global
intersection-union null after Holm correction. Thus the candidate must improve
held-out transport and the preregistered behavioral outcome beyond every
required control while satisfying validation-NLL and feasibility
noninferiority at matched compute, with
at least 90% packet acceptance overall and 80% in each frozen
context-by-operation stratum, validation NLL within 0.01 nats/token of each
required control, a minimum behavioral point improvement of at least 0.20
frozen D1-pilot between-seed standard deviations, the decision-compatible and
unadjusted one-sided confidence bounds defined in Section 9 for every
global-test component, held-out
Fisher-alignment singular values in \([0.5,2]\), and any claimed relative
transport certificate at most 0.10. In every frozen stratum, the student's
median declared-intervention Fisher length must be at least 90% of D1's. It
must survive rank, chart-scale, and
stencil sensitivity and
generalize to contexts and operations excluded from packet construction. Lower
packet loss alone is not evidence of useful compression.

### 8.4 Numerical gates

- construct packets in the declared float64 teacher inference dtype, with
  score-moment cubics from JVPs or logit central differences as primary and
  the probability-difference cubic as audit;
- bind every serialized packet to an immutable teacher revision, tokenizer and
  outcome-map hashes, chart identifier and bounds, context identifier, and
  inference dtype; require those bounds to contain every axial and mixed
  stencil point and fail before teacher evaluation otherwise; untracked
  accepted packets are invalid training artifacts;
- reject nonfinite, nonpositive, or unnormalized probability vectors rather
  than clipping or silently excluding outcomes; casting lower-precision model
  outputs to float64 after inference does not satisfy the dtype contract;
- refine central differences on both \((h,h/2,h/4)\) and the incommensurate
  ladder \((h/\sqrt2,h/(2\sqrt2),h/(4\sqrt2))\); require internal Richardson
  convergence on each ladder and agreement between their fine extrapolants.
  Compare only chart-invariant quantities. For the metric,
  use
  \(d_G=\|G_2^{-1/2}(G_1-G_2)G_2^{-1/2}\|_F\) with
  \(s_G=\sqrt m\). For each \(X\in\{L,C\}\), raise each extrapolant with its
  own metric, \(\Gamma_{X,k}=G_k^{-1}X_k\), and use the Fisher norm induced by
  the fine metric:
  \(d_X=\|\Gamma_{X,1}-\Gamma_{X,2}\|_{G_2}\) and
  \(s_X=\max(\|\Gamma_{X,1}\|_{G_2},\|\Gamma_{X,2}\|_{G_2})\).
  Require \(d_Y\le a_{\rm ref}+r_{\rm ref}s_Y\) for every
  \(Y\in\{G,L,C\}\), and store the maxima of
  \(d_Y/\max(s_Y,a_{\rm ref})\), \(d_Y\), and
  \(d_Y/(a_{\rm ref}+r_{\rm ref}s_Y)\). The primary analysis freezes the
  dimensionless values \(a_{\rm ref}=10^{-6}\) and
  \(r_{\rm ref}=10^{-3}\). The absolute term means only that an invariant
  coefficient below the declared numerical threshold is treated as
  indistinguishable from zero by this gate; it is not a proof that the exact
  coefficient vanishes. Never apply a dimensionful floor to raw \(G,L,C\)
  tensors or silently replace their norm scale by one;
- when exact logit JVPs are supplied, require a logits function and independently
  compare its finite-difference Jacobian after removing the rowwise softmax
  gauge, with primary absolute floor zero and any nonzero sensitivity floor
  derived and recorded as above;
- after float32 sidecar conversion, recompute the metric eigenvalues and the
  complete ordered gate from the actual serialized tensors and stored
  diagnostics; refuse serialization if quantization changes the verdict, and
  require every rejected packet's reason to equal the recomputed first failing
  gate;
- report algebraic rank, relative conditioning, and absolute chart-scale
  failures separately; require teacher and student Fisher spectral floors on
  the declared chart;
- audit the theorem-matched integer Sobolev order
  \(s_*=\min\{k\in\mathbb N:k>2+m/2\}\); label an \(H^2\)-style roughness
  penalty as heuristic rather than theorem sufficient;
- report chart integration/sampling error for the forward-KL quantity;
- treat rank loss and intervention shrinkage as failures, not successful loss
  minimization;
- regenerate and checksum a random 1% of teacher packets;
- refine integrated transport independently of packet construction;
- perform a same-model packet sanity check before teacher--student training.

## 9. Sampling and statistics

Model outputs are deterministic. Vocabulary entries and finite-difference grid points are not independent observations.

The only automatically independent units are independently initialized and
trained model runs. The following are design factors, not a list of mutually
independent units:

- training seed;
- prompt template;
- lexical instantiation;
- semantic operation;
- checkpoint as a repeated measurement.

Freeze the population or finite sampling frame for prompt templates, lexical
items, and operations before training. Record which factors are crossed and
which are nested. Use paired arm and checkpoint comparisons. For uncertainty,
resample trained seeds first and then resample prompt families and lexical
items within their realized nesting; if prompt and operation are crossed, use
a multiway cluster bootstrap or a crossed random-effects model. Preserve all
arm and checkpoint pairing. Report independent-model count separately from
prompt, packet, and token counts, and report effect sizes and intervals rather
than a p-value over vocabulary entries.

The three hard-coded factorials in the historical smoke driver are a finite
convenience set and support inference only to those prompts. The confirmatory
run is blocked until a versioned manifest defines the target prompt/operation
frame, inclusion and exclusion rules, selection probabilities or complete
enumeration, and train/validation/test split. Unless a probability sample from
a named superpopulation is supplied, all confirmatory intervals are explicitly
finite-manifest intervals and no broader prompt-population generalization is
claimed.

The confirmatory unit is a candidate connection arm. Its required control sets
are

\[
\mathcal B_{D3}=\{D1,J_z,J_\psi,D2\},
\qquad
\mathcal B_{D4}=\{D1,J_z,J_\psi,D2,D3\},
\]

with D5 added to \(\mathcal B_{D4}\) when cubic orientation is part of the
confirmatory claim.

The versioned analysis manifest must name one scalar behavioral score
\(S_{\rm beh}\), with higher values better, and freeze its transformation,
aggregation over the prompt/operation frame, fitted model or resampling
statistic, missing-data rule, and number of resamples before confirmatory
training. For the Section 6.3 composition endpoint,
\(S_{\rm beh}=-e_\nabla\). Freeze
\(s_{\rm beh}>0\) as the D1 between-seed standard deviation estimated in the
engineering pilot; do not re-estimate it from unblinded confirmatory outcomes.
For each candidate \(C\) and each \(B\in\mathcal B_C\), define

\[
\Delta^{\rm beh}_{C,B}
=\frac{\mathbb E(S_{{\rm beh},C}-S_{{\rm beh},B})}{s_{\rm beh}},
\quad
\Delta^{\rm tr}_{C,B}=\mathbb E(E_{{\rm tr},B}-E_{{\rm tr},C}),
\]

\[
\Delta^{\rm NLL}_{C,B}
=\mathbb E(\operatorname{NLL}_C-\operatorname{NLL}_B),
\qquad
\Delta^{\rm feas}_{C,B}=\pi_C-\pi_B.
\]

Obtain from the frozen seed-first analysis one-sided p-values for the nulls
\(\Delta^{\rm beh}_{C,B}\le0\),
\(\Delta^{\rm tr}_{C,B}\le0\),
\(\Delta^{\rm NLL}_{C,B}\ge0.01\), and
\(\Delta^{\rm feas}_{C,B}\le-0.02\). Define

\[
p_C=\max_{B\in\mathcal B_C}
\max\{p^{\rm beh}_{C,B},p^{\rm tr}_{C,B},
p^{\rm NLL}_{C,B},p^{\rm feas}_{C,B}\}.
\]

This is an intersection-union p-value: the candidate succeeds only when every
required component rejects, so no within-candidate multiplicity adjustment is
needed. Apply Holm at familywise 0.05 to \(p_{D3}\) and \(p_{D4}\): compare the
smaller with 0.025 and, only if it rejects, the larger with 0.05. Report every
component estimate and raw p-value, plus the standard Holm-adjusted global
p-value for each candidate. If a candidate is reached at Holm local level
\(\alpha_C\), invert the same analysis to report decision-compatible one-sided
bounds at level \(1-\alpha_C\): every behavioral and transport lower bound must
exceed zero, every NLL upper bound must be below 0.01, and every feasibility
lower bound must exceed \(-0.02\). Also report ordinary unadjusted 95%
one-sided bounds for estimation; they do not override the Holm decision. The
observed practical-effect gate is
\(\min_{B\in\mathcal B_C}\widehat\Delta^{\rm beh}_{C,B}\ge0.20\); it is not a
confidence claim that the population effect is at least 0.20.

Packet coverage, Fisher alignment, intervention length, certificate, and
numerical-sensitivity thresholds are deterministic conjunctive gates after the
Holm decision. Failed seeds and infeasible paths remain in the
intention-to-treat analysis; conditional error among jointly feasible paths is
reported afterward. The confirmatory analysis uses the stated NLL
noninferiority margin. Pareto-frontier analyses are exploratory unless a
dominance statistic and multiplicity family are separately frozen.

If cubic orientation is claimed, preregister D5 and include D4--D5 behavioral
and transport superiority plus NLL and feasibility noninferiority inside
\(p_{D4}\), subject to comparable geometric-loss floors. Without D5, D4 may
support only a generic connection-distillation claim. Operation-specific
claims add global hypotheses to the Holm family before data inspection; with
\(m\) frozen global hypotheses, use the complete \(m\)-step Holm procedure
rather than the two-candidate shortcut. Jpsi--Jz, D2--D1, regularity
variants, D6, and geometric mechanism outcomes are secondary unless a separate
confirmatory family is frozen.

Before a confirmatory run, simulate this frozen multilevel design and the full
intersection-union/Holm decision rule using the
engineering-pilot variance components and choose enough independent training
seeds for at least 80% power to detect a 0.20 frozen-D1-pilot-standard-deviation
behavioral improvement after the stated multiplicity correction. Four or five
seeds alone is a pilot scale, not a confirmatory sample-size justification.

Secondary mechanistic analyses (not members of the Section 9 confirmatory
family):

1. paired change in semantic curvature from step 0 to final;
2. semantic minus spectrum-matched-control curvature at the same base, with Fisher-Haar curvature reported separately;
3. paired connection-error differences on prompts feasible for all compared connections;
4. a sensitivity analysis that treats infeasibility as failure rather than dropping it;
5. regression of composition error on curvature with entropy, the softmax-entropy cumulant-profile observables of [Viswanathan and Park](https://arxiv.org/abs/2510.04285), Fisher lengths, conditioning, and random-plane curvature as covariates.

## 10. Falsification criteria

The proposed claim is unsupported if:

- results depend strongly on solver or rank threshold;
- checkpoint patterns do not replicate across seeds;
- semantic/random differences vanish after matching conditioning and Fisher scale;
- curvature adds no predictive value beyond entropy, existing cumulant probes, and edge lengths;
- alternative interpolation charts give incompatible surface conclusions;
- LC transport is usually infeasible or performs no better than exponential transport at the semantic scale;
- the effect appears only in unnatural continuous interiors;
- claimed semantic operations are not approximately commuting or metric preserving when the theorem requires those assumptions.
- connection-distillation gains disappear after matching student NLL and compute;
- metric matching explains all gains attributed to connection or cubic matching;
- centered-logit or square-root Jacobian matching explains all behavioral gains
  attributed to connection transfer;
- a second-order roughness penalty is treated as the \(H^{s_*}\) theorem
  hypothesis without the required higher-order audit;
- packet losses fall through chart shrinkage or Fisher rank loss;
- distillation improvements fail on held-out chart constructions or semantic operations.

## 11. Completed Pythia-14M pilot

This pilot used one model seed, three hand-written factorials, four checkpoints, and eight Fisher-Haar planes per base. It predates the spectrum-matched control and validates execution only. With eight controls its minimum plus-one one-sided tail rank is \(1/9\); neither that rank nor its standardized differences are calibrated tests for the hand-selected semantic planes.

| Checkpoint | Factorial | Entropy | Semantic \(K\) | Random-plane mean \(K\) | \(\kappa(G)\) | Exponential FR error |
|---|---|---:|---:|---:|---:|---:|
| step0 | number × time | 10.6254 | 0.001125 | 0.000890 | 1.261 | 0.9473 |
| step0 | gender × number | 10.6265 | 0.001416 | 0.000937 | 1.271 | 0.4806 |
| step0 | animal × number | 10.6266 | 0.001012 | 0.000959 | 1.268 | 0.3844 |
| step1000 | number × time | 7.2467 | 0.051662 | -0.012872 | 168.705 | 0.5375 |
| step1000 | gender × number | 6.3610 | 0.112266 | -0.056202 | 251.709 | 0.6468 |
| step1000 | animal × number | 6.2919 | 0.073880 | -0.066966 | 264.088 | 0.6239 |
| step10000 | number × time | 6.2878 | 0.029023 | -0.119029 | 622.550 | 0.9760 |
| step10000 | gender × number | 6.3953 | -0.006836 | -0.065435 | 714.686 | 0.6108 |
| step10000 | animal × number | 6.6270 | 0.016065 | -0.080633 | 597.971 | 0.6139 |
| step143000 | number × time | 6.4136 | 0.021106 | -0.085388 | 3097.106 | 0.3637 |
| step143000 | gender × number | 6.3486 | 0.013702 | -0.077892 | 3874.121 | 0.5583 |
| step143000 | animal × number | 5.7406 | -0.044455 | -0.152915 | 4211.037 | 0.5226 |

Checks:

- maximum logit reconstruction error was below \(10^{-5}\);
- maximum relative linear-solve residual was \(1.46\times10^{-16}\);
- all 12 decoded metrics passed the full-rank numerical check;
- exponential output composition was feasible in all 24 orientations;
- unscaled mixture and ambient LC composition were feasible in 0 of 24 orientations each.
- LC transport's mean absolute log-length distortion stayed below \(1.3\times10^{-16}\).
  This is an implementation-identity check implied by metric compatibility,
  not an empirical measurement supporting H1--H6. At the final checkpoint the corresponding
  pilot values were \(0.0877\) for exponential transport and \(0.4389\) for mixture
  transport.

Interpretation: the exact curvature is computable and differs strongly across training, planes, and contexts. The Fisher-Haar comparison is confounded by the semantic plane's alignment with the Fisher spectrum, especially as metric conditioning grows. The pilot therefore does **not** establish a trend or semantic effect. It also shows that finite-domain feasibility must be part of the connection comparison.

Raw local result: `results/pythia_14m_pilot.json`.

That archived file uses the legacy unversioned result schema. New smoke outputs
use schema `pythia-smoke-2` and record the repository commit and dirty flag,
Python/platform/package versions, requested and resolved model revisions,
model-run identifier, declared model-training seed when known, base and
per-control seeds, tokenizer/outcome hashes, and a design hash. A missing
training seed is represented as unknown, never inferred from the number of
checkpoint records. Checkpoints from one pretrained run are repeated measures,
not independent seeds. The driver refuses to create a scientific artifact from
an unidentified or dirty checkout by default. `--allow-dirty-repository` is an
explicit smoke-only override; the payload records the override and driver
source hash and is not a reproducible result.

## 12. Commands

Run model-free tests:

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run synthetic connection recovery:

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe experiments\synthetic_connection_recovery.py
```

Run the real-model CPU pilot after installing CPU PyTorch and `requirements-model.txt`:

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe experiments\pythia_cpu_smoke.py `
  --revisions step0 step1000 step10000 step143000 `
  --control-mode fisher-haar `
  --random-planes 8 `
  --output results\pythia_14m_pilot.json
```

The command above reruns the historical execution-only pilot design. It does not reproduce the stored control draws: the committed JSONs predate the order-independent hashed control seeding (`stable_control_seed`, introduced in commit `93fa3f0`), so the Fisher-Haar planes are redrawn deterministically under the current scheme and per-plane values differ from the archived outputs. Design, prompts, checkpoints, and all deterministic model quantities are unchanged. The following command is a corrected spectrum-matched **finite-chord diagnostic**, not the primary inferential experiment: it still uses three hard-coded prompt factorials, conventional exponential-coordinate chords, and one model seed. The continuous-tangent, preregistered-prompt, multi-seed study specified above requires a separate driver and data files.

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe experiments\pythia_cpu_smoke.py `
  --revisions step0 step1000 step10000 step143000 `
  --control-mode spectrum-block `
  --spectrum-band-mode relative-gap `
  --random-planes 999 `
  --metric-eigenvalue-rtol 1e-12 `
  --plane-gram-rtol 1e-4 `
  --solve-residual-rtol 1e-10 `
  --solve-forward-error-rtol 1e-4 `
  --same-plane-curvature-rtol 1e-10 `
  --eigenspace-rtol 1e-6 `
  --vocabulary-chunk-size 4096 `
  --chunking-curvature-rtol 1e-7 `
  --output results\pythia_14m_spectrum_matched.json
```

Repeat the control sensitivity run with
`--spectrum-band-mode relative-log --log10-band-width 1.0` and preregister any
additional widths before inspecting the semantic curvature ranks.

No GPU is used by this protocol.
