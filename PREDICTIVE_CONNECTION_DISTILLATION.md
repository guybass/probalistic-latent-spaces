# Predictive Connection Distillation

> **Status:** formal research and training protocol; not yet implemented and not
> an empirical result.
>
> **Purpose:** compress a large teacher's local predictive geometry and semantic
> transport law into a smaller student without requiring their hidden dimensions
> or hidden coordinates to align.

## 1. Research question

Ordinary knowledge distillation asks the student to reproduce teacher outputs.
The proposed experiment asks a stronger question:

> At matched student capacity, language-model loss, and training compute, does
> transferring a compact local Fisher--Amari connection packet preserve the
> teacher's held-out semantic transport and behavior better than output-only,
> generic derivative, square-root Jacobian, and metric-only distillation; and
> when does chart KL plus audited square-root Sobolev regularity already suffice
> for Levi--Civita transport agreement?

The intended claim is conditional and empirical. The project does not assume
that a student can reproduce every teacher capability, that one connection is
universally semantic, or that connection matching alone determines the global
predictive function.

We call the method **Predictive Connection Distillation (PCD)**. The name refers
specifically to the predictive categorical Fisher pullback and its Amari
connections; it should not be shortened to the already-used generic phrase
"geometric knowledge distillation."

## 2. Shared chart instead of hidden-state alignment

Let \(T\) be a frozen teacher and \(S\) a trainable student. They may have
different hidden dimensions. For context \(c\) and a shared low-dimensional
intervention coordinate \(z\in U\subset\mathbb R^m\), define model-specific
intervention realizations

\[
I_T(c,z),\qquad I_S(c,z),
\]

and aligned predictive maps

\[
F_T(c,z)=Q_Tp_T(\cdot\mid I_T(c,z)),
\qquad
F_S(c,z)=Q_Sp_S(\cdot\mid I_S(c,z)).
\]

Here \(Q_T,Q_S\) are fixed Markov maps into a shared outcome simplex. For the
primary Pythia experiment, the tokenizer and vocabulary are shared and
\(Q_T=Q_S=I\). A cross-tokenizer coarsening is a later experiment and must be
fixed independently of the evaluation prompts.

The same coordinate \(z\) must encode the same declared intervention in both
models. The primary differentiable construction uses mixtures of the same fixed
anchor tokens: the mixture weights are shared, while each model realizes them
through its own embedding table. This aligns the intervention chart without
asserting that teacher and student hidden vectors are comparable.

Concretely, fix anchor tokens \(a_0,\ldots,a_m\) and an intervention position.
For \(z\in\mathbb R^m\), set

\[
\pi_0(z)=\frac{1}{1+\sum_{r=1}^m e^{z_r}},
\qquad
\pi_r(z)=\frac{e^{z_r}}{1+\sum_{s=1}^m e^{z_s}},
\]

and, for model \(M\) with embedding table \(E_M\), inject

\[
x_M(z)=\sum_{r=0}^m\pi_r(z)E_M(a_r)
\]

at that position. Thus the two models receive the same token-mixture
intervention even when their embedding dimensions differ. Anchor sets,
positions, and the compact domain \(U\) are frozen before packet generation.
The central chart point need not be \(z=0\), but it must be recorded.

All connection differences below are evaluated after this common-chart
identification. A difference of Christoffel coefficients on unrelated hidden
coordinates would not be meaningful.

Equivalently, the teacher and student parameter manifolds used in this
experiment are two copies of \(U\), identified by
\(\Phi=\operatorname{id}_U\). This does not assert that \(F_T=F_S\), that their
full hidden spaces align, or that their predictive images are already
diffeomorphic. The shared tokenizer identifies the outcome coordinates; the
frozen chart construction identifies the intervention coordinates.

## 3. Predictive connection packet

Write

\[
q_M(z)=F_M(c,z),
\qquad
\psi_M(z)=2\sqrt{q_M(z)},
\qquad M\in\{T,S\}.
\]

At a chart point, define

\[
(G_M)_{ij}
=\langle\partial_i\psi_M,\partial_j\psi_M\rangle,
\]

\[
(L_M)_{ij,k}
=\langle\partial_{ij}\psi_M,\partial_k\psi_M\rangle,
\]

and

\[
(C_M)_{ijk}
=\sum_a
  \frac{\partial_iq_{M,a}\,\partial_jq_{M,a}\,
        \partial_kq_{M,a}}{q_{M,a}^2}.
\]

\(G_M\) is the pullback Fisher metric, \(L_M\) is the first-kind
Levi--Civita coefficient, and \(C_M\) is the pulled-back Amari--Chentsov
tensor. Whenever \(G_M\) is positive definite on the declared chart,

\[
(\Gamma_M^{(\alpha)})^\ell{}_{ij}
=(G_M^{-1})^{\ell k}
\left((L_M)_{ij,k}-\frac{\alpha}{2}(C_M)_{ijk}\right).
\]

Thus the teacher packet

\[
\mathcal P_T(c,z)=
\bigl(q_T,G_T,L_T,C_T\bigr)
\]

contains the local output anchor and the entire Amari connection family. The
geometric part is independent of the teacher hidden dimension.

### 3.1 Packet size

Using tensor symmetries, an \(m\)-dimensional chart needs

\[
\frac{m(m+1)}2
+m\frac{m(m+1)}2
+\binom{m+2}{3}
\]

geometric scalars for \((G,L,C)\). For \(m=4\), this is

\[
10+40+20=70
\]

floats per chart point. At 100,000 points this is approximately 28 MB in
float32, excluding optional top-\(k\) logits. Full teacher distributions need
not be retained once the packet and ordinary distillation targets have been
created.

### 3.2 Required packet metadata

Every packet shard must store:

- schema version;
- teacher model identifier and immutable revision;
- tokenizer and outcome-map hashes;
- intervention-chart identifier, anchor-token list, intervention position, and
  compact-domain bounds;
- context identifier, \(z\), and finite-difference step size;
- \((G,L,C)\), metric eigenvalues, and numerical acceptance flags;
- derivative-refinement errors from at least \((h,h/2,h/4)\);
- optional top-\(k\) teacher probabilities plus a declared tail treatment;
- dtype and quantization metadata.

Packets that fail the rank, derivative-convergence, or finite-value gates are
retained as failures in the audit log but excluded from connection training.

### 3.3 Two complementary routes to transport agreement

The project tests two routes that should not be conflated.

The **risk--regularity route** starts from the chart-integrated forward
distillation divergence

\[
\mathcal K_{T,S}
=\int_U
D_{KL}\bigl(q_T(z)\Vert q_S(z)\bigr)\,d\mu(z).
\]

The boundary-robust theorem in [paper/main.tex](paper/main.tex) states that if
\(\psi_T,\psi_S\) have a common uniform \(H^s(U;\ell_2)\) envelope with
\(s>2+m/2\), the sampling measure has a density bounded below, both Fisher
metrics have a common spectral floor, and path lengths are bounded, then

\[
\|P_{\gamma,T}^{LC}-P_{\gamma,S}^{LC}\|_{\mathrm{op}}
\le C\,\mathcal K_{T,S}^{(s-r)/(2s)},
\qquad 2+\frac m2<r<s.
\]

The constant has no explicit vocabulary-size or minimum-token-probability
factor, but still depends on the measured Sobolev envelope, Fisher spectral
floor, chart, sampling density, and path-length bound. Empirical KL and
finite-grid derivative penalties do not by themselves certify these
hypotheses. Nonzero-\(\alpha\) transport additionally requires control of the
raised cubic/third-score moment.

The **packet route** directly matches \(G,L,C\) at sampled chart points. It
provides finite-dimensional local connection control and covers the full
Amari family, but sparse packet agreement alone does not exclude oscillations
between samples. The combined experiment therefore treats Sobolev regularity
as an audited anti-oscillation condition and packet matching as the direct
connection-transfer intervention.

## 4. Distillation objectives

The complete student objective is

\[
\mathcal L
=\mathcal L_{\mathrm{data}}
+\lambda_{KD}\mathcal L_{KD}
+\lambda_{Jz}\mathcal L_{Jz}
+\lambda_{J\psi}\mathcal L_{J\psi}
+\lambda_H\mathcal R_{H^{s_*}}
+\lambda_g\mathcal L_g
+\lambda_{LC}\mathcal L_{LC}
+\lambda_C\mathcal L_C
+\lambda_{P}\mathcal L_P.
\]

The teacher is frozen and every teacher target is stop-gradient. The displayed
formula is a menu: the matched arms below set most coefficients to zero rather
than activating every term simultaneously.

### 4.1 Output anchor

Use ordinary temperature-scaled output distillation on natural data and an
untempered forward-KL anchor on the frozen chart measure \(\mu\):

\[
\begin{aligned}
\mathcal L_{KD}
={}&
\tau^2\mathbb E_{c\sim\mathcal D}
D_{KL}\bigl(
\operatorname{sg}[p_T^{(\tau)}(\cdot\mid c)]
\Vert p_S^{(\tau)}(\cdot\mid c)
\bigr)\\
&+\lambda_{\mathrm{chart}}
\mathbb E_{(c,z)\sim\mu}
D_{KL}\bigl(
\operatorname{sg}[q_T(c,z)]
\Vert q_S(c,z)
\bigr).
\end{aligned}
\]

Retain ground-truth language-model loss. The untempered chart term, not a
temperature-softened substitute, estimates the \(\mathcal K_{T,S}\) used by
the risk-to-transport theorem. Geometry alone does not determine the global
predictive function, so output anchoring is mandatory.

This chart loss is an empirical estimate of \(\mathcal K_{T,S}\), not the
population integral automatically. The chart sampler, density, and quadrature
or Monte Carlo error must be reported. Reverse-KL arms may be studied as
language-model distillation baselines, but the risk-to-transport theorem is
stated for the forward order above.

### 4.2 Derivative-matching controls

Let \(\ell_M(z)\in\mathbb R^V\) be the model logits and remove the softmax
gauge by

\[
\bar\ell_M
=\ell_M-\frac{1}{V}\boldsymbol 1\boldsymbol 1^\top\ell_M.
\]

The conventional derivative-distillation control is

\[
\mathcal L_{Jz}
=\sum_{i=1}^m
\|\partial_i\bar\ell_S-\partial_i\bar\ell_T\|_2^2.
\]

This control is defined only in the shared-tokenizer primary experiment.
Cross-tokenizer studies require a separately justified common log-odds
coordinate system and may not silently compare raw logits.

The square-root-output Jacobian loss is

\[
\mathcal L_{J\psi}
=\sum_{i=1}^m
\|\partial_i\psi_S-\partial_i\psi_T\|_{\ell_2}^2.
\]

\(\mathcal L_{J\psi}\) is stronger than Fisher metric matching: it preserves
the orientation of predictive tangents in the shared square-root outcome
coordinates, whereas \(G=J_\psi^\top J_\psi\) retains only their intrinsic
Gram matrix. The two losses therefore require separate arms. Small empirical
Jacobian loss is an integrated derivative discrepancy, not automatically
uniform \(C^1\) agreement.

### 4.3 Sobolev regularity and roughness

Let

\[
s_*=\min\left\{k\in\mathbb N:k>2+\frac m2\right\};
\]

thus \(s_*=3\) for a one-dimensional chart and \(s_*=4\) for a
two-dimensional chart. On a regular compact domain, the theorem-matched
integer Sobolev audit is

\[
\mathcal R_{H^{s_*}}(\psi_S)
=\sum_{|\beta|\le s_*}
\int_U\|\partial^\beta\psi_S(z)\|_{\ell_2}^2\,dz.
\]

The frozen teacher must pass the same audit, and the reported envelope is

\[
B_{\mathrm{audit}}
=\|\psi_T\|_{H^{s_*}}+\|\psi_S\|_{H^{s_*}}.
\]

Freeze an admissible \(B_{\max}\) before confirmation runs. A regularized arm
may be interpreted through the theorem only if
\(B_{\mathrm{audit}}\le B_{\max}\) and the remaining density, rank, and
path-length gates pass.

A second-derivative roughness penalty

\[
\mathcal R_2
=\mathbb E_z\|D^2\psi_S(z)\|_{\mathrm{HS}}^2
\]

is cheaper and may be used as a heuristic crossed arm, but it is not the
\(H^{s_*}\) hypothesis when \(s_*>2\). Finite-grid estimates of either
quantity are numerical audits rather than proofs of a uniform bound. A
band-limited chart basis may convert finite coefficient control into a
certified bound, but its bandwidth must be frozen and sensitivity tested.

### 4.4 Fisher metric loss

Measure relative metric distortion in teacher-Fisher units:

\[
\mathcal L_g
=\left\|
G_T^{-1/2}(G_S-G_T)G_T^{-1/2}
\right\|_F^2.
\]

The primary metric gate requires

\[
\lambda_{\min}(G_T),\lambda_{\min}(G_S)
\ge\lambda_{\mathrm{train}}>0.
\]

Rank failures are outcomes; the primary analysis does not silently replace the
geometry by a ridge-deformed metric.

### 4.5 Levi--Civita connection loss

On the common chart, let

\[
D^{LC}=\Gamma_S^{LC}-\Gamma_T^{LC}.
\]

The difference is a \((1,2)\)-tensor. Its teacher-Fisher squared norm is

\[
\mathcal L_{LC}
=(G_T)_{\ell r}(G_T^{-1})^{ia}(G_T^{-1})^{jb}
(D^{LC})^\ell{}_{ij}(D^{LC})^r{}_{ab}.
\]

This loss is invariant under a simultaneous reparameterization of the shared
chart. Matching unaligned raw Christoffel arrays is forbidden.

### 4.6 Raised cubic loss

Define the raised cubic operator

\[
(K_M)^\ell{}_{ij}
=(G_M^{-1})^{\ell k}(C_M)_{ijk}.
\]

The nonzero-\(\alpha\) information absent from Levi--Civita metric
compatibility is transferred through

\[
\mathcal L_C
=\|K_S-K_T\|_{G_T}^2.
\]

Training with \((G,L,C)\) is preferred to choosing one \(\alpha\) in advance,
because it transfers the full canonical family and allows connection selection
to remain a held-out empirical question.

### 4.7 Integrated transport loss

For a preregistered path \(\gamma:[0,1]\to U\) and chart tangent probe \(u\),

\[
\mathcal L_P^{(\alpha)}
=\mathbb E_{\gamma,u}
\left\|
P_{\gamma,S}^{(\alpha)}u
-P_{\gamma,T}^{(\alpha)}u
\right\|_{G_T(\gamma(1))}^2.
\]

This is a secondary loss after pointwise packet training is stable. Its RK4 or
adaptive-ODE tolerance must be refined independently. Exponential and mixture
closed forms should be used when the chosen chart admits them exactly.

## 5. Forward-only finite-difference construction

The teacher packet can be generated offline without differentiating through
teacher parameters. For a unit chart direction \(u\), use central differences

\[
D_h\psi[u]
=\frac{\psi(z+hu)-\psi(z-hu)}{2h},
\]

and, for distinct directions \(u,v\), use the mixed stencil

\[
D_h^2\psi[u,v]
=\frac{
\psi(z+hu+hv)-\psi(z+hu-hv)
-\psi(z-hu+hv)+\psi(z-hu-hv)
}{4h^2}.
\]

For a pure second derivative, use the axial stencil

\[
D_h^2\psi[u,u]
=\frac{\psi(z+hu)-2\psi(z)+\psi(z-hu)}{h^2}.
\]

Then estimate

\[
G_h(u,v)=\langle D_h\psi[u],D_h\psi[v]\rangle,
\]

\[
L_h(u,v,w)
=\langle D_h^2\psi[u,v],D_h\psi[w]\rangle.
\]

The cubic tensor uses central differences of \(q\):

\[
C_h(u,v,w)
=\sum_a
\frac{D_hq_a[u]D_hq_a[v]D_hq_a[w]}{q_a(z)^2}.
\]

Square-root calculations use float64 and stable probabilities. The LC packet
does not require a tokenwise probability floor, but the nonzero-\(\alpha\)
cubic calculation must pass the raised-moment and numerical gates described in
the paper.

For smooth maps, the central first derivative is \(O(h^2)\); both second
derivative stencils are \(O(h^2)\) under sufficient fourth-order regularity.
Use \((h,h/2,h/4)\) and accept only a visible convergence plateau. Directional
subsampling may be used during training, but the held-out audit evaluates the
complete small-chart tensors.

## 6. Training curriculum

Connection losses should not be activated before the student has a usable
predictive map and a nondegenerate chart metric.

### Phase 0: freeze the experimental contract

Before packet generation, freeze:

- teacher revision, student architecture, and trainable parameter subset;
- shared tokenizer or fixed outcome coarsening;
- chart construction and anchor tokens;
- chart measure \(\mu\), density audit, Sobolev order \(s_*\), and
  confirmation threshold \(B_{\max}\);
- contexts, semantic operations, paths, and train/validation/test split;
- finite-difference step grid and numerical gates;
- optimizer, token budget, lambda grid, and random seeds;
- behavioral and geometric primary outcomes.

### Phase 1: build and audit teacher packets

1. Evaluate teacher outputs on the central-difference stencil.
2. Construct \((q_T,G_T,L_T,C_T)\) in float64.
3. Repeat at \((h,h/2,h/4)\).
4. Reject nonconvergent or rank-deficient packets according to the frozen gates.
5. Audit the teacher \(H^{s_*}\) norm or declare the lower-order roughness
   quantity to be heuristic only.
6. Serialize accepted packets in float32 and retain float64 audit summaries.
7. Regenerate a random 1% sample and require checksum and tolerance agreement.

### Phase 2: output-distillation warm start

Train the student with

\[
\mathcal L_{\mathrm{data}}+\lambda_{KD}\mathcal L_{KD}
\]

until validation NLL and teacher--student KL reach a preregistered plateau. Do
not use geometric losses to rescue a student that cannot learn the output task.

### Phase 3: derivative controls and regularity audit

Run the centered-logit Jacobian, square-root Jacobian, and metric arms
separately. This determines whether a later connection gain exceeds generic
derivative distillation and whether full square-root tangent orientation adds
information beyond the intrinsic metric. Cross the output-only and metric arms
with the preregistered regularity condition. Report \(B_{\mathrm{audit}}\);
do not infer a theorem hypothesis merely because a roughness penalty was
present in the objective.

### Phase 4: metric curriculum

Ramp \(\lambda_g\) linearly from zero. Track the student metric floor,
condition number, and held-out metric defect. If the student lowers the loss by
losing rank or shrinking intervention strength, stop the run and count it as a
failure.

### Phase 5: connection curriculum

After metric stability, activate \(\mathcal L_{LC}\). Then add
\(\mathcal L_C\) in a separate arm. The fixed-arm order prevents cubic
instability from being misattributed to LC geometry.

### Phase 6: optional path refinement

Only after pointwise connection defects decrease on held-out packets, add
\(\mathcal L_P\) on short preregistered paths. Compare direct integrated
transport with the pointwise packet surrogate and refine the ODE step size.

### Phase 7: frozen evaluation

Select checkpoints by validation NLL plus the declared Pareto rule, not by
test geometric loss. Evaluate test behavior, output KL, metric defect,
connection defect, and transport commutators without further tuning.

## 7. Matched experimental arms

Every arm starts from the same student initialization for a given seed and sees
the same examples in the same order.

| Arm | Objective beyond data NLL | Question |
|---|---|---|
| D0 | none | student-only baseline |
| D1 | output KD | standard teacher compression baseline |
| Jz | output KD + centered-logit Jacobian | does generic derivative matching explain the gain? |
| Jpsi | output KD + square-root Jacobian | does predictive-tangent orientation add value, and does square-root weighting beat logits? |
| D2 | output KD + \(G\) | does local Fisher sensitivity add value? |
| D3 | output KD + \((G,L)\) | does LC connection transfer add value beyond metric matching? |
| D4 | output KD + \((G,L,C)\) | does the full Amari family add value beyond LC? |
| D5 | output KD + \(G\) + Fisher-orthogonally scrambled \(L,C\) targets | does the correctly oriented teacher connection matter beyond loss scale? |
| D6 | D4 + integrated path loss | does explicit transport improve over pointwise packet matching? |

Use **Jpsi** as the machine-readable arm name while writing it as
\(J_\psi\) in equations. Cross D1 and D2 with either the full
\(H^{s_*}\) regularizer and audit or the explicitly heuristic
\(\mathcal R_2\) condition. These regularity variants are labeled **D1-H**, **D2-H**,
**D1-R2**, and **D2-R2**; they are not silently pooled with their parent arms.

For D5, sample a nonidentity linear map \(R\) from the orthogonal group of
\(G_T\), transform every covariant index of \(L_T\) and \(C_T\) by \(R\), and
use those transformed tensors in place of the teacher tensors. Because
\(R^\top G_TR=G_T\), this preserves the tensor symmetries and teacher-Fisher
norms while scrambling their alignment with the declared semantic chart. Draw
the map once per packet and seed and hold it fixed. Entrywise random tensors
rescaled to the same norms may be reported as a secondary stress test, but they
need not be realizable as a coherent local jet and are not the primary control.

The primary contrasts are

\[
J_\psi-J_z,\qquad
D2-D1,\qquad
D3-D2,\qquad
D3-J_\psi,\qquad
D4-D3,\qquad
D4-D5.
\]

The \(J_\psi-J_z\) contrast tests square-root predictive coordinates against
an established derivative-matching baseline. \(D3-D2\) and
\(D3-J_\psi\) jointly test whether connection information helps beyond both
an intrinsic first-order target and the stronger oriented predictive
Jacobian. The **-H** and **-R2** contrasts test anti-oscillation regularization;
only the former can be compared to the theorem assumptions after its measured
audit passes. D6 is secondary because it introduces additional solver and
path-design choices.

Hyperparameters are selected with either an NLL-matched constraint or a full
behavior--NLL Pareto frontier. Every arm receives the same number of pilot
trials and the same lambda grid size.

## 8. CPU-first pilot

The first implementation target is a synthetic saturated-softmax
teacher--student pair with a known predictive manifold. It must recover the
analytic packet, demonstrate the KL-only oscillatory escape, and verify that
the derivative, Sobolev, packet, and integrated-transport estimators converge
before any language-model result is interpreted.

The first language-model pilot is deliberately small:

- teacher: frozen Pythia-70M;
- student: Pythia-14M;
- shared Pythia tokenizer and full output vocabulary;
- one-dimensional charts for the initial \(H^3\) regularity audit and
  two-dimensional soft-token charts for the packet experiment;
- 5,000 discovery contexts and 1,000/2,000 validation/test contexts;
- four training seeds;
- train only adapters or the final transformer block; freeze the LM head;
- sequence length at most 64;
- nine teacher stencil evaluations per context for a complete two-dimensional
  second-order central-difference jet; the \(H^4\) audit uses a separately
  preregistered wider stencil or a frozen band-limited basis;
- teacher inference and packet generation performed once on CPU;
- stage A: D0, D1, Jz, Jpsi, and D2;
- stage B: D3 and D4 only after stage A passes its numerical gates;
- D5--D6 and the two-dimensional \(H^4\) arm only after stage B.

If Pythia-70M packet generation is too slow, first validate the pipeline with a
Pythia-14M teacher and a custom 2--4-layer student. That engineering run cannot
support a large-to-small scaling conclusion.

Freezing the student's LM head is the primary condition because it prevents the
student from changing the reference geometry merely by rotating or collapsing
the decoder. A learned-head robustness arm may follow only after the primary
result.

## 9. Evaluation

### 9.1 Predictive and behavioral outcomes

- held-out NLL and perplexity;
- teacher--student forward and reverse KL;
- task accuracy and target-token log odds;
- held-out semantic composition and intervention transfer;
- off-target KL, anti-steering rate, and general text degradation;
- latency, trainable parameters, packet size, and total training compute.

### 9.2 Geometric outcomes

- relative Fisher metric defect;
- centered-logit and square-root Jacobian defects;
- measured teacher/student \(H^{s_*}\) envelopes and second-order roughness,
  reported separately;
- LC connection-tensor defect;
- raised cubic defect;
- integrated \(e\), LC, \(m\), and fitted-\(\alpha\) transport defects;
- teacher--student semantic-field commutator error;
- rank, condition number, finite-difference convergence, and path feasibility.

### 9.3 Cross-model semantic commutator

For a semantic field \(X_T\) defined on the shared chart, the primary
cross-model diagnostic is

\[
\mathcal E_{T\to S}^{(\alpha)}
=\left\|
P_{\gamma,S}^{(\alpha)}X_T(z_0)
-P_{\gamma,T}^{(\alpha)}X_T(z_0)
\right\|_{G_T(z_1)}.
\]

When teacher and student fields are estimated separately, additionally report
their base-point field mismatch. Transport agreement must not receive credit
for starting from different semantic operations.

### 9.4 Fixed-representation sufficiency diagnostic

When the student representation has a verified noninjective predictive map
\(F_S\), let \(Z_S\) be the operation's predictive tangent effect and define

\[
\overline Z_S(p)
=\mathbb E[Z_S\mid F_S=p].
\]

For a teacher conditional-mean field \(Y_{T\to S}\) transferred into the same
student predictive tangent bundle, conditional expectation gives the exact
orthogonal decomposition

\[
\begin{aligned}
\mathbb E\|Z_S-Y_{T\to S}(F_S)\|_{g_S}^2
={}&
\underbrace{\mathbb E\operatorname{Var}_{g_S}(Z_S\mid F_S)}
_{\text{fixed-representation predictive insufficiency}}\\
&+
\underbrace{\mathbb E\|\overline Z_S-Y_{T\to S}\|_{g_S}^2}
_{\text{cross-model field mismatch}}.
\end{aligned}
\]

Only the second term is directly a teacher--student alignment error. The first
term says that no field depending only on this fixed predictive representation
can encode the operation exactly. It is not a universal statement about the
student architecture: training may change \(F_S\). At an injective final head
the first term vanishes identically and must not be reported as a capacity
diagnostic.

For a declared coarsening \(\bar F_S=QF_S\), recompute the effect
\(d\bar F_S X\), metric, conditional mean, and both terms on the coarsened
image. Conditioning the original tangent field on coarse labels is not this
theorem. Because most primary final-head charts are injective, this
decomposition is a secondary failure-localization study at a verified
noninjective earlier map or complete coarsening.

### 9.5 Statistical units

The independent units are training seed, prompt template, lexical family, and
semantic operation. Token probabilities and finite-difference stencil points
are not independent observations. Use paired seed comparisons and hierarchical
bootstrap intervals over seeds and prompt families.

## 10. Success and falsification

A positive connection-distillation result requires all of:

1. D3 or D4 improves held-out transport over D1, Jz, Jpsi, and D2;
2. it improves a preregistered behavioral transfer outcome;
3. validation NLL is matched or the arm lies on a better Pareto frontier;
4. the result replicates across at least three of four seeds;
5. the gain survives finite-difference, rank, and ODE-step sensitivity;
6. it beats the Fisher-orthogonally scrambled control D5 when that arm is run;
7. it transfers to contexts and semantic operations excluded from packet
   generation;
8. any risk-to-transport interpretation reports stable measured
   \(H^{s_*}\) envelopes, chart-sampling error, and Fisher spectral floors.

The hypothesis is unsupported when:

- packet losses fall but behavior does not improve;
- D2 explains all gains from D3/D4;
- Jz or Jpsi explains all behavioral gains from the connection arms;
- a roughness penalty is presented as satisfying the Sobolev theorem without a
  measured \(H^{s_*}\) audit;
- improvements disappear after matching NLL or compute;
- the student exploits metric rank loss, intervention shrinkage, or chart scale;
- the preferred connection is inconsistent across seeds or operations;
- connection alignment adds nothing beyond output KL and ordinary relational
  distillation;
- results exist only at packet points or on the packet-generation chart and
  fail between-sample or held-out chart constructions.

These negative outcomes remain useful: they distinguish fixed-representation
predictive insufficiency, generic derivative transfer, intrinsic metric
transfer, connection-specific information, and chart overfitting.

## 11. Mathematical guarantees and remaining target

### 11.1 Proven risk-to-LC bridge

The manuscript already proves the following implication. Let \(U\) be a
regular compact \(m\)-dimensional chart, let \(\mu\) have density bounded below,
and suppose

\[
\psi_T,\psi_S\in H^s(U;\ell_2^V),
\qquad
\|\psi_T\|_{H^s}+\|\psi_S\|_{H^s}\le B,
\qquad
s>2+\frac m2,
\]

with \(G_T,G_S\succeq\lambda I\). For bounded chart-path length and every
\(r\) satisfying \(2+m/2<r<s\),

\[
\|P_{\gamma,T}^{LC}-P_{\gamma,S}^{LC}\|_{\mathrm{op}}
\le
C\,
\mathcal K_{T,S}^{(s-r)/(2s)}.
\]

The constant is independent of vocabulary size and a minimum token
probability, but depends on \(B,\lambda^{-1}\), the chart, sampling-density
floor, and path-length envelope. The experiment can compare this rate with
observations only after auditing those quantities. The theorem concerns LC;
it does not replace the raised-cubic requirement for nonzero \(\alpha\).

### 11.2 Exact packet-to-connection bound

Fix compatible matrix and tensor norms for which index contraction is
submultiplicative. The following estimate is algebraic, not an unproved
empirical claim.

Assume on a common compact chart that teacher and student metrics satisfy

\[
G_T,G_S\succeq\lambda I,
\]

and packet errors obey

\[
\|G_S-G_T\|\le\varepsilon_g,
\quad
\|L_S-L_T\|\le\varepsilon_L,
\quad
\|C_S-C_T\|\le\varepsilon_C.
\]

If \(\|L_T\|\le M_L\) and \(\|C_T\|\le M_C\), then the inverse identity
\(G_S^{-1}-G_T^{-1}=G_S^{-1}(G_T-G_S)G_T^{-1}\) gives the exact algebraic
estimate

\[
\|\Gamma_S^{(\alpha)}-\Gamma_T^{(\alpha)}\|
\le
\lambda^{-1}
\left(\varepsilon_L+\frac{|\alpha|}{2}\varepsilon_C\right)
+\lambda^{-2}\varepsilon_g
\left(M_L+\frac{|\alpha|}{2}M_C\right)
.
\]

No smallness assumption on \(\varepsilon_g\) is needed beyond the common
metric floor.

### 11.3 Packet-to-transport target

The remaining theorem target is to combine the packet estimate with
finite-difference and packet-quantization errors and the exact
variation-of-connection identity. For paths of length at most \(L_0\), the
desired consequence is

\[
\|P_{\gamma,S}^{(\alpha)}-P_{\gamma,T}^{(\alpha)}\|
\le
L_0\,\mathcal A
\left(
\varepsilon_g+\varepsilon_L+|\alpha|\varepsilon_C
+\varepsilon_{FD}+\varepsilon_Q
\right),
\]

where \(\mathcal A\) is a declared transport-growth or mixed-norm factor,
\(\varepsilon_{FD}\) is finite-difference error, and \(\varepsilon_Q\) is packet
quantization error. For Levi--Civita, the metric-compatible and intrinsic
mixed-norm bounds in the paper should replace a generic exponential growth
factor.

The proof must state chart norms, regularity orders, packet envelopes, and
whether constants are pointwise or uniform. It must not infer derivative
agreement from output KL alone.

## 12. Relationship to prior work

The surrounding ideas are established:

- [Sobolev Training for Neural Networks](https://arxiv.org/abs/1706.04859)
  incorporates target derivatives into function approximation and explicitly
  includes network compression/distillation as an application.
- [Knowledge Transfer with Jacobian Matching](https://arxiv.org/abs/1803.00443)
  transfers output derivatives and relates Jacobian matching to noisy-input
  distillation.
- [FitNets](https://arxiv.org/abs/1412.6550) transfers intermediate teacher
  representations through learned hints.
- [Relational Knowledge Distillation](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html)
  transfers distances and angles between examples rather than matching only
  individual outputs.
- [MiniLM](https://arxiv.org/abs/2002.10957) and
  [MiniLMv2](https://arxiv.org/abs/2012.15828) transfer internal attention
  relations across differently sized Transformers.
- [MiniLLM](https://arxiv.org/abs/2306.08543) studies reverse-KL distillation
  for generative language models and is a required divergence-choice baseline.
- [Geometric Knowledge Distillation](https://openreview.net/forum?id=7WGNT3MHyBm)
  aligns neural heat kernels for graph-topology compression.

A focused literature search did not identify a method that jointly distills an
autoregressive model's predictive Fisher pullback, Levi--Civita coefficient,
Amari--Chentsov tensor, and connection-dependent semantic transport. This is a
research-position statement, not proof that no such paper exists. The work must
be described as a specific synthesis and tested against Sobolev/Jacobian,
relational, hidden-representation, and Transformer distillation baselines. The
conditional-variance decomposition is presented as a fixed-representation
failure-localization tool, not as a claim that prior distillation work lacks
all notions of student capacity.

## 13. Implementation roadmap

No item below should be described as implemented until its tests and output
schema exist.

1. `src/predictive_geometry/distillation.py`
   - shared-chart packet dataclass and schema validation;
   - central-difference jet assembly;
   - centered-logit and square-root Jacobian controls;
   - integer \(H^{s_*}\) audit and lower-order roughness diagnostic;
   - metric, first-kind LC, cubic, and raised-connection losses.
2. `experiments/build_teacher_packets.py`
   - batched CPU teacher inference;
   - refinement and rank gates;
   - sharded deterministic packet serialization.
3. `experiments/train_connection_student.py`
   - D0--D6, Jz, Jpsi, and crossed regularity arms;
   - curriculum scheduling and NLL/Pareto selection;
   - resumable CPU checkpoints.
4. `experiments/evaluate_connection_student.py`
   - frozen predictive, geometric, transport, and semantic evaluation.
5. `tests/test_distillation_packets.py`
   - analytic Bernoulli and saturated-softmax packet fixtures;
   - coordinate-change covariance;
   - finite-difference convergence;
   - packet serialization and quantization error;
   - logit-gauge invariance and the strict distinction between Jacobian and
     Gram-metric matching;
   - analytic Sobolev-order and oscillatory-counterexample fixtures;
   - conditional-variance decomposition on a noninjective map;
   - zero loss for identical teacher/student maps;
   - rejection of rank-deficient and mismatched-chart packets.

## 14. Decision sequence

1. Prove and test the finite-dimensional packet identities on synthetic maps.
2. Implement deterministic packet generation and schema validation.
3. Run a same-model self-distillation sanity check; all packet defects should be
   zero up to numerical error.
4. Verify the risk--regularity rate and KL-only oscillatory escape on a
   synthetic saturated-softmax teacher--student pair.
5. Run Pythia-14M to tiny-student engineering validation.
6. Freeze the primary Pythia-70M to Pythia-14M experiment.
7. Compare D0, D1, Jz, Jpsi, and D2 before activating connection losses.
8. Compare D3/D4 against both derivative controls, then add D5/D6.
9. Continue only if connection alignment improves held-out behavior beyond
   output, generic derivative, square-root Jacobian, and metric baselines.

This ordering keeps the project CPU-first and guarantees an interpretable
stopping point even if connection distillation does not improve compression.
