# Predictive Connection Distillation

> **Status:** formal research and training protocol. The model-free packet
> core (Section 13, items 1 and 5) is implemented and tested; packet
> generation against real models, student training, and evaluation are not
> implemented, and no empirical result exists.
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

at that position. Thus the two models receive the same declared barycentric
intervention protocol even when their embedding dimensions differ. This is an
operational identification, not a semantic one: the transformer is nonlinear
and the two embedding tables need not assign identical meaning to their
barycenters, so the chart aligns interventions without asserting that they
are the same semantic intervention. Any behavioral conclusion must survive
preregistered variation of anchor sets, intervention positions, chart radii,
and interpolation constructions. Anchor sets,
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

One dependence inside the packet must be stated explicitly. Because the inner
product \(\langle\partial_{ij}\psi,\partial_k\psi\rangle\) extracts the
tangential part of the second derivative,

\[
(L_M)_{ij,k}
=\frac12\bigl(
\partial_i(G_M)_{jk}+\partial_j(G_M)_{ik}-\partial_k(G_M)_{ij}
\bigr)
\]

exactly: the first-kind coefficient is a pointwise function of the first
derivatives of the metric field. At isolated packet points \(L_M\) carries
derivative information that sparse samples of \(G_M\) do not determine, but in
the dense-sampling limit the pair \((G,L)\) is redundant, since matching
\(G\) on a fine grid pins \(L\). The cubic tensor \(C\) is the only packet
component carrying geometric information not determined by the metric field.
The interpretation of the matched arms in Section 7 depends on this fact.

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
float32. That figure covers the geometric sidecar \((G,L,C)\) only: the
packet as defined also contains the output anchor \(q_T\), and full-vocabulary
anchors are not compact. At Pythia's 50,304-token vocabulary, 100,000 central
float32 outputs occupy about 18.7 GiB, the 8,000-point pilot about 1.5 GiB,
and retaining all nine stencil outputs about 13.5 GiB. The honest claim is
therefore that \((G,L,C)\) is a compact incremental geometric sidecar to
ordinary output distillation, not a compact total teacher dataset. The
storage plan for \(q_T\) must be declared: full-vocabulary anchors, possibly
float16, for the chart-KD points, or top-\(k\) anchors with a declared,
error-bounded tail treatment. Top-\(k\) storage does not by itself
instantiate the full-vocabulary forward-KL estimate used by the
risk--regularity route; that requires either a bounded tail approximation
error or a fixed outcome coarsening declared in advance. Full teacher
distributions need not be retained for points used only by the geometric
losses once the packet and declared distillation targets have been created.

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

Packet rejection conditions the experiment on teacher regions where the
method is numerically well behaved, which is a selection effect. The
experimental contract must therefore preregister a minimum packet acceptance
coverage, report acceptance rates by context family and semantic operation,
compare the behavioral difficulty of rejected and accepted contexts, and
declare a failure rule that stops the run when coverage falls below the
preregistered floor.

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

In every arm, the chart expectation runs over the identical frozen
stencil-point set used for packet generation, not only over central points.
This equalizes chart exposure across arms: no arm receives student gradients
at chart locations another arm never sees, so the output-only arm D1 is
itself the stencil-matched exposure control, and derivative matching cannot
be confounded with denser neighborhood augmentation.

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
band-limited chart basis controls the fitted surrogate, not automatically the
transformer's true predictive map: fitting sampled outputs certifies
\(\|\psi_{\mathrm{fit}}\|_{H^{s_*}}\) only, and a theorem-grade certificate
additionally requires a certified approximation residual
\(\|\psi_S-\psi_{\mathrm{fit}}\|_{H^{s_*}}\) through order-\(s_*\)
derivatives, which this protocol does not currently supply. The band-limited
quantity is therefore an **empirical spectral audit**, not a certificate,
unless the response is architecturally restricted to the band-limited family
or such a residual bound is proved. Its bandwidth must be frozen and
sensitivity tested. For \(s_*\ge4\), that is for two-dimensional charts, this
empirical spectral audit is the primary instrument; direct fourth-order
finite differences of full-vocabulary predictive maps are numerically
unreliable at that order and serve as a secondary check only.

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

The cubic tensor uses the score-moment identity of
[paper/main.tex](paper/main.tex) as its primary estimator. With logit
directional derivatives estimated by exact JVPs or central differences of
logits, the scores are

\[
S_a(u)=D_u\ell_a-\sum_b q_b\,D_u\ell_b,
\]

which is invariant under the softmax gauge, and

\[
C(u,v,w)
=\sum_a q_a\,S_a(u)S_a(v)S_a(w).
\]

Logits are order one and this form carries no inverse-probability factors, so
it avoids the catastrophic cancellation of probability differences near rare
tokens. The direct probability-difference form

\[
C_h(u,v,w)
=\sum_a
\frac{D_hq_a[u]D_hq_a[v]D_hq_a[w]}{q_a(z)^2}
\]

subtracts tiny numbers and divides by \(q_a^2\); it is retained only as an
independent audit on accepted packets, never as the primary estimator.

Square-root calculations use float64 and stable probabilities. The LC packet
does not require a tokenwise probability floor, but the nonzero-\(\alpha\)
cubic calculation must pass the raised-moment and numerical gates described in
the paper.

For smooth maps, the central first derivative is \(O(h^2)\); both second
derivative stencils are \(O(h^2)\) under sufficient fourth-order regularity.
Finite-difference accuracy is bounded by the teacher inference precision:
evaluating a float32 forward pass at \(z\pm h\) and casting the outputs to
float64 does not produce float64 derivatives, and the refinement sequence can
then measure the quantization floor rather than convergence. The packet
builder therefore declares the teacher inference dtype, runs teacher
evaluation in float64 on CPU, which is affordable at the pilot scales, and
prefers exact JVP derivatives wherever autodiff is available, with finite
differences as the audit path. Use \((h,h/2,h/4)\) with a formal
Richardson-style acceptance rule: accept a packet only when the extrapolated
derivative changes by less than a preregistered relative tolerance between
consecutive refinements, not on a visually judged plateau. Directional
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

1. Evaluate teacher outputs on the central-difference stencil in the
   declared teacher inference dtype, float64 on CPU at pilot scales.
2. Construct \((q_T,G_T,L_T,C_T)\) in float64, with score-moment cubics from
   JVPs or logit differences as the primary estimator.
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
the same examples in the same order. All arms share the same chart stencil
exposure through the output anchor (Section 4.1); residual per-point compute
differences from JVP evaluation in the geometric arms are governed by the
NLL-matched or Pareto budget rule and must be reported.

| Arm | Objective beyond data NLL | Question |
|---|---|---|
| D0 | none | student-only baseline |
| D1 | output KD | standard teacher compression baseline |
| Jz | output KD + centered-logit Jacobian | does generic derivative matching explain the gain? |
| Jpsi | output KD + square-root Jacobian | does predictive-tangent orientation add value, and does square-root weighting beat logits? |
| D2 | output KD + \(G\) | does local Fisher sensitivity add value? |
| D3 | output KD + \((G,L)\) | does densified metric-derivative information add value beyond sparse \(G\)? |
| D4 | output KD + \((G,L,C)\) | does the metric-independent cubic tensor add value beyond LC? |
| D5 | output KD + \((G,L)\) + context-shuffled teacher cubic target | does the correctly oriented cubic tensor matter beyond its scale and realizability? |
| D6 | D4 + integrated path loss | does explicit transport improve over pointwise packet matching? |

Use **Jpsi** as the machine-readable arm name while writing it as
\(J_\psi\) in equations. Cross D1 and D2 with either the full
\(H^{s_*}\) regularizer and audit or the explicitly heuristic
\(\mathcal R_2\) condition. These regularity variants are labeled **D1-H**, **D2-H**,
**D1-R2**, and **D2-R2**; they are not silently pooled with their parent arms.

For D5, keep the \(G_T\) and \(L_T\) targets intact and replace the cubic
target at each packet with the teacher's own cubic tensor drawn from a
different, preregistered donor context: shuffle \(C_T\) between context
blocks within preregistered strata of matched Fisher conditioning and metric
scale, holding the assignment fixed per seed. The donor tensors are smooth
along each donor chart, realizable as actual teacher jets, and norm-matched
in distribution by the stratification, but their orientation is wrong for the
recipient context. This context-shuffled field is the primary negative
control.

A per-packet Fisher-orthogonal scramble --- drawing \(R\) with
\(R^\top G_TR=G_T\) independently at each packet point and transforming every
covariant index of \(C_T\) --- preserves pointwise symmetries and
teacher-Fisher norms but produces a discontinuous, generally unrealizable
cubic target field. Optimizing against incoherent neighboring targets pushes
the student toward shrinking its cubic tensor toward zero, which is a
different inductive bias rather than an orientation control. The scramble and
entrywise random tensors rescaled to matched norms are therefore secondary
stress tests only.

\(L_T\) must never be scrambled in any control. Since \(L\) is a pointwise
function of the metric derivatives (Section 3), a scrambled-\(L\) target is
jointly inconsistent with the retained \(G_T\) target at dense packet
sampling: no smooth predictive map satisfies both, so that arm acquires an
elevated loss floor for feasibility reasons unrelated to semantic
orientation. Every arm must report the achieved training values of its active
geometric terms; a control-arm floor visibly above D4's invalidates the
corresponding contrast for that run.

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
an established derivative-matching baseline. Because \(L\) is a pointwise
function of the metric derivatives (Section 3), \(D3-D2\) tests densification
of metric-derivative information at sparse packet points, not transfer of a
geometric object beyond the metric field; a \(D3-D2\) gain must not be
reported as connection transfer beyond the metric. The only contrast that
adds geometric information not determined by the metric field is
\(D4-D3\), through the Amari--Chentsov tensor, and \(D4-D5\) tests whether
its teacher orientation matters beyond loss scale. \(D3-J_\psi\) tests the
intrinsic packet route against the stronger oriented predictive Jacobian.
The **-H** and **-R2** contrasts test anti-oscillation regularization;
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
- four training seeds, a preliminary pilot scale (Section 10);
- train only adapters or the final transformer block; freeze the LM head;
- sequence length at most 64;
- nine teacher stencil evaluations per context for a complete two-dimensional
  second-order central-difference jet; the two-dimensional \(H^4\) audit uses
  a frozen band-limited chart basis with preregistered, sensitivity-tested
  bandwidth as its primary empirical spectral audit (Section 4.3), with a
  wider finite-difference stencil as a secondary check only;
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
4. the preregistered primary behavioral outcome improves with paired seed
   effects and hierarchical bootstrap intervals (Section 9.5) clearing a
   preregistered practical-effect threshold; with four seeds this is a
   preliminary pilot criterion, and simple three-of-four counting is a pilot
   stopping rule, not a confirmatory standard;
5. the gain survives finite-difference, rank, and ODE-step sensitivity;
6. it beats the context-shuffled cubic control D5 when that arm is run, with
   comparable achieved geometric-loss floors across the compared arms;
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

## 11. Mathematical guarantees and remaining targets

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

### 11.3 Packet-to-transport bound

This bound is now proved at the same level of rigor as the packet estimate.
The missing sampling ingredient is Hölder interpolation between packet
points; the transport step is the Duhamel--Gronwall argument already used by
the paper's transport theorems.

Fix compatible submultiplicative norms as in Section 11.2, a declared Hölder
exponent \(\rho\in(0,1]\), and packet points \(\{z_i\}\subset U\) with fill
distance \(h_{\mathrm{fill}}=\sup_{z\in U}\min_i\|z-z_i\|\).

**Proposition (packet-to-transport).** Assume on \(U\):

1. \(G_T,G_S\succeq\lambda I\);
2. the connection fields satisfy
   \(\|\Gamma_M^{(\alpha)}\|_{C^0(U)}\le M_M\) and
   \([\Gamma_M^{(\alpha)}]_{C^\rho(U)}\le H_M\) for \(M\in\{T,S\}\);
3. at every packet point the true-tensor defects obey the Section 11.2
   hypotheses, so that

\[
\delta_{\mathrm{pack}}
=\lambda^{-1}\left(\varepsilon_L+\frac{|\alpha|}2\varepsilon_C\right)
+\lambda^{-2}\varepsilon_g\left(M_L+\frac{|\alpha|}2M_C\right)
\]

bounds \(\|\Gamma_S^{(\alpha)}(z_i)-\Gamma_T^{(\alpha)}(z_i)\|\).

Then every \(C^1\) path \(\gamma\subset U\) of background length
\(L_\gamma\) satisfies

\[
\|P_{\gamma,S}^{(\alpha)}-P_{\gamma,T}^{(\alpha)}\|
\le
L_\gamma\,
e^{(M_T+M_S)L_\gamma}
\Bigl(
\delta_{\mathrm{pack}}
+(H_T+H_S)\,h_{\mathrm{fill}}^{\rho}
\Bigr).
\]

*Proof.* Every \(z\in\gamma\) lies within \(h_{\mathrm{fill}}\) of a packet
point \(z_i\), so with \(\Delta\Gamma=\Gamma_S^{(\alpha)}-\Gamma_T^{(\alpha)}\),

\[
\|\Delta\Gamma(z)\|
\le\|\Delta\Gamma(z_i)\|
+(H_T+H_S)\,h_{\mathrm{fill}}^\rho
\le\delta_{\mathrm{pack}}+(H_T+H_S)\,h_{\mathrm{fill}}^\rho .
\]

Transport solves the linear ODE
\(\dot V=-\Gamma_M(\gamma(t))[\dot\gamma(t)]V\) with fundamental solutions
\(\Phi_M\). The Duhamel identity

\[
\Phi_T(1,0)-\Phi_S(1,0)
=\int_0^1\Phi_S(1,s)\,
\bigl(A_S(s)-A_T(s)\bigr)\,\Phi_T(s,0)\,ds,
\qquad
A_M(t)=-\Gamma_M(\gamma(t))[\dot\gamma(t)],
\]

together with the Gronwall bounds \(\|\Phi_M(t,s)\|\le e^{M_ML_\gamma}\) and
\(\int_0^1\|A_S-A_T\|\,ds\le L_\gamma\sup_\gamma\|\Delta\Gamma\|\) gives the
display. \(\blacksquare\)

Three remarks make the bound operational.

1. **Measurement inflation.** Measured packets differ from true tensors by
   finite-difference and quantization error. If per-model, per-tensor
   measurement errors are bounded by \(\eta_g,\eta_L,\eta_C\), replace each
   \(\varepsilon_\bullet\) by \(\varepsilon_\bullet+2\eta_\bullet\). This is
   where the \(\varepsilon_{FD}\) and \(\varepsilon_Q\) of the earlier
   target statement enter, and \(\mathcal A=e^{(M_T+M_S)L_\gamma}\) is the
   declared transport-growth factor.
2. **Levi--Civita sharpening.** For \(\alpha=0\) the paper's
   metric-compatible transport bound replaces the exponential growth factor
   by its sharper metric-compatible counterpart.
3. **Audited constants.** \(M_M\) and \(H_M\) are reported as audited
   quantities: sup norms and finite-difference Hölder quotients of the
   connection fields over the packet grid, with declared safety margins. The
   audited \(H^{s_*}\) envelope of Section 4.3 with the spectral floor
   supplies \(C^\rho\) control in principle, but the bound is evaluated from
   the directly measured \(M_M,H_M\), not from embedding constants.

Without the sampling term, small packet loss says nothing about a path
running between packet points; with it, packet loss, fill distance, and the
audited moduli convert into an explicit transport guarantee. The bound and a
numerical verification against exactly integrated transport are implemented
in `src/predictive_geometry/distillation.py` and
`tests/test_distillation_packets.py`.

### 11.4 Remaining open target: certified surrogate residual

The genuinely open item is the Sobolev certificate of Section 4.3: bounding
\(\|\psi_S-\psi_{\mathrm{fit}}\|_{H^{s_*}}\) for a band-limited surrogate
fitted from finitely many chart samples. Finite samples cannot certify this
residual without an a priori class assumption on the true map. A certificate
therefore requires one of: an architecturally band-limited chart response, a
proved modulus-of-continuity bound for the transformer's chart response, or
acceptance of audit-only status. Until one of these is supplied, the
\(H^{s_*}\) quantity remains an empirical spectral audit and the
risk--regularity route remains conditional on it.

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

The information content of the packet must also be stated honestly.
Pointwise, \(G\) and \(C\) are functions of the output anchor and the
centered-logit Jacobian, so the first-order Jacobian arms' targets already
determine them; \(L\) adds second-jet information at sparse packet points,
and on a dense chart even that reduces to derivatives of the Jacobian
targets. PCD therefore transfers no information fundamentally unavailable to
derivative distillation. Its defensible advantage claim is that the invariant
tensor packet is a more compact, dimension-independent, and possibly
better-conditioned supervision signal --- an inductive-bias claim. If D4
wins, the result must be described as better-compressed or better-conditioned
supervision, not as access to information absent from Jacobian matching,
which is established prior art.

## 13. Implementation roadmap

No item below may be described as implemented until its tests and output
schema exist. Under this rule, items 1 and 5 are implemented; items 2--4 are
not.

1. `src/predictive_geometry/distillation.py` --- **implemented**:
   - shared-chart packet dataclass with checksummed float32 serialization
     and schema-version validation;
   - central-difference jet assembly for \(G\) and \(L\), and score-moment
     cubic assembly with the probability-difference form as audit;
   - Richardson-tolerance, finite-value, and rank acceptance gates with
     rejection reasons retained for coverage reporting;
   - centered-logit and square-root Jacobian control losses;
   - metric, first-kind LC, and raised-cubic losses in teacher-Fisher norms;
   - the Section 11.2 packet-to-connection and Section 11.3
     packet-to-transport bounds;
   - a one-dimensional finite-difference Sobolev grid audit (empirical, per
     Section 11.4), the sufficiency decomposition of Section 9.4, and the
     context-shuffled cubic control of Section 7.
   The multi-dimensional \(H^{s_*}\) audit and the band-limited chart basis
   are not yet implemented.
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
5. `tests/test_distillation_packets.py` --- **implemented**, covering:
   - analytic affine-softmax and boundary Bernoulli packet fixtures with
     exact closed forms, including the vanishing of \(\Gamma^{(1)}\);
   - score-moment versus probability-difference cubic estimators: affine
     exactness and float32-underflow robustness;
   - linear chart-change covariance of \(G\), \(C\), and
     \(\Gamma^{(\alpha)}\);
   - consistency of the packet \(L\) with finite differences of the packet
     \(G\) field;
   - zero defects under self-distillation;
   - numerical domination of the Section 11.2 and 11.3 bounds over directly
     computed connection and transport defects;
   - the one-dimensional Sobolev grid audit against a closed form;
   - the KL-only oscillatory escape;
   - rank-gate and Richardson refinement-gate rejections;
   - checksummed serialization round-trip and float32 quantization error;
   - the strict distinction between Jacobian and Gram-metric matching;
   - exact additivity of the sufficiency decomposition on discrete fibers;
   - stratum-preserving context-shuffled cubic controls.
   An explicit schema-version rejection test and a dedicated logit-gauge
   test remain to be added.

## 14. Decision sequence

Step 1 and the synthetic form of step 3 are complete via
`src/predictive_geometry/distillation.py` and its tests; the remaining steps
are open.

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
