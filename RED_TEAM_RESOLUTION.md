# Red-Team Resolution

Date: 2026-08-06; external review rounds appended 2026-08-16

This file records the disposition of the theoretical and empirical audits. The authoritative formal source is [paper/main.tex](paper/main.tex); the authoritative empirical design is [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). Historical notes remain in the repository to show the development of the project, but they do not override those two files.

## Project conclusion

The defensible claim is:

> A smooth probabilistic decoder induces Fisher--Amari geometry on its predictively identifiable quotient. At an affine softmax head, ordinary hidden-vector reuse is exactly exponential parallel transport on the ambient intervention family. Cross-entropy convergence alone does not force convergence of this geometry. Whether Levi--Civita, mixture, or another alpha-connection improves semantic transfer is a held-out empirical question.

The project does not claim that every latent space is intrinsically a probabilistic manifold, that next-token pretraining selects one semantic connection, or that parallel transport necessarily beats vector addition.

## Theoretical audit

### Accepted: intrinsic versus ambient curvature

The historical commuting-Killing-flow result concerns the intrinsic Gaussian curvature of a declared semantic surface. The pilot computes ambient decoded-manifold sectional curvature. Gauss' equation separates them:

\[
K_\Sigma
=K_M(u,v)
+\langle II(u,u),II(v,v)\rangle
-\|II(u,v)\|^2.
\]

Ambient curvature alone therefore does not falsify the surface theorem unless the surface is totally geodesic or its second fundamental form is controlled. The experiment protocol keeps ambient decoded curvature and intrinsic intervention-surface curvature as separate observables.

### Accepted with repair: spanning semantic flows

If more vector fields are listed than the local dimension, they are not themselves a frame. The historical proof is repaired by selecting a locally independent commuting subfamily; independence persists on a neighborhood and the simultaneous-straightening argument then applies.

### Accepted: unrestricted connection fitting is locally vacuous

For one nonvanishing field, a locally chosen flat torsion-free connection can always make the field parallel. Connection comparison is informative only when the connection family is fixed independently, shared across operations, complexity controlled, and evaluated out of sample. The paper restricts the descriptive comparison to the Amari alpha-family.

### Accepted with corrected domain: common e/m-parallel fields

A field parallel for both dual connections lies in the common curvature-nullity space

\[
\mathcal N_R(p)=\bigcap_{X,Y}\ker R_p(X,Y).
\]

The field must vanish when this nullity is trivial. Constant curvature \(1/4\) gives that conclusion on the full categorical simplex of dimension at least two. Nonzero sectional curvature alone is not sufficient on every lower-dimensional decoded family.

### Accepted: finite hidden differences privilege the exponential chart

At an affine head, \(h(Tc)-h(c)\) is an exponential logarithm. It is valid when the estimand is transfer of the conventional activation-vector intervention, but it is not connection neutral. The primary field definition is now the derivative of a declared continuous intervention. Finite endpoint comparisons must use each candidate's own logarithm, transport, and exponential.

### Accepted: architectural support is smaller than the ambient head family

Post-LayerNorm natural states satisfy an affine mean constraint and are nearly fixed-radius when the normalization epsilon is small. Arbitrary post-normalization interventions occupy a larger ambient family. Addition may remain in the affine hyperplane but can leave the natural reachable region. The paper and protocol now distinguish ambient intervention geometry from support-intrinsic geometry.

### Accepted: the conditional-variance obstruction is inert at an injective head

For

\[
F(h)=\operatorname{softmax}(Wh+b),
\]

\[
F(h)=F(h')
\quad\Longleftrightarrow\quad
W(h-h')\in\operatorname{span}\{\mathbf 1\}.
\]

If the centered decoder is injective, equivalently if the pullback Fisher matrix is positive definite, then \(F(H)\) determines \(H\) and

\[
\operatorname{Var}_g(Z_T\mid F)=0.
\]

The theorem remains an exact obstruction for nontrivial predictive fibers. The paper now states this corollary and limits the diagnostic and its training regularizer to earlier or noninjective predictive maps. Under a declared coarsening \(S\), the theorem is applied to the new map \(\bar F=S\circ F\), effect \(d\bar F\,X\), and geometry of the coarsened image; merely conditioning the original tangent field on \(S(F)\) is not covered.

### Accepted: general probability-coordinate constants are not LLM certificates

The original general-alpha estimates depend on vocabulary size, a uniform token-probability floor, and inverse Fisher eigenvalues. They are valid structural estimates but are not calibrated finite-scale certificates for a large-vocabulary model. The audit's specific claim of a \(10^{30}\) constant was not derived and is not adopted.

The paper now proves a stronger result. With the square-root map

\[
\psi_p=2(\sqrt{p_1},\ldots,\sqrt{p_N}),
\]

\[
g_{ij}=\langle\partial_i\psi_p,\partial_j\psi_p\rangle,
\qquad
\Gamma^{LC}_{ij,k}
=\langle\partial_{ij}\psi_p,\partial_k\psi_p\rangle.
\]

This gives Levi--Civita metric, connection, transport, and curvature stability constants with no explicit vocabulary-size or minimum-probability factor. A Hilbert-valued Sobolev argument then converts integrated KL into Levi--Civita transport convergence.

For a general alpha-connection,

\[
g_{ij}=\mathbb E_p[S_iS_j],
\qquad
C_{ijk}=\mathbb E_p[S_iS_jS_k],
\]

\[
\Gamma^{(\alpha)}_{ij,k}
=\mathbb E_p[(\partial_iS_j)S_k]
+\frac{1-\alpha}{2}\mathbb E_p[S_iS_jS_k].
\]

Thus nonzero-alpha stability additionally requires control of the raised third score moment. A tokenwise probability floor is sufficient but not necessary. At an affine head these are centered decoder moments and the raw probability denominators cancel.

In square-root coordinates the distinction is explicit:

\[
C_{ijk}
=2\sum_a
\frac{\partial_i\psi_a\,\partial_j\psi_a\,\partial_k\psi_a}
{\psi_a}.
\]

This formula shows that square-root \(C^2\) bounds alone do not control a
general nonzero-alpha connection near a vanishing component. It does not say
that the tensor must diverge: its numerator can cancel the denominator, and
affine-head score moments provide a concrete floor-free route. The paper now
states the exact equivalence between stability of a fixed nonzero-alpha
connection and stability of the raised cubic tensor once Levi--Civita
stability is known.

### Corrected: the Levi--Civita theorem is not intrinsically short-path

The latest audit correctly noticed that the original Gronwall exponent could
be large, but its conclusion \(B_2L_\gamma\lesssim10^{-4}\) does not follow.
Levi--Civita transport is metric compatible. Under

\[
\lambda I\preceq g,\widetilde g\preceq B_1^2I,
\]

each subpath transport has background operator norm at most
\(B_1/\sqrt\lambda\). The exact Duhamel identity therefore gives

\[
\|P_\gamma^{p,LC}-P_\gamma^{\widetilde p,LC}\|_{\mathrm{op}}
\le L_\gamma\frac{B_1^2}{\lambda}D_{LC}.
\]

The manuscript now takes the minimum of this polynomial-conditioning bound and
the generic Gronwall bound. Pointwise pilot eigenvalues still show materially
worsening conditioning, but they are not the theorem's uniform path constants.
The pilot did not measure \(B_2,\delta_1,\delta_2\), or along-path extrema, so it
does not instantiate either finite-scale certificate. The protocol and code now
make those missing measurements explicit; at an affine head \(B_2\) is linked
exactly to a fourth score moment. In fact,

\[
\|D^2\psi_x\|_{\mathrm{op}}
=\frac12\sqrt{\sup_{\|u\|\le1}\mathbb E_{p(x)}[S_Y(u)^4]},
\]

so the fourth-moment envelope is not merely a directional heuristic. Numerical
directions still give only lower bounds unless their optimization is certified.

### Strengthened: the nonzero-alpha separation is pairwise

The original Bernoulli example showed an unbounded nonzero-alpha coefficient
under bounded square-root derivatives and \(g=1\). The paper now proves the
stronger stability failure on one fixed compact chart: paired phase-shifted
Bernoulli maps converge to one another in square-root \(C^2\), retain identical
unit Fisher metrics, and have every fixed nonzero-alpha connection defect diverge.
Thus alpha zero is unique only in the precisely scoped sense of stability from
square-root \(C^2\) agreement plus a spectral floor; the paper does not claim that
Levi--Civita is the only natural connection.

### Added: an intrinsic transport comparison

The coordinate Euclidean certificate remains useful for an operational hidden
chart, but a second exact consequence of the Duhamel identity is now stated in
mixed Fisher norms:

\[
\|P^{g,LC}-P^{\widetilde g,LC}\|_{\widetilde g_0\to g_1}
\le \int
\|\nabla^{g,LC}-\nabla^{\widetilde g,LC}\|_{
\widetilde g,\widetilde g\to g}
\|\dot\gamma\|_{\widetilde g}\,dt.
\]

This removes a separate chart-condition-number multiplier, not conditioning
itself: rank sensitivity is absorbed by the mixed norm of the connection defect.

### Rejected overclaims

- Exact shared-image naturality is restrictive for independent models but has direct reparameterization, duplication, and exact-distillation instances.
- Different curvature values at selected unaligned points and planes do not rule out every possible Fisher isometry.
- The cross-model Pythagorean decomposition needs a differentiable field alignment; exact Fisher isometry is stronger than the identity requires.
- A large generic Gronwall exponent does not prove an intrinsic short-path barrier for Levi--Civita transport; metric compatibility supplies a nonexponential bound.
- \(B_1^2/\lambda\) equals a condition number only for pointwise-tight constants. Uniform theorem bounds use a supremum of \(\lambda_{\max}\) and an infimum of \(\lambda_{\min}\), which need not occur together.
- The repository URL is misspelled in English but is the actual working remote.

### Textual repairs

- The counterexample now says \(g_{p_n}(0)\not\to g_{p_*}(0)\), rather than saying the constant sequence \(g_{p_n}(0)\) does not converge.
- The malformed author email was removed pending confirmation.
- The abstract and introduction now describe the same result hierarchy.

## Empirical audit

### Accepted: Fisher-Haar controls are spectrally confounded

An isotropic Fisher-Haar plane is a valid literal null, but it does not preserve the semantic plane's alignment with the Fisher spectrum. In the historical pilot, the semantic-minus-control difference grew strongly with metric condition number. It cannot support a semantics-specific interpretation.

The primary implemented control now:

1. Fisher-orthonormalizes the semantic plane;
2. whitens it in the Fisher eigensystem;
3. partitions the spectrum at preregistered relative eigengaps;
4. applies Haar rotation within each multi-axis band and a sign in each singleton band;
5. maps the plane back.

This preserves every per-band leverage contribution, is invariant to arbitrary basis choice inside repeated eigenspaces, and randomizes orientation relative to decoder third moments. Fisher-Haar controls remain a secondary null.

The critique that a fine relative-gap rule can collapse to mostly singleton
sign flips is accepted. The executable now also supports coarser relative-log
bands (one decade by default), preserves leverage at that declared resolution,
and records the band rule and sizes. The protocol requires this as a sensitivity
analysis; a conclusion that changes with the banding rule is reported as
control-sensitive.

### Accepted: eight controls cannot produce significance

With eight controls the smallest attainable plus-one tail rank is

\[
\frac{1}{8+1}=\frac19.
\]

The historical z-scores are descriptive standardized differences, not tests. The new tail ranks are calibrated randomization p-values only under the explicitly stated conditional block-group-invariance null; otherwise they remain descriptive. The primary protocol uses 999 matched controls, a two-sided alternative, and multiplicity correction.

### Accepted: the first-order alpha estimator has finite-edge bias

Integrated transport differs from the first-order approximation by \(O(\ell^2)\), while the connection signal is \(O(\ell)\). Dividing the two gives generic \(O(\ell)\) parameter bias. Its sign is decoder dependent; the earlier claim of universally downward bias was rejected.

The estimator now reports the full Fisher edge-length distribution, a dimensionless excitation ratio, and unrestricted, exponential, Levi--Civita, and mixture residuals. It rejects weak excitation below the preregistered threshold. The protocol requires at least three nested length scales with zero-length extrapolation or a direct integrated-transport fit. The executable synthetic finite-edge validation now uses integrated targets at four nested scales. Exact exponential and mixture transports do not require RK4 step convergence.

### Accepted: numerical gates must be executable

The code now separates and enforces:

- metric eigenvalue ratio threshold \(10^{-12}\);
- normalized plane Gram threshold \(10^{-4}\);
- relative linear-solve residual threshold \(10^{-10}\);
- condition-weighted solve-residual threshold \(10^{-4}\);
- same-plane Fisher-orthonormalization invariance;
- independent vocabulary-block recomputation, with a declared \(10^{-7}\) scaled curvature tolerance.

The code also checks the square-root cubic identity and both branches of the
finite-scale Levi--Civita stability certificate. Transport and field fitting now
apply \(G^{-1}C(u,v,\cdot)\) directly and reuse the cached spectral factorization,
rather than materializing a full cubic operator for every action. The direct and
materialized paths are tested for equality.

All thresholds are exposed by the real-model command and stored in new result payloads. No ridge is added to the primary geometry.

### Still required before scientific interpretation

- run the spectrum-matched protocol on preregistered prompt files;
- use multiple model seeds and checkpoints;
- use continuous tangent fields or declare finite chords as exponential-coordinate interventions;
- compare integrated e, LC, m, and fitted-alpha transport on held-out endpoints;
- make behavioral transfer the primary outcome;
- test whether curvature predicts controlled small-loop/order effects beyond entropy, edge length, conditioning, and linguistic interaction.

## External review rounds (August 2026)

Four adversarial reviews were run against the manuscript, the protocol, the
distillation specification, and the implemented packet core. Findings are
recorded here whether accepted or refuted.

### Round 1 --- manuscript and pilot

Seven findings. Four restated limitations the manuscript already disclosed
(no semantic evidence yet; ambient-versus-intrinsic pilot geometry; Sobolev
and rank hypotheses; almost-sure, distribution-relative semantic descent) and
one asked for engagement with literature already cited and discussed in the
related-work section. Two findings contained accepted parts. The source-level
repairs landed in commit `3c422f8`:

- **Accepted:** "aligned tangent norms" was used undefined. The cross-model
  commutator corollary now defines the aligned norm as the \(\dd\Phi\)
  pullback of the background norm, states the equivalent form on \(M_A\) that
  the proof bounds, and gives the extra \(\sup\|\dd\Phi\|\) factor incurred
  under independently chosen norms.
- **Accepted:** the transport commutation score left its norm unspecified.
  Commit `3c422f8` made the endpoint and base-point Fisher norms explicit;
  Round 3 later strengthened the denominator to the aligned input norm to
  remove the uniform-scale gaming mode.
- **Accepted:** the claim that the current command reproduced the historical
  pilot's control draws was false because the archived JSON predates the
  order-independent hashed seeding. The protocol now says exactly which
  deterministic quantities reproduce and why the regenerated controls differ.
  The PDF-staleness half of the same finding remains an explicit build task.

### Round 2 --- distillation specification

Twelve findings, all accepted and repaired in commit `d0a4a59`, notably:
stencil exposure was unequal across arms (the chart output-KD term now runs
over the identical stencil set in every arm, making D1 the exposure control);
the cubic estimator divided probability differences by \(q^2\) instead of
using the manuscript's own score-moment identity; D5's per-packet scramble
produced a discontinuous target field; the packet-to-transport target omitted
a sampling term; the band-limited Sobolev quantity was called a certificate;
and the "compact packet" claim ignored the full-vocabulary output anchor.

### Round 3 --- implemented packet core

Seven findings. One reproduction was factually wrong on this code (the
proposed invalid-simplex examples were rejected for degenerate rank, not
accepted), but its underlying vulnerability was real and confirmed with a
corrected \(z\)-dependent counterexample. The remainder were accepted and
repaired in commit `db9e77b`:

- **Accepted, executable defect.** `alpha_connection` raises the final index
  of \(L\) and \(C\) while `tensor_operator_norm` unfolded the first index,
  so the implemented Section 11.2 bound could be violated; a reproduction
  reached a defect/bound ratio of 1.48. `tensor_operator_norm` now takes an
  `output_axis` argument, the violating instance is a regression test, and a
  5000-trial adversarial stress over random metrics, \(\alpha\), and
  perturbation scales peaks at 0.89. The proposition itself was unaffected:
  it assumes compatible norms, which the implementation had not supplied.
- **Accepted:** the builder accepted invalid probability vectors. It now
  validates the finite open simplex and normalization, and
  `build_packet_from_logits` provides a precision-safe entry point with
  float64 enforcement and optional exact logit Jacobians.
- **Accepted:** `context_shuffled_cubics` rolled individual packets. It now
  moves complete donor-context fields at matching chart keys within strata
  and rejects singleton strata. D5 is documented as a coherent donor-field
  control, **not** a realizability control, since the composite
  recipient-\((G,L)\) plus donor-\(C\) target need not be the jet of any one
  predictive map.
- **Accepted:** \(\kappa(\Phi)=\sup\|\dd\Phi\|\sup\|\dd\Phi^{-1}\|\) cannot
  detect uniform rescaling. The score's denominator is now the aligned input
  norm, and both Lipschitz factors are reported separately.
- **Accepted:** grid Hölder quotients were called an explicit transport
  guarantee. Section 11.3 now separates certified from sampled moduli.
- **Accepted:** the serialization checksum covered only tensor payloads, and
  compute matching was asserted rather than designed. Both repaired.

### Round 4 --- repository red team

Adversarial numerical probes beyond the test suite confirmed Levi--Civita
metric compatibility (relative error \(10^{-15}\)), the Amari \(\pm\alpha\)
duality pairing (\(10^{-16}\)), loop transport against the closed-form
holonomy angle (\(10^{-12}\)), agreement between the finite-difference packet
builder and the independent exact-moment implementation (\(10^{-13}\)), and
matrix-valued transport-bound domination in a two-dimensional chart. No
mathematical defect was found. Four hygiene findings were accepted and
repaired: a vestigial `excluded_outcomes` field that strict validation had
made permanently zero; a non-portable `allow_nan` checksum; this ledger's own
omission of the external rounds; and a metric-norm-blind eigenvalue tolerance
on deserialization.

The packet envelope is now strict JSON, while its checksum is computed from a
labeled, length-prefixed binary encoding with explicit shapes and little-endian
IEEE-754 values. It therefore does not depend on Python-specific JSON float
spellings and can be reproduced by a cross-language shard reader.

The hygiene repair itself introduced and then corrected an error worth
recording. Removing the exclusion path came with the claim that both cubic
estimators are finite on every returned packet. That is false: squaring a
strictly positive but tiny float64 probability underflows to zero, so the
probability-difference audit can be non-finite while the score-moment
primary estimator remains exact. Confirmed at \(q_{\min}\approx10^{-174}\).
The schema therefore represents a non-finite audit as `null` with an explicit
`cubic_audit_finite` flag, requires finiteness only of the primary tensors,
and carries a regression test. The eigenvalue tolerance was likewise replaced
by a principled Weyl bound: the stored
`serialization_metric_eigenvalue_error_bound` is
\(\|G-G_{\mathrm{float32}}\|_2\), which bounds the eigenvalue perturbation
introduced by float32 payload quantization, plus numerical slack.

### Standing limitations these rounds did not remove

- The committed PDF is stale in mathematical content and must be rebuilt.
- No independent re-derivation of the manuscript's analysis has been
  performed; passing tests and surviving review are evidence of care, not
  proof of correctness.
- The certified surrogate residual of Section 11.4 remains open, so the
  risk--regularity route and any sampled Hölder constants stay conditional.
- Sections 13.2--13.4 of the distillation protocol --- the real-model packet
  builder, student trainer, and evaluator --- are unimplemented, and no
  empirical semantic result exists.

## Outcome logic

- If exponential transport wins, ordinary vector reuse is the best tested law for that operation.
- If Levi--Civita wins, Fisher-metric-compatible transport improves the tested transfer.
- If mixture or an intermediate alpha wins, third-order probabilistic structure matters, but Fisher-metric preservation is not the whole explanation.
- If no alpha-family member generalizes, the decoder Fisher geometry or the parallel-field model is insufficient.
- If cross-model transport agreement adds nothing beyond output KL and standard representation similarity, it is not a useful marker of generalization.

The historical Pythia-14M JSON remains an execution record only. No semantic, training, or cross-model conclusion is drawn from it.
