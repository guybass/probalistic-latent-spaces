# Red-Team Resolution

Date: 2026-08-06; external review rounds appended through 2026-08-18

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

### Round 5 --- combined mathematical closure

The Opus 5 audit reported independent hand derivations of its 16-result core
table, and a second repository-wide audit found no theorem-level
counterexample. The current manuscript contains 21 theorem, proposition, and
corollary environments. A strict proof-environment audit found that four
corollaries had correct but implicit or omitted proofs. This round made every
one explicit:

- `cor:intrinsic-field` now follows by differentiating
  (F_A=F_B\circ\Phi) and using injectivity of (dd F_B);
- `cor:two-model-convergence` compares both transports to the common target
  transport and uses the operator-norm triangle inequality, not a nonexistent
  KL triangle inequality;
- `cor:connection-decomposition` now explicitly takes the infimum of the
  pointwise Pythagorean identity; and
- `cor:cross-model-semantic` now explicitly applies the semantic-obstruction
  theorem to the transferred model-A field on model B.

All 21 formal results now have explicit proofs. Static validation finds balanced
LaTeX environments, no duplicate labels, and no undefined references.

The same round strengthened the finite-scale interpretation of the square-root
Levi--Civita stability theorem. Substituting (D_{LC}) into the
metric-compatible transport branch gives

\[
L_\gamma\left[
\frac{B_1^2(B_1\delta_2+B_2\delta_1)}{\lambda^2}
+\frac{2B_1^4B_2\delta_1}{\lambda^3}
\right].
\]

The appendix now records the archived pointwise spectral diagnostics. From
initialization to `step143000`, the worst inverse, inverse-square, and
inverse-cube factors grow by approximately (2.31\times10^2),
(5.34\times10^4), and (1.23\times10^7). This does not alter the theorem, but
it makes explicit that vocabulary independence does not imply a nonvacuous
trained-model certificate. The pilot still lacks (B_2), (delta_1),
(delta_2), and pathwise extrema, so no complete numerical certificate is
claimed.

### Round 5 remediation --- protocol, code, artifacts, and statistics

The same audit found nine implementation and numerical defects. The current
checkout resolves them without changing the mathematical results:

- centered-logit Jacobians use the declared `(chart_dimension, outcomes)`
  layout and remove rowwise softmax gauge;
- packet refinement uses a stored pure-relative primary tolerance (zero
  dimensionful absolute floor), supplied
  exact JVPs are checked independently against finite-difference logits, and
  algebraic rank, relative conditioning, and absolute chart scale are distinct
  gates;
- packet schema v4 binds immutable model/tokenizer/outcome/chart/context/dtype
  provenance, validates accepted tensors and gate decisions, and refuses
  untracked training serialization;
- continuum transport-bound results carry mandatory `certified` or `sampled`
  status and a source rather than returning an unlabeled scalar;
- Fisher small-distance/log calculations use a stable chord formula,
  exponential boundary calculations are stabilized, and unrepresentable
  float64 endpoints fail closed;
- the sufficiency API declares empirical versus population-unbiased estimands,
  the one-dimensional Sobolev audit uses the full grid with composite Simpson
  quadrature and resolution checks, quantization error is truly relative, and
  transport-bound overflow returns infinity;
- new Pythia smoke artifacts use result schema `pythia-smoke-2`, immutable
  snapshot commits, distinct model/control seed fields, tokenizer/outcome/design
  hashes, repository state, runtime versions, and lock hashes; and
- the protocol now specifies the realized crossed/nested sampling analysis,
  concrete effect/NLL/coverage/feasibility/chart-scale/certificate thresholds,
  multiplicity hierarchy, intention-to-treat failure handling, and an 80%-power
  rule.

All 100 model-free unit tests pass, including every adversarial regression from
the combined audit; Ruff, editable packaging, `pip check`, all three synthetic
drivers, and a cached one-checkpoint `pythia-smoke-2` run pass. Static manuscript
checks find no missing citation key, undefined reference, duplicate label,
unused bibliography entry, or environment imbalance. The PDF remains unbuilt
because no LaTeX engine is installed.

### Follow-up remediation --- small-chart acceptance and Duhamel sign

A verification pass correctly found that the repaired refinement diagnostic
could still yield a false acceptance: the shipped \(10^{-6}\) absolute floor
dominated \(r_{\rm ref}\|L\|\) on small charts. The diagnostic reported the
large relative discrepancy, but the acceptance ratio used the dimensionful
absolute leg. The primary defaults now freeze both `refinement_atol=0` and
`jvp_atol=0`; a nonzero absolute floor is permitted only as a separately
labeled sensitivity analysis backed by a recorded tensor-specific
inference/stencil roundoff calculation in the frozen chart.

The new regression constructs an analytic oscillatory Bernoulli chart whose
first-kind tensor has more than 10% relative error at chart scale \(10^{-4}\).
The former default accepted it; the primary gate now rejects it as
`refinement`.

### Follow-up remediation --- chart-invariant acceptance

The \(a_{\rm ref}=0\) repair above over-corrected, and a second verification
pass found the regression. A pure relative rule on raw tensors is ill posed
when a tensor is structurally zero: it divides roundoff by roundoff. The
boundary Bernoulli family has \(\Gamma^{LC}\equiv0\) because
\(\langle\partial_{\theta\theta}\psi,\partial_\theta\psi\rangle=0\), so
`build_packet` rejected that family at every \(\theta\), with `metric` and
`cubic` converged to \(3\times10^{-12}\) and \(1\times10^{-11}\) relative and
`cubic` equal to \(2\cot\theta\) to eleven digits. The same held for any
antipodal head at its symmetric point. The rejection reason was also false:
it read `refinement` when refinement had demonstrably converged, repeating
the mislabelling that the `rank`/`conditioning`/`metric_scale` split had just
removed. Acceptance additionally depended on whether roundoff happened to
cancel to exactly `0.0`, which it does for a one-dimensional symmetric head
and does not for an antipodal multi-token head. Both were disclosed in a
code comment, and the closed-form Bernoulli test had been rewritten to assert
the rejection, so the suite stayed green while the paper's own separating
family could no longer yield an accepted packet.

Two candidate repairs were rejected on evidence. Restoring a dimensionful
\(a_{\rm ref}\) reinstates the original chart-scale defect. Deriving
\(a_{\rm ref}\) from a propagated stencil-roundoff floor was measured and
fails: the \(56\%\)-error small-chart case has an absolute difference of
\(6.6\times10^{-16}\), below any defensible roundoff floor, so it would be
re-admitted.

The accepted repair changes what is compared rather than the tolerance
constant. The acceptance rule now runs on chart invariants: the dimensionless
relative metric defect, and the Fisher norms of the raised \(G^{-1}L\) and
\(G^{-1}C\). Both are unchanged by \(z\mapsto sz\), so \(a_{\rm ref}\) becomes
dimensionless and the primary gate freezes
\(a_{\rm ref}=10^{-6}\), \(r_{\rm ref}=10^{-3}\). The reported relative error
is regularized by the same constant so that it stays well defined at zero
scale and monotone with the verdict. The JVP audit keeps \(a_{\rm JVP}=0\):
it compares two estimates of one gauge-centred Jacobian, whose scale cannot
vanish on a chart direction surviving the rank gate.

Measured behaviour on the seven-case fixture: the boundary Bernoulli family at
\(\theta=1\) and \(\theta=\pi/2\), an antipodal eight-token head with
\(L=C=0\), and accurate affine charts are accepted; the \(16.5\%\) and
\(55.8\%\) small-chart cases are rejected as `refinement`. Across eight orders
of magnitude of chart units at fixed physical stencil, the refinement
diagnostic stays at \(3\times10^{-8}\) and the verdict never moves; the only
surviving unit-dependent gate is the truthfully named `metric_scale`, whose
preregistered dimensionless partner is `metric_relative_floor`. Two
regressions lock this in, and the closed-form Bernoulli test asserts
acceptance again. The packet schema is `pcd-packet-5`: the field names are
unchanged but `refinement_error`, `refinement_absolute_error`, and
`refinement_tolerance_ratio` now carry invariant units, so a v4 reader would
misinterpret them.

The same pass corrected the sign in the packet-to-transport Duhamel display:

\[
\Phi_T(1,0)-\Phi_S(1,0)
=\int_0^1\Phi_S(1,s)(A_T(s)-A_S(s))\Phi_T(s,0)\,ds.
\]

The previous sign did not affect the norm bound, but it was not the exact
identity under the displayed convention. At that checkpoint the full suite
contained 100 passing tests.

### Follow-up remediation --- packet release blockers

Four executable attacks against the packet bridge were then accepted. First,
the nested \((h,h/2,h/4)\) grid could alias a sinusoidal logit perturbation and
accept a metric with 494% relative error; even a false declared JVP passed
because its audit reused the aliased grid. Packet construction now requires a
second \(1/\sqrt2\)-rescaled ladder, checks convergence within both ladders and
agreement between them, and audits supplied JVPs on both. Exact JVPs form the
metric and score cubic directly. The original attack is rejected as
`refinement`, and its false-JVP variant as `jvp_consistency`.

Second, binary32 quantization could make an accepted SPD metric singular while
the reader retained the pre-quantization eigenvalues and `accepted=True`.
Schema `pcd-packet-6` stores eigenvalues recomputed from the actual float32
metric, reruns the ordered gate on that payload, and refuses serialization if
the verdict changes. The perturbation bound remains diagnostic; it no longer
serves as tolerance for disagreement with the stored matrix.

Third, a center inside declared chart bounds could use stencil points outside
them. The builder now validates the largest axial and mixed stencil before the
first teacher call. Fourth, rejected-packet validation previously trusted any
recognized reason. It now recomputes the complete ordered gate for accepted
and rejected records and rejects mismatched reasons. Four adversarial
regressions cover these cases; the full model-free suite contains 106 tests.

### Round 6 --- mathematics-only red team and the deferred-proofs appendix

A review restricted to the manuscript graded every one of the 21 formal
results against the standard of a strict analysis course: each hypothesis
actually used, each "thus" earned, each quantifier and constant tracked. **No
false statement was found.** Every deduction was of one kind --- a true claim
presented as immediate --- and their distribution was the finding: the three
least-justified steps all lay on the single path from "the risk is small" to
"Levi--Civita transport is close, with constants free of \(N\)", which is the
paper's advertised contribution.

The repair is Appendix~B, `Deferred proofs`, which supplies fifteen lemmas and
their proofs and is cross-referenced from every point in the main text that
previously asserted. The manuscript now contains 36 formal results and 36
proofs.

**The principal gap.** The proof of the boundary-robust risk-to-transport
theorem read, in full, "Hilbert-valued Sobolev interpolation and embedding
give ...". That single clause carries two distinct theorems, neither proved
nor cited, and the second is the whole of the vocabulary-independence claim:
a componentwise application of the scalar Sobolev embedding to the \(N\)
coordinates of \(\psi_p\) produces a factor \(\sqrt N\) and destroys it. The
appendix now proves both. Lemma `hilbert-interpolation` obtains the
interpolation inequality with constant exactly one, by Hölder on the scalar
integrand \((1+|\xi|^2)^r\|\hat f(\xi)\|^2\). Lemma `hilbert-embedding` gives
the embedding with the explicit constant

\[
\kappa=(2\pi)^{-m/2}\max_{|\beta|\le k}
\Bigl(\int_{\mathbb R^m}|\xi|^{2|\beta|}(1+|\xi|^2)^{-r}\,d\xi\Bigr)^{1/2},
\]

finite exactly when \(r>k+m/2\), by applying Cauchy--Schwarz to two
nonnegative *scalar* functions of \(\xi\); the Hilbert structure enters only
through the Bochner triangle inequality and a vector-valued Plancherel lemma,
so no step refers to the target dimension. Lemma `hilbert-domain` transfers
both to bounded Sobolev-extension domains and compact manifolds using Stein's
total extension operator, which is a scalar kernel and therefore has the same
norms for every target. A discrete check confirms the conclusion: over
\(N=2,10,10^2,10^3,10^4\) the ratio \(\sup_x\|f(x)\|/\|f\|_{H^r}\) reads
\(0.00714, 0.00745, 0.00730, 0.00711, 0.00706\) --- flat across four orders of
magnitude --- and the interpolation ratio stays below one throughout.

**Other deferred proofs now supplied.** The probability-coordinate identity for
\(\Gamma^{(\alpha)}_{ij,k}\) was used twice and proved nowhere; it is now
Lemma `lower-alpha`, derived from the paper's own definition by the Koszul
cancellation. The identification of \(\alpha=\pm1\) with the exponential and
mixture connections was asserted with "thus" and leaned on twice; it is now
Lemma `alpha-identification`, which computes \(\Gamma^{(\alpha)}=\frac{1-\alpha}{2}C\)
on an exponential family from the score-moment form and then shows directly
that the expectation coordinates \(\eta=\nabla\psi\) are
\(\nabla^{(-1)}\)-affine. The Hellinger--Kullback inequality
\(\sum_a(\sqrt{p_a}-\sqrt{q_a})^2\le D_{KL}(p\Vert q)\), the sole bridge from
risk to square-root coordinates, is now proved from Jensen and
\(\log t\le t-1\). The two-model \(L^1\) bound silently used Jensen alongside
the two tools it named; the step is now explicit and flagged as
non-removable. The kernel identity \(\ker G=\mathcal N\), and the fact that
the affine-head quotient *embeds* rather than merely injectively immerses,
are proved via an intermediate log-odds chart. A remark states the hypothesis
under which a general predictive quotient is a manifold, and records that no
estimate in the paper needs it.

**Two steps that were correct for unstated reasons.** The mean-value argument
behind the \(C^2\) stability constant requires the regularity class to be
*convex*, or the mean-value inequality does not apply along the connecting
segment; Lemma `lipschitz-class` establishes convexity and compactness first.
The transport constant \(L_\gamma e^{M_\Gamma L_\gamma}\) is correct only
because the two Gronwall growth factors, \(e^{M(L_\gamma-s(t))}\) and
\(e^{Ms(t)}\), multiply to a quantity independent of \(t\); bounding each
propagator separately gives the weaker \(e^{2M_\Gamma L_\gamma}\). Lemma
`duhamel-cancellation` performs the cancellation, notes the weaker
alternative, and records that unequal bounds give
\(e^{\max\{M,\widetilde M\}L_\gamma}\) --- which also corrects a claim made
during review that the sum \(M_T+M_S\) in Section 11.3 of the distillation
protocol was sharp. It is valid but not sharp.

**Scope repair.** Theorem 5.1 is stated on a Euclidean chart and was then
applied on a compact manifold. Lemma `chart-transfer` supplies the atlas
argument and states the constants it introduces.

Every new lemma was verified numerically as well as proved: the
probability-coordinate Christoffel identity and the \(\pm\alpha\) duality
against an independent Koszul computation on a non-exponential family
(\(2\times10^{-6}\), the finite-difference floor), the exponential-family
specialisation and \(g=\mathrm{Hess}\,\psi\) (\(10^{-6}\), \(10^{-8}\)), the
variance form of \(u^\top Gu\) (exact), and the Duhamel bound against an
integrated transport (actual \(0.1447\) against the sharp bound \(0.3532\) and
the naive bound \(1.1531\)).

The static audit reports 36 results, 36 proofs, no undefined reference, no
duplicate label, no missing citation, no unused bibliography entry, no
unbalanced environment, and no unreferenced new lemma.

### Follow-up remediation --- confirmatory statistics and release artifacts

The confirmatory distillation claim is now one candidate-level
intersection--union test per D3/D4 arm. Each global p-value is the maximum over
behavioral superiority, held-out transport superiority, NLL noninferiority,
and intention-to-treat feasibility noninferiority against every required
control. Holm controls the frozen candidate family at 0.05; operation-specific
claims enlarge that same family. The protocol now fixes the effect directions,
one scalar behavioral endpoint, a pre-confirmatory D1 standardization scale,
decision-compatible one-sided bounds, and the distinction between the observed
0.20-SD gate and a population effect claim. Curvature randomization tests are
explicitly a separate mechanistic family, and the stale later list of
"primary comparisons" is now secondary. A versioned sampling/analysis manifest
and pilot-derived seed count remain mandatory before execution.

The manuscript was rebuilt from a clean directory with Tectonic 0.17.0 after
making the first-line `\\pdfoutput=1` directive engine-aware. Two overfull
lines were repaired. The final 31-page build has no undefined references,
undefined citations, duplicate labels, overfull boxes, or TeX errors. The
source audit resolves 149 references to 98 unique labels, cites all 20
bibliography entries with no missing or unused keys, and balances all 36 proof
environments. All pages were rendered and visually inspected. The generated
`paper/main.bbl` and `output/pdf/predictive_geometric_agreement.pdf` are the
verified submission artifacts. Citation review also corrected
arXiv:2607.04525 from an unsupported ICML/PMLR classification to a preprint.

### Standing limitations these rounds did not remove

- Two adversarial reviews found no mathematical defect, but no complete
  line-by-line derivation ledger is committed; explicit manuscript proofs and
  surviving review are evidence of care, not formal certification.
- The certified surrogate residual of Section 11.4 remains open, so the
  risk--regularity route and any sampled Hölder constants stay conditional.
- Sections 13.2--13.4 of the distillation protocol --- the real-model packet
  builder, student trainer, and evaluator --- are unimplemented, and no
  empirical semantic result exists.
- A confirmatory prompt/operation sampling manifest and pilot-derived power
  calculation do not yet exist; the protocol blocks population claims and a
  confirmatory run until they do.
- No software license has been selected. Package metadata and CI configuration
  now exist, but reuse rights remain intentionally unspecified until the author
  makes that legal choice.

## Outcome logic

- If exponential transport wins, ordinary vector reuse is the best tested law for that operation.
- If Levi--Civita wins, Fisher-metric-compatible transport improves the tested transfer.
- If mixture or an intermediate alpha wins, third-order probabilistic structure matters, but Fisher-metric preservation is not the whole explanation.
- If no alpha-family member generalizes, the decoder Fisher geometry or the parallel-field model is insufficient.
- If cross-model transport agreement adds nothing beyond output KL and standard representation similarity, it is not a useful marker of generalization.

The historical Pythia-14M JSON remains an execution record only. No semantic, training, or cross-model conclusion is drawn from it.
