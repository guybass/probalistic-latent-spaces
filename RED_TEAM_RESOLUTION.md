# Red-Team Resolution

Date: 2026-08-06

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

Thus nonzero-alpha stability additionally requires third score-moment control. At an affine head these are centered decoder moments and the raw probability denominators cancel.

### Rejected overclaims

- Exact shared-image naturality is restrictive for independent models but has direct reparameterization, duplication, and exact-distillation instances.
- Different curvature values at selected unaligned points and planes do not rule out every possible Fisher isometry.
- The cross-model Pythagorean decomposition needs a differentiable field alignment; exact Fisher isometry is stronger than the identity requires.
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

All thresholds are exposed by the real-model command and stored in new result payloads. No ridge is added to the primary geometry.

### Still required before scientific interpretation

- run the spectrum-matched protocol on preregistered prompt files;
- use multiple model seeds and checkpoints;
- use continuous tangent fields or declare finite chords as exponential-coordinate interventions;
- compare integrated e, LC, m, and fitted-alpha transport on held-out endpoints;
- make behavioral transfer the primary outcome;
- test whether curvature predicts controlled small-loop/order effects beyond entropy, edge length, conditioning, and linguistic interaction.

## Outcome logic

- If exponential transport wins, ordinary vector reuse is the best tested law for that operation.
- If Levi--Civita wins, Fisher-metric-compatible transport improves the tested transfer.
- If mixture or an intermediate alpha wins, third-order probabilistic structure matters, but Fisher-metric preservation is not the whole explanation.
- If no alpha-family member generalizes, the decoder Fisher geometry or the parallel-field model is insufficient.
- If cross-model transport agreement adds nothing beyond output KL and standard representation similarity, it is not a useful marker of generalization.

The historical Pythia-14M JSON remains an execution record only. No semantic, training, or cross-model conclusion is drawn from it.
