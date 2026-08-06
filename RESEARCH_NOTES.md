# Parallel Transport on Predictive Statistical Manifolds

> **Historical notebook.** The authoritative results are [paper/main.tex](paper/main.tex), the current experiment is [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), and the consolidated audit is [RED_TEAM_RESOLUTION.md](RED_TEAM_RESOLUTION.md).

Research notes and experiment specification
Date: 2026-08-04

> **Audit status.** This file records the development of the idea. The later
> mathematical and literature audits in [PROOFS_AND_NOVELTY.md](PROOFS_AND_NOVELTY.md)
> and the revised [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) are authoritative
> where they sharpen or supersede provisional statements below. In particular,
> pullback Fisher geometry and generic representation holonomy are not new claims;
> the narrowed target is exact Fisher--Levi-Civita decoder curvature plus a
> controlled three-connection composition benchmark.

## 1. Core question

Autoregressive pretraining learns conditional distributions

\[
p_\theta(\cdot\mid c)\in\Delta^{V-1},
\]

where \(c\) is a context and \(V\) is the vocabulary size. In practice, however, the hidden states associated with these conditional distributions are treated as elements of a global vector space: representations are added, subtracted, averaged, and compared using a fixed inner product.

The proposed research question is:

> Does semantic composition behave like addition in one global vector space, or should semantic displacements be treated as tangent vectors on a statistical manifold and compared using parallel transport?

A sharper version is:

> Which connection describes semantic composition: the mixture connection, the exponential connection, the Fisher--Rao Levi-Civita connection, or a learned connection?

The intended contribution is not the generic observation that probability distributions form a manifold, nor the generic use of parallel transport in representation learning. Both already exist. Pullback metrics and scalar softmax-cumulant checkpoint probes also already exist. The potentially new contribution is to construct the mixed-cumulant Fisher--Levi-Civita Riemann tensor of a trained next-token decoder, validate its intrinsic holonomy, and test whether these invariants predict which connection best models held-out semantic transformations.

## 2. Terminology that must remain separate

### 2.1 Shannon entropy, KL divergence, and Fisher--Rao metric

Shannon entropy is

\[
H(p)=-\sum_i p_i\log p_i.
\]

Kullback--Leibler divergence is

\[
D_{\mathrm{KL}}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}.
\]

The Fisher--Rao metric is the local quadratic form obtained from the second-order behavior of KL divergence. On the interior of the categorical probability simplex,

\[
g_p(u,v)=\sum_i\frac{u_i v_i}{p_i},
\qquad
\sum_i u_i=\sum_i v_i=0.
\]

The phrase "Shannon metric" is ambiguous. It can refer to metrics constructed from conditional entropy, such as variation of information, but those are not the same as the Fisher--Rao metric. In this project, all differential-geometric claims should explicitly name the Fisher--Rao metric unless a different metric is intended.

### 2.2 Points and tangent vectors

A distribution \(p\in\Delta^{V-1}\) is a point. A displacement \(u\in T_p\Delta^{V-1}\) is a tangent vector satisfying \(\sum_i u_i=0\).

Points at different locations cannot be subtracted or added intrinsically without additional structure. Tangent vectors in \(T_pM\) and \(T_qM\) also cannot be compared directly. A connection supplies a rule for transporting a tangent vector between tangent spaces.

### 2.3 Flatness is connection-dependent

A statistical manifold generally carries several connections:

\[
\nabla^{(1)} \quad \text{exponential connection},
\]

\[
\nabla^{(-1)} \quad \text{mixture connection},
\]

\[
\nabla^{(0)} \quad \text{Fisher--Rao Levi-Civita connection}.
\]

The Levi-Civita connection is the midpoint

\[
\nabla^{(0)}
=\frac12\left(\nabla^{(1)}+\nabla^{(-1)}\right).
\]

Each connection has its own curvature tensor \(R^{(\alpha)}\). A statement that a manifold is "flat" is incomplete unless the connection is named.

## 3. What *Geometric Modeling in Probability and Statistics* means by flatness

The relevant book is Ovidiu Calin and Constantin Udrişte, [*Geometric Modeling in Probability and Statistics*](https://link.springer.com/book/10.1007/978-3-319-07779-6), Springer, 2014.

The book uses affine-connection flatness. A torsion-free affine connection \(\nabla\) is flat when

\[
R^\nabla(X,Y)Z=0.
\]

Locally, this is equivalent to the existence of affine coordinates in which its Christoffel symbols vanish throughout a neighbourhood:

\[
\Gamma^k_{ij}=0.
\]

This is stronger than choosing normal coordinates that make Christoffel symbols vanish at one point.

The book shows that:

- exponential families are \(\nabla^{(1)}\)-flat in natural parameters;
- mixture families are \(\nabla^{(-1)}\)-flat in mixture parameters;
- when both dual connections are flat, the statistical structure is called dually flat.

These statements appear around [pages 31--33 of the public preview](https://books.google.com/books?id=D1wlBAAAQBAJ&pg=PA31). They do not imply that the Fisher--Rao Levi-Civita curvature \(R^{(0)}\) vanishes.

The book's statement around [page 275](https://books.google.com/books?id=D1wlBAAAQBAJ&pg=PA275) that a dually flat statistical manifold has curvature zero concerns the selected statistical affine connections. It means

\[
R^{(1)}=R^{(-1)}=0,
\]

not necessarily

\[
R^{(0)}=0.
\]

The book itself discusses models with nonzero Fisher curvature. Its chapter overview describes the normal-distribution manifold as having negative constant curvature; see [page 51](https://books.google.com/books?id=D1wlBAAAQBAJ&pg=PA51).

### 3.1 Why two flat connections can have a curved midpoint

Write

\[
\nabla^{(1)}=\nabla^{(0)}+K,
\qquad
\nabla^{(-1)}=\nabla^{(0)}-K,
\]

where \(K\) is the difference tensor, related to the Amari--Chentsov cubic tensor. Curvature depends nonlinearly on a connection. Under the usual statistical-manifold compatibility conditions, dual flatness gives schematically

\[
R^{(0)}(X,Y)=-[K_X,K_Y].
\]

Thus \(R^{(1)}=R^{(-1)}=0\) does not force \(R^{(0)}=0\).

### 3.2 The square-root representation in the book

The book's \(0\)-representation, around [page 69](https://books.google.com/books?id=D1wlBAAAQBAJ&pg=PA69), is

\[
\Phi_0(p)=2\sqrt p
=2(\sqrt{p_1},\ldots,\sqrt{p_n}).
\]

Because \(\sum_i p_i=1\),

\[
\|\Phi_0(p)\|^2=4.
\]

The image is the positive orthant of a sphere of radius \(2\). Moreover,

\[
\left\langle D\Phi_{0,p}(u),D\Phi_{0,p}(v)\right\rangle
=\sum_i\frac{u_i v_i}{p_i}
=g_p(u,v).
\]

Therefore the square-root representation is Fisher-isometric, but its image is a sphere, not a linear subspace. For a simplex of dimension at least two, its Fisher--Rao sectional curvature is

\[
K^{(0)}=\frac14.
\]

This resolves the apparent contradiction:

- the representation is global;
- the representation preserves the Fisher metric;
- the image is not closed under vector addition;
- the mixture and exponential connections are flat;
- the Fisher Levi-Civita connection is curved.

## 4. Established mathematical claims

The following are foundational results, not proposed new contributions.

### Claim E1: Global coordinates do not imply metric flatness

A manifold can have one global coordinate chart while its Riemannian curvature is nonzero. The upper half-plane model of hyperbolic geometry and the normal-distribution family are examples. Coordinates describe topology and differentiable structure; curvature depends on the metric and connection.

### Claim E2: An isometric embedding need not have a linear image

An isometric embedding into Euclidean space preserves the intrinsic metric. It does not turn the image into a vector subspace. The sphere is the canonical example. Ambient vector addition generally leaves the embedded manifold.

### Claim E3: The categorical Fisher manifold has positive Levi-Civita curvature

Let

\[
\Delta_+^{n-1}=\{p_i>0,\ \sum_i p_i=1\}
\]

with Fisher metric

\[
g_p(u,v)=\sum_i\frac{u_i v_i}{p_i}.
\]

The map \(p\mapsto2\sqrt p\) is an isometry onto the positive part of \(S^{n-1}(2)\). Hence, for \(n\ge3\),

\[
K^{(0)}=\frac14.
\]

### Claim E4: Cross-entropy identifies output conditionals, not hidden geometry

Let the data conditional be \(P(\cdot\mid c)\) and the model conditional be \(q_\theta(\cdot\mid c)\). Then

\[
\mathcal L(\theta)
=\mathbb E_c H(P(\cdot\mid c))
+\mathbb E_cD_{\mathrm{KL}}
\left(P(\cdot\mid c)\|q_\theta(\cdot\mid c)\right).
\]

At the population optimum, under ideal capacity and optimization assumptions,

\[
q_\theta(\cdot\mid c)=P(\cdot\mid c)
\]

almost surely on the data distribution. This constrains the output conditional map, not a unique geometry for internal states.

### Claim E5: Hidden geometry is non-identifiable from the likelihood alone

Write the model as

\[
q_\theta(\cdot\mid c)=D(H(c)).
\]

For any invertible transformation \(\varphi\), define

\[
H'=\varphi\circ H,
\qquad
D'=D\circ\varphi^{-1}.
\]

Then

\[
D'(H'(c))=D(H(c)).
\]

Thus the same conditional distributions admit different hidden geometries. For a linear unembedding \(\ell=Wh+b\), every invertible matrix \(A\) gives the symmetry

\[
h'=Ah,
\qquad
W'=WA^{-1}.
\]

A non-orthogonal \(A\) changes Euclidean angles and distances without changing any output probabilities. Therefore convergence of cross-entropy does not imply that hidden Euclidean geometry is Fisher-isometric.

### Claim E6: Parallel transport generalizes vector displacement

For \(p,q,r\in M\), define the transported analogy

\[
\mathcal A_\gamma(p,q;r)
=\operatorname{Exp}_r
\left(
P_{\gamma:p\to r}\operatorname{Log}_p(q)
\right).
\]

In Euclidean space with its flat Levi-Civita connection,

\[
\mathcal A(p,q;r)=r+q-p.
\]

On a curved manifold the result can depend on the path \(\gamma\).

### Claim E7: Infinitesimal holonomy measures curvature

For a small loop generated by tangent directions \(X,Y\), parallel transport satisfies, up to orientation convention,

\[
P_{\partial\Box_\varepsilon}v-v
=\varepsilon^2R(X,Y)v+O(\varepsilon^3).
\]

Consequently, path-independent Levi-Civita parallel transport implies zero Riemann curvature on a simply connected domain. Nontrivial holonomy rules out a single global metric-compatible translation structure.

## 5. Candidate model-level theorem

The following packages the geometric observation in a form relevant to language models. Its proof is short because it is an application of standard differential geometry; the novelty would come from identifying and measuring the model-derived metric.

### Theorem candidate: obstruction to global metric-compatible semantic vectors

Let \(U\subset\mathbb R^d\) be a smooth context chart, for example a low-dimensional continuous soft-prompt chart, and let

\[
F_\theta:U\to\Delta_+^{V-1},
\qquad
z\mapsto p_\theta(\cdot\mid z)
\]

be a smooth predictive immersion. If it has lower constant rank, first pass to a
regular local quotient by the null directions of \(F_\theta\); constant rank alone
does not make the degenerate pullback on all of \(U\) Riemannian. Equip the resulting
regular space with the pullback Fisher metric

\[
g_\theta=F_\theta^*g_{\mathrm{FR}}.
\]

Suppose there exists a metric-preserving coordinate map

\[
\Psi:(U,g_\theta)\to\mathbb R^d
\]

in which every semantic transformation in a specified family acts as a context-independent translation

\[
\Psi(T_a(z))=\Psi(z)+v_a.
\]

Then the Levi-Civita curvature of \(g_\theta\) must vanish on the region where this representation holds. Therefore, if

\[
R^{g_\theta}\ne0
\]

at any point in that region, no such global metric-preserving vector representation exists there.

#### Proof sketch

Euclidean space with its ordinary metric and translation structure has zero Levi-Civita curvature. Riemann curvature is invariant under local isometry. Therefore a metric-preserving Euclidean coordinate representation would force \(R^{g_\theta}=0\). The contrapositive gives the obstruction.

#### What this theorem does and does not say

It does not say that a curved statistical manifold cannot be embedded in a high-dimensional Euclidean space. It can. It says that the image cannot simultaneously be treated as an open Euclidean vector space with metric-compatible, context-independent translation.

It also does not rule out approximate linear behavior in a small region. Every smooth manifold is locally approximated by its tangent space to first order. The experiment must therefore measure whether curvature effects are large enough to matter at semantic scales.

## 6. Model-derived metric

### 6.1 Pullback from output probabilities

For a smooth context parameter \(z\), define

\[
p(z)=p_\theta(\cdot\mid z).
\]

The pullback Fisher metric is

\[
g_{ij}(z)
=\sum_{k=1}^{V}
\frac{\partial_i p_k(z)\,\partial_jp_k(z)}{p_k(z)}.
\]

Equivalently, using the square-root representation \(x(z)=2\sqrt{p(z)}\),

\[
g_{ij}(z)
=\langle\partial_i x(z),\partial_jx(z)\rangle.
\]

This metric is invariant under a reparameterization of the context chart when transformed as a tensor. It measures changes in the model's predictive distribution rather than arbitrary Euclidean changes in hidden coordinates.

### 6.2 Pullback onto a hidden state

If

\[
p(h)=\operatorname{softmax}(\ell(h)),
\]

then

\[
G(h)=J_\ell(h)^\top
\left[\operatorname{diag}(p)-pp^\top\right]
J_\ell(h).
\]

For a linear unembedding \(\ell(h)=Wh+b\),

\[
G(h)=W^\top
\left[\operatorname{diag}(p(h))-p(h)p(h)^\top\right]W.
\]

This metric depends on \(h\), even though \(h\in\mathbb R^d\) has global coordinates. A varying metric can have nonzero Levi-Civita curvature.

The metric can be degenerate. Its null directions are hidden perturbations that do not change the output distribution to first order. The correct statistical object is then the quotient of hidden space by these predictive null directions.

For computation, the full matrix should usually not be materialized. Its action on a vector is

\[
G(h)v
=J_\ell(h)^\top
F(p)
J_\ell(h)v,
\qquad
F(p)=\operatorname{diag}(p)-pp^\top,
\]

which can be calculated with Jacobian-vector and vector-Jacobian products.

## 7. Potentially nontrivial empirical claims

These are hypotheses, not established facts.

### Hypothesis H1: predictive curvature evolves during pretraining

For a fixed family of context charts, the pullback Fisher curvature of a model changes systematically across training checkpoints and differs from a randomly initialized network.

This is stronger than observing the known \(1/4\) curvature of the ambient Fisher simplex. It concerns the model-dependent pullback metric on a low-dimensional context surface.

### Hypothesis H2: semantic directions produce structured holonomy

Loops formed by controlled semantic transformations produce holonomy that is systematic across prompts and larger or more structured than matched random loops with similar Fisher lengths and areas.

Examples of transformation pairs include:

- singular/plural and present/past;
- affirmative/negative and formal/informal;
- active/passive and present/past;
- entity substitution and relation substitution in controlled templates.

The transformations should be selected so that their intended semantic actions approximately commute. Otherwise path dependence may merely reflect ordinary linguistic order effects rather than geometric curvature.

### Hypothesis H3: no single flat connection explains every composition

Mixture, exponential, and Levi-Civita transports make different predictions for composed transformations. Their relative performance depends on the semantic operation and context.

This hypothesis challenges the assumption that logit or hidden-state arithmetic provides a universal composition law.

### Hypothesis H4: Fisher parallel transport improves cross-context displacement consistency

Let an operation \(A\) produce a displacement at a base context \(p\):

\[
u_A(p)=\operatorname{Log}_p(T_Ap).
\]

After transporting \(u_A(p)\) to another base point \(q\), the transported vector should predict \(T_Aq\) more accurately than untransported Euclidean hidden or logit displacement:

\[
\widehat{T_Aq}
=\operatorname{Exp}_q
\left(P_{p\to q}u_A(p)\right).
\]

### Hypothesis H5: semantic noncommutativity is reflected in holonomy

For two transformations \(A,B\), compare the paths

\[
p_{00}\to p_{10}\to p_{11}
\]

and

\[
p_{00}\to p_{01}\to p_{11}.
\]

If the transformations commute at the symbolic level but their predictive effects do not form a flat parallelogram, the transport discrepancy provides a geometric measure of contextual interaction.

## 8. CPU-only experimental program

No GPU is required for the first stages.

Recommended initial models:

- [EleutherAI Pythia-70M](https://huggingface.co/EleutherAI/pythia-70m), because it has many saved training checkpoints;
- [Pythia project and checkpoint documentation](https://github.com/EleutherAI/pythia);
- [GPT-2 124M](https://huggingface.co/openai-community/gpt2) as a second architecture/checkpoint.

Pythia provides 154 checkpoints over training, making it possible to distinguish geometry caused by architecture and softmax from geometry that develops during pretraining.

### Experiment 0: verify the geometry independently of language models

Purpose: validate formulas and code.

1. Select several distributions in the interior of a three-category simplex.
2. Map them to the radius-2 sphere using \(x=2\sqrt p\).
3. Verify numerically that Fisher distances equal spherical distances.
4. Verify that mixture paths are straight in probability coordinates.
5. Verify that exponential paths are straight in log-odds coordinates.
6. Transport a vector around a small spherical loop and recover curvature approximately equal to \(1/4\).

This experiment proves nothing new; it prevents implementation errors.

### Experiment 1: discrete semantic quadrilaterals in output space

Construct controlled \(2\times2\) prompt families:

\[
c_{00},\quad c_{10}=A(c_{00}),\quad
c_{01}=B(c_{00}),\quad c_{11}=A(B(c_{00})).
\]

For example, \(A\) changes grammatical number and \(B\) changes tense. Extract

\[
p_{ab}=p_\theta(\cdot\mid c_{ab}).
\]

Compare transport along

\[
p_{00}\to p_{10}\to p_{11}
\]

with transport along

\[
p_{00}\to p_{01}\to p_{11}.
\]

For an initial tangent vector \(v\in T_{p_{00}}M\), define the two-path holonomy discrepancy

\[
H(v)=
P_{p_{10}\to p_{11}}P_{p_{00}\to p_{10}}v
-P_{p_{01}\to p_{11}}P_{p_{00}\to p_{01}}v.
\]

Use the normalized statistic

\[
h=\frac{\|H(v)\|}{\|v\|}.
\]

Important limitation: nonzero holonomy here partly reflects the already-known curvature of the full Fisher simplex. It does not by itself establish model-specific intrinsic curvature. Model specificity must be assessed using matched random quadrilaterals, checkpoint comparisons, or the pullback experiment below.

### Experiment 2: curvature of a continuous predictive surface

This was the original central experiment. After mathematical audit it is a secondary,
chart-dependent experiment: nonzero Gaussian curvature of the selected two-dimensional
surface does not by itself prove nonzero sectional curvature of a larger hidden manifold.
The revised central experiment computes exact sectional curvature of the full final
decoded softmax family from vocabulary cumulants; see
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

Take a fixed prompt represented by input embeddings \(z_0\). Choose two perturbation directions \(u,v\), initially random orthogonal directions and later semantic token-difference directions. Define

\[
z(a,b)=z_0+a u+b v.
\]

The frozen model gives a smooth two-dimensional predictive surface

\[
F_\theta(a,b)=p_\theta(\cdot\mid z(a,b)).
\]

Evaluate this surface on, for example, an \(11\times11\) grid. Estimate

\[
g_{ij}(a,b)
=\sum_k
\frac{\partial_i p_k\partial_jp_k}{p_k}
\]

using central finite differences or automatic differentiation. Because the metric is only \(2\times2\), Christoffel symbols and Gaussian curvature can be computed on CPU:

\[
\Gamma^k_{ij}
=\frac12g^{k\ell}
\left(
\partial_i g_{j\ell}
+\partial_jg_{i\ell}
-\partial_\ell g_{ij}
\right).
\]

Then compute the two-dimensional Riemann tensor and

\[
K(a,b)=\frac{R_{1212}(a,b)}{\det g(a,b)}.
\]

The chart must be rejected or regularized where \(\det g\) is numerically close to zero.

Run the same chart at several Pythia checkpoints, for example:

\[
\text{initial},\ 1{,}000,\ 3{,}000,\ 10{,}000,\ 50{,}000,\ 143{,}000.
\]

The main quantities are:

- signed curvature \(K\);
- absolute curvature \(|K|\);
- curvature normalized by local Fisher scale;
- holonomy per enclosed Fisher area;
- variation across prompts and directions;
- change across checkpoints.

### Experiment 3: compare the three connections

For each semantic quadrilateral, compare:

1. hidden-state Euclidean displacement;
2. logit displacement / exponential connection;
3. probability-mixture displacement / mixture connection;
4. Fisher--Rao Levi-Civita parallel transport.

Given the observed displacement from \(p_{00}\) to \(p_{10}\), predict \(p_{11}\) from \(p_{01}\). For Levi-Civita transport,

\[
\widehat p_{11}^{\mathrm{LC}}
=\operatorname{Exp}_{p_{01}}
\left(
P_{p_{00}\to p_{01}}
\operatorname{Log}_{p_{00}}p_{10}
\right).
\]

Evaluate with:

- Fisher--Rao distance to the true \(p_{11}\);
- KL divergence in both directions;
- Jensen--Shannon divergence;
- top-token ranking agreement;
- calibration change where applicable.

The substantive result would not merely be that the methods differ. It would be that one connection predicts held-out semantic compositions more consistently, or that different semantic operations reliably select different connections.

## 9. Closed-form Fisher--sphere operations

Let

\[
x=2\sqrt p,\qquad y=2\sqrt q,
\qquad \|x\|=\|y\|=R=2.
\]

Define

\[
\theta=\arccos\frac{\langle x,y\rangle}{R^2}.
\]

The Fisher--Rao distance is

\[
d_{\mathrm{FR}}(p,q)=R\theta
=2\arccos\left(\sum_i\sqrt{p_iq_i}\right).
\]

The spherical logarithmic map is

\[
\operatorname{Log}_x(y)
=\frac{\theta}{\sin\theta}
\left(y-\cos\theta\,x\right).
\]

The exponential map is

\[
\operatorname{Exp}_x(v)
=\cos\left(\frac{\|v\|}{R}\right)x
+R\sin\left(\frac{\|v\|}{R}\right)
\frac{v}{\|v\|}.
\]

Parallel transport along the minimal great-circle geodesic from \(x\) to \(y\) is

\[
P_{x\to y}(v)
=v-
\frac{\langle v,y\rangle}
{R^2+\langle x,y\rangle}
(x+y),
\]

provided \(x\) and \(y\) are not antipodal.

These formulas make the output-level experiments cheap and remove the need to estimate Christoffel symbols on the full vocabulary simplex.

## 10. Controls and failure modes

### 10.1 Ambient curvature is not learned curvature

The full categorical Fisher simplex already has curvature \(1/4\). Showing nonzero ambient holonomy is therefore a demonstration, not a model discovery. A model-specific result requires at least one of:

- curvature of a model-dependent pullback metric;
- comparison across training checkpoints;
- comparison with random initialization;
- comparison with matched random conditional distributions;
- semantic structure beyond what is explained by Fisher distances and loop area.

### 10.2 Contexts are discrete

Ordinary text prompts form a discrete set, not automatically a smooth manifold. Smooth curvature needs a chart. Continuous input embeddings or soft prompts provide such a chart, but the interpretation must be stated explicitly: the experiment studies the model's smooth extension between tokenized prompts.

### 10.3 Softmax boundary behavior

Next-token distributions may contain extremely small probabilities. Use log-softmax and float64 for geometric computations where possible. If smoothing is required, report it explicitly:

\[
p_i^{(\varepsilon)}=(1-\varepsilon)p_i+\frac{\varepsilon}{V}.
\]

Results should be tested across several \(\varepsilon\) values.

### 10.4 Degenerate pullback metric

If the selected context directions barely affect predictions, \(g\) will be nearly singular. Monitor the eigenvalues and condition number of the \(2\times2\) metric. Reject charts whose smallest eigenvalue is below a predetermined tolerance.

### 10.5 Finite-difference error can look like curvature

Repeat the calculation at multiple grid spacings. A trustworthy curvature estimate should stabilize as the step size changes. Compare finite differences with automatic derivatives on a small subset.

### 10.6 Linguistic operations may not commute

If \(A\) and \(B\) have genuine order-dependent semantics, loop discrepancy is expected even before geometry is considered. Begin with controlled synthetic grammar and factorial templates where the intended operations commute, then expand to natural language.

### 10.7 Quantization can distort derivatives

Use full-precision CPU inference for the 70M model when estimating derivatives. Quantization is unnecessary at this scale and can introduce nonsmooth or distorted local geometry.

## 11. Criteria for a meaningful result

A weak result would be:

> The probability simplex is curved and parallel transport differs from vector addition.

That is already known.

A stronger and potentially publishable result would satisfy several of the following:

1. Pullback Fisher curvature changes systematically over pretraining checkpoints.
2. The change is reproducible across prompts, seeds, or model families.
3. Semantic perturbation planes differ from matched random planes.
4. Holonomy predicts failures of Euclidean analogy or composition.
5. Levi-Civita transport improves held-out semantic transformation prediction.
6. Exponential, mixture, and Levi-Civita connections show stable task-dependent differences.
7. A curvature- or holonomy-based statistic correlates with context sensitivity, compositionality, or calibration beyond ordinary Fisher distance.

The cleanest negative result would also be valuable: if exponential/logit transport matches or outperforms Levi-Civita transport everywhere and curvature effects are negligible at semantic scales, that would support the adequacy of current flat-coordinate approximations.

## 12. Novelty boundary and related work

**Post-audit correction.** The provisional gap stated later in this section is too
broad. Park et al. (ICML 2026), FishBack, and several 2026 holonomy papers already
occupy softmax information geometry, transformer Fisher pullbacks, and generic
representation holonomy. Viswanathan and Park already use softmax-entropy cumulants
as layer- and checkpoint-level probes. Mir's July 2026 RGF report already proves
final-affine-layer flatness for a *Euclidean centered-logit* pullback, but that metric
is \(W^\top\Pi W\), not the probability-dependent Fisher pullback
\(W^\top(\operatorname{diag}p-pp^\top)W\). The current defensible gap is the exact
Fisher--Levi-Civita third-cumulant Riemann-curvature estimator, its checkpoint/seed
dynamics, and a controlled comparison of mixture, exponential, and LC semantic
transfer. See the full matrix and KL expansion in
[PROOFS_AND_NOVELTY.md](PROOFS_AND_NOVELTY.md).

The following nearby ideas already exist:

- information geometry and dual flatness of exponential/mixture families;
- Fisher--Rao geometry of probability simplices;
- invariance of Fisher geometry under sufficient statistics, discussed in [Ay, Jost, Lê, and Schwachhöfer](https://arxiv.org/abs/1207.6736);
- conditional embeddings as tangent vectors on a probability simplex in [Natural Alpha Embeddings](https://arxiv.org/abs/1912.02280);
- parallel transport for analogy in hyperbolic embeddings, including [Poincaré GloVe](https://arxiv.org/abs/1810.06546);
- parallel transport in latent generative-model geometry in [The Riemannian Geometry of Deep Generative Models](https://arxiv.org/abs/1711.08014);
- connection-based neural architectures, such as the [Gauge Equivariant Transformer](https://proceedings.neurips.cc/paper/2021/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html);
- a recent interpretation of attention as connection propagation in [*From Self-Attention to Connection Laplacian*](https://arxiv.org/abs/2607.10677);
- softmax-entropy cumulants across layers and Pythia training in [Viswanathan and Park](https://arxiv.org/abs/2510.04285);
- Euclidean centered-logit pullback geometry in Mir's [*Representation Geometry Fields*](https://www.researchgate.net/publication/408490280_Representation_Geometry_Fields_Pullback_Metrics_Induced_by_Transformer_Computation_Mathematical_Specification), whose final-layer flatness is metric-specific rather than a Fisher-flatness result.

Therefore the generic claim "embeddings should use manifolds and parallel transport" is not novel. The more specific research gap to investigate is:

> The exact Fisher--Levi-Civita Riemann curvature of trained next-token decoder families, its evolution during pretraining, its semantic holonomy, and a controlled comparison of exponential, mixture, and Levi-Civita transport as models of language composition.

As of 2026-08-04, no exact match was found. This is a moderate-confidence novelty
assessment, not proof of absence; a reviewer-level citation-network search remains
necessary before making a priority claim.

## 13. Immediate implementation order

This original list is retained as history. The current order is in
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md): exact decoder curvature first,
hard-output connection transfer second, and chart-dependent surface holonomy last.

1. Implement and test sphere distance, log, exp, and parallel transport on synthetic categorical distributions.
2. Load Pythia-70M on CPU and extract full next-token distributions.
3. Build a small controlled set of semantic \(2\times2\) prompt families.
4. Measure output-space transport and matched random controls.
5. Define a two-dimensional continuous soft-prompt chart.
6. Estimate its pullback metric and curvature at one checkpoint.
7. Repeat at selected Pythia training checkpoints.
8. Only after numerical stability is established, compare the three connections on held-out semantic transformations.

## 14. Original thesis in one sentence

> Next-token pretraining identifies a predictive conditional map but does not identify a global Euclidean hidden geometry; nonzero curvature or holonomy of the model's pullback Fisher manifold obstructs metric-compatible, context-independent vector arithmetic and motivates connection-aware semantic composition by parallel transport.

## 15. Audited thesis

> Final hidden-state addition at a linear softmax decoder is exactly the flat
> exponential-affine composition law. It is globally feasible and path independent,
> but not Fisher-metric-compatible. Fisher--Levi-Civita transport preserves predictive
> distinguishability but can be curved, path dependent, and only locally feasible.
> The empirical question is which tradeoff semantic composition follows and whether
> exact decoder curvature predicts failures of exponential vector arithmetic.
