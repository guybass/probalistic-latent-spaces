# Predictive Fisher Geometry: Audited Claims, Proofs, and Novelty Boundary

Date of literature audit: 2026-08-04
Status: working research memorandum, not a publication claim

## Executive conclusion

The original intuition contains a real distinction, but the broad claim is not novel:

- a next-token softmax family carries Fisher information geometry;
- a hidden space inherits a pullback Fisher metric;
- exponential and mixture connections are flat even when the Fisher Levi--Civita connection is curved;
- Riemannian parallel transport, semantic manifolds, and representation holonomy have all appeared in representation-learning work.

The strongest defensible research program is narrower:

> Compute the exact Fisher--Levi-Civita sectional curvature of a trained language model's final decoded softmax family from mixed predictive cumulants and inverse-metric contractions; track it through open pretraining checkpoints; and compare the flat exponential and mixture transports with metric-compatible Levi--Civita transport on controlled semantic composition tasks.

The pure mathematical identities below are classical or short corollaries of classical results. Potential novelty lies in their exact specialization to language-model decoder geometry, the curvature--torsion--nonmetricity framing, and a controlled empirical link to semantic composition. Scalar softmax cumulants and Euclidean transformer pullback metrics are already prior art; the audited gap is specifically the probability-weighted Fisher Riemann tensor and its LC holonomy. The literature search found no paper containing this exact combination, but absence cannot be proved by search.

## 1. Three different geometric objects

These must not be conflated.

### 1.1 Full categorical simplex

\[
\Delta^{V-1}_+
=\left\{p\in\mathbb R^V:p_i>0,\ \sum_i p_i=1\right\}
\]

with Fisher metric

\[
g_p(u,v)=\sum_i\frac{u_iv_i}{p_i},
\qquad \sum_i u_i=\sum_i v_i=0.
\]

Its Fisher--Levi-Civita sectional curvature is the known constant \(1/4\) when \(V\ge3\).

### 1.2 Decoded intervention manifold

For a linear language-model head,

\[
p_h(y)=\operatorname{softmax}(Wh+b)_y,
\]

define

\[
\mathcal M_{\mathrm{dec}}
=\{p_h:h\in\mathbb R^d\}.
\]

This is a continuous exponential family. Its Fisher pullback metric and its intrinsic sectional curvature depend on the learned unembedding \(W\) and the current distribution \(p_h\). It is the cleanest object for an exact CPU experiment.

Under the square-root immersion \(x(h)=2\sqrt{p_h}\), the Gauss equation for a
Fisher-orthonormal pair \(X,Y\) is

\[
K_{\mathcal M_{\mathrm{dec}}}(X,Y)
=\frac14
+\langle II(X,X),II(Y,Y)\rangle
-\|II(X,Y)\|^2.
\]

The learned family can therefore have curvature above, below, or equal to \(1/4\),
and can even be intrinsically flat through cancellation with its second fundamental
form.

If

\[
N=\{u:Wu\text{ is constant across vocabulary entries}\},
\]

then directions in \(N\) do not change the distribution. The Riemannian parameter space is therefore the effective quotient \(\mathbb R^d/N\). If the Fisher matrix is full rank, this quotient is represented directly by \(\mathbb R^d\).

### Proposition 1: the predictive quotient is a regular statistical manifold

Let \(F(h)=\operatorname{softmax}(Wh+b)\) and let \(N\) be defined above. Then

\[
F(h)=F(h')\quad\Longleftrightarrow\quad h-h'\in N.
\]

The descended map

\[
\bar F:\mathbb R^d/N\longrightarrow\Delta^{V-1}_+
\]

is a smooth embedding onto the regular exponential family
\(\mathcal M_{\mathrm{dec}}\). Its Fisher metric is positive definite on the
quotient. Before quotienting, the pullback matrix is

\[
G(h)=W^\top\left(\operatorname{diag}p_h-p_hp_h^\top\right)W
\]

and \(\ker G(h)=N\) for every finite \(h\).

#### Proof

Softmax logits give the same distribution exactly when they differ by a scalar
multiple of the all-ones vector. Hence

\[
F(h)=F(h')
\iff W(h-h')\in\operatorname{span}\{\mathbf1\}
\iff h-h'\in N.
\]

Use global log-ratio coordinates on the open simplex,

\[
\rho_i(p)=\log\frac{p_i}{p_V},\qquad i<V.
\]

Writing \(B_i^\top=w_i^\top-w_V^\top\), one obtains

\[
\rho(F(h))=Bh+\beta,
\]

where \(\ker B=N\). Thus \(B\) descends to a linear isomorphism from
\(\mathbb R^d/N\) onto \(\operatorname{im}B\), and \(\rho^{-1}\) maps the
corresponding affine subspace smoothly onto \(\mathcal M_{\mathrm{dec}}\). This
proves the embedding claim.

Finally,

\[
u^\top G(h)u
=\operatorname{Var}_{Y\sim p_h}(w_Y^\top u).
\]

Every softmax probability is positive, so this variance vanishes exactly when
\(w_y^\top u\) is constant in \(y\), namely when \(u\in N\). Therefore the
pullback is positive semidefinite on \(\mathbb R^d\) and positive definite on the
predictive quotient. \(\square\)

This is the rigorous sense in which a decoded latent space is a statistical
manifold. It does not turn the discrete natural hidden-state cloud into a smooth
manifold without an interpolation assumption.

### 1.3 Natural hidden-state set

\[
\mathcal H_{\mathrm{nat}}
=\{h(c):c\text{ is a tokenized context}\}.
\]

This is a discrete point cloud. It has no canonical tangent bundle or curvature without an additional interpolation or manifold assumption. Curvature of \(\mathcal M_{\mathrm{dec}}\) describes arbitrary final-state interventions; it does not prove that natural contexts densely occupy that whole manifold.

An explicit soft-prompt chart is a fourth object: a chosen smooth intervention surface. Its intrinsic curvature belongs to that chart and is not automatically a sectional curvature of a larger hidden manifold.

## 2. The flatness issue in the book

The book [*Geometric Modeling in Probability and Statistics*](https://link.springer.com/book/10.1007/978-3-319-07779-6) does not imply that every probabilistic manifold has zero Riemannian curvature.

A statistical manifold carries several affine connections:

\[
\nabla^{(1)}\quad\text{(exponential)},\qquad
\nabla^{(-1)}\quad\text{(mixture)},\qquad
\nabla^{(0)}\quad\text{(Fisher Levi--Civita)}.
\]

They satisfy

\[
\nabla^{(0)}
=\frac12\left(\nabla^{(1)}+\nabla^{(-1)}\right).
\]

Exponential and mixture families can be flat for their corresponding affine connections:

\[
R^{(1)}=R^{(-1)}=0.
\]

This does not imply

\[
R^{(0)}=0,
\]

because curvature is nonlinear in the connection. “Global coordinates,” “dual affine flatness,” and “zero Levi--Civita curvature” are three different statements.

## 3. Classical Fisher-sphere theorem

### Theorem 1: categorical Fisher geometry is spherical

The map

\[
\Phi(p)=2(\sqrt{p_1},\ldots,\sqrt{p_V})
\]

is a Fisher isometry from \(\Delta^{V-1}_+\) to the positive orthant of the sphere \(S^{V-1}(2)\).

#### Proof

Because \(\sum_i p_i=1\),

\[
\|\Phi(p)\|^2=4.
\]

For a tangent vector \(u\),

\[
(D\Phi_pu)_i=\frac{u_i}{\sqrt{p_i}}.
\]

Hence

\[
\langle D\Phi_pu,D\Phi_pv\rangle
=\sum_i\frac{u_iv_i}{p_i}
=g_p(u,v).
\]

A sphere of radius \(2\) has sectional curvature \(1/2^2=1/4\). Therefore, for \(V\ge3\), the categorical Fisher manifold has positive Levi--Civita curvature \(1/4\). The image is global and isometric but is not a vector subspace and is not closed under ambient addition. \(\square\)

This alone resolves the apparent contradiction in the original idea.

## 4. What cross-entropy does and does not identify

### Theorem 2: output optimum does not identify hidden Euclidean geometry

Let the true conditional be \(P(\cdot\mid c)\) and the model conditional be \(q_\theta(\cdot\mid c)\). Then

\[
\mathcal L(\theta)
=\mathbb E_c H(P(\cdot\mid c))
+\mathbb E_cD_{\mathrm{KL}}
\left(P(\cdot\mid c)\|q_\theta(\cdot\mid c)\right).
\]

Under realizability and global optimization, minimizing cross-entropy identifies the output conditional almost surely on the data support. It does not identify a unique hidden metric.

#### Proof of non-identifiability

Write

\[
q(c)=D(H(c)).
\]

For any invertible latent reparameterization \(\varphi\),

\[
H'=\varphi\circ H,
\qquad
D'=D\circ\varphi^{-1}
\]

gives exactly the same conditional map. For a linear decoder \(Wh+b\), the exact affine symmetry is

\[
h'=Ah+a,
\qquad
W'=WA^{-1},
\qquad
b'=b-WA^{-1}a.
\]

Every logit is unchanged, but a nonorthogonal \(A\) changes hidden Euclidean lengths and angles. Thus likelihood alone cannot justify a unique hidden Euclidean geometry. \(\square\)

This conclusion is about observational identifiability. Arbitrary diffeomorphisms need not preserve a restricted neural architecture, and tied embeddings or normalization can restrict exact architectural symmetries.

The predictive pullback metric handles this gauge correctly. If \(F'=F\circ\varphi^{-1}\), then

\[
\varphi^*((F')^*g_{\mathrm{FR}})=F^*g_{\mathrm{FR}}.
\]

Intrinsic curvature is preserved even though raw Euclidean coordinates change.

## 5. Exact obstruction to Fisher-compatible vector addition

### Theorem 3: isometric abelian semantic sum implies flatness

Let \((M,g)\) be a connected Riemannian manifold. Suppose a smooth operation \(\oplus\) makes \(M\) an abelian Lie group and every translation

\[
L_a(p)=a\oplus p
\]

is a Fisher isometry: \(L_a^*g=g\). Then \(g\) is flat.

#### Proof

The metric is translation invariant. Let \(X,Y,Z\) be left-invariant vector fields. The group is abelian, so all their Lie brackets vanish. The Koszul formula reduces to

\[
2g(\nabla_XY,Z)=0.
\]

Thus \(\nabla_XY=0\) for every left-invariant \(X,Y\). These fields span every tangent space, so the curvature tensor vanishes. \(\square\)

### Corollary 3.1

For \(V\ge3\), no smooth abelian group law on the categorical simplex can have Fisher-isometric translations, because its Levi--Civita curvature is \(1/4\).

Logit coordinates do define a global additive structure. The theorem says its translations cannot also preserve the Fisher metric.

### Theorem 4: local semantic-flow version

Let \(X_1,\ldots,X_m\) be vector fields on an open region that:

1. span every tangent space;
2. commute, \([X_i,X_j]=0\);
3. generate metric-preserving flows, \(\mathcal L_{X_i}g=0\).

Then \(g\) is flat on that region.

#### Proof

The commuting frame supplies local coordinates with \(X_i=\partial_i\). The Killing condition gives

\[
(\mathcal L_{\partial_k}g)_{ij}=\partial_k g_{ij}=0.
\]

The metric components are constant, so all Levi--Civita Christoffel symbols and the curvature tensor vanish. \(\square\)

Therefore, if two independent semantic flows span a regular predictive surface and its measured curvature is nonzero, at least one premise fails: the effects are not commuting, not Fisher-isometric, not context-independent, or not spanning.

Nonzero curvature does **not** rule out useful vector encodings, non-isometric addition, an isolated symmetry direction, or a higher-dimensional isometric embedding.

## 6. Curvature--torsion--nonmetricity trilemma

For an affine connection \(\nabla\), write

\[
R^\nabla=\text{curvature},\qquad
T^\nabla=\text{torsion},\qquad
Q^\nabla=\nabla g=\text{nonmetricity}.
\]

Operationally:

- curvature measures local path dependence of transport;
- torsion measures infinitesimal affine-parallelogram closure/order defect;
- nonmetricity measures failure to preserve Fisher lengths and angles.

### Theorem 5: no connection has all three Euclidean properties on a curved Fisher manifold

If the Levi--Civita curvature of \((M,g)\) is nonzero, no affine connection can satisfy simultaneously

\[
R^\nabla=0,\qquad T^\nabla=0,\qquad Q^\nabla=0.
\]

#### Proof

The conditions \(T^\nabla=0\) and \(Q^\nabla=0\) uniquely characterize the Levi--Civita connection. Hence \(\nabla=\nabla^{LC}\), contradicting \(R^\nabla=0\). \(\square\)

For the categorical Fisher simplex:

| Connection | Curvature | Torsion | Fisher nonmetricity |
|---|---:|---:|---:|
| Levi--Civita | \(K=1/4\) | zero | zero |
| Exponential | zero | zero | generally nonzero |
| Mixture | zero | zero | generally nonzero |
| A local teleparallel choice | zero | generally nonzero | zero |

The resulting research statement is precise:

> On curved predictive geometry, semantic composition cannot simultaneously have path-independent transport, torsion-free affine composition, and exact Fisher-metric preservation.

Torsion must not be identified automatically with linguistic order effects; such an identification requires a specified parallel frame and semantic flows.

### Theorem 6: no nonzero field is parallel for both flat dual connections

On the categorical simplex of dimension at least two, no nonzero tangent field on an open region is parallel under both the exponential and mixture connections.

#### Proof

If \(V\) were parallel for both, it would be parallel for their average, the Levi--Civita connection. A parallel field obeys \(R(X,Y)V=0\). On a constant-curvature manifold,

\[
R(X,Y)V
=\frac14\left(g(Y,V)X-g(X,V)Y\right).
\]

At a point where \(V\ne0\), choose \(X\perp V\) and \(Y=V\); the right side is nonzero. Contradiction. \(\square\)

Thus a nontrivial semantic effect cannot be context-independent in both probability-affine and logit-affine senses on an open region.

## 7. Exact connection formulas and the meaning of hidden addition

For \(p,q,r\in\Delta_+^{V-1}\), define the connection-dependent analogy

\[
\mathcal A^\nabla(p,q;r)
=\operatorname{Exp}^{\nabla}_r
\left(P^\nabla_{p\to r}\operatorname{Log}^{\nabla}_p q\right).
\]

### Mixture connection

\[
\operatorname{Log}^m_pq=q-p,
\qquad
P^m_{p\to r}u=u,
\qquad
\mathcal A^m(p,q;r)=r+q-p.
\]

The endpoint exists only when every component is positive.

### Exponential connection

\[
\operatorname{Log}^e_pq
=p\left(\log\frac qp-\mathbb E_p\log\frac qp\right),
\]

\[
(P^e_{p\to r}u)_i
=r_i\left(\frac{u_i}{p_i}-\sum_jr_j\frac{u_j}{p_j}\right),
\]

and

\[
\mathcal A^e_i(p,q;r)
=\frac{r_iq_i/p_i}{\sum_jr_jq_j/p_j}.
\]

This endpoint is positive for every interior \(p,q,r\).

### Fisher--Levi-Civita connection

Map \(p\) to \(x=2\sqrt p\). On the radius-two sphere,

\[
d_{FR}(p,q)=2\arccos\sum_i\sqrt{p_iq_i},
\]

\[
P_{x\to y}v
=v-\frac{\langle v,y\rangle}{4+\langle x,y\rangle}(x+y).
\]

The analogy is obtained with the spherical log, this transport, and the spherical exponential. It preserves Fisher lengths exactly, but a finite transported displacement can leave the positive square-root orthant.

### Theorem 7: hidden vector arithmetic is exactly exponential transport at a linear softmax head

For

\[
p_h(y)=\exp(w_y^\top h+b_y-A(h)),
\]

one has

\[
p_{h_r+h_q-h_p}(y)
=\frac{p_{h_r}(y)p_{h_q}(y)/p_{h_p}(y)}
{\sum_zp_{h_r}(z)p_{h_q}(z)/p_{h_p}(z)}.
\]

#### Proof

Add the log probabilities for \(h_r\) and \(h_q\), subtract that for \(h_p\), and collect the terms depending on \(y\):

\[
w_y^\top(h_r+h_q-h_p)+b_y.
\]

All log-partition terms are constants in \(y\) and disappear under normalization. \(\square\)

The title also holds at the tangent level. In the natural quotient coordinates
\([h]\in\mathbb R^d/N\), the exponential-connection coefficients vanish, so

\[
P^e_{[h_s]\to[h_t]}[v]=[v],\qquad
\operatorname{Log}^e_{[h_p]}[h_q]=[h_q-h_p],\qquad
\operatorname{Exp}^e_{[h_r]}[v]=[h_r+v].
\]

Consequently, Theorem 7 is precisely

\[
\operatorname{Exp}^e_{[h_r]}
\left(P^e_{[h_p]\to[h_r]}
\operatorname{Log}^e_{[h_p]}[h_q]\right)
=[h_r+h_q-h_p].
\]

This is a central correction to the original thesis. Hidden addition is not geometrically meaningless; at the final linear head it is exactly the flat exponential-affine composition law. The empirical question is whether language semantics prefers that law or the metric-compatible Fisher law.

This identity is exact for the representation immediately entering a fixed affine
softmax head. At an earlier transformer layer, the remaining network is nonlinear;
raw addition there is not generally exponential-parallel under the pulled-back
predictive connection.

### Corollary 7.1: the alpha family gives a precise correction to addition

Let

\[
\psi(h)=\log\sum_y\exp(b_y+w_y^\top h),\qquad
G=\nabla^2\psi,\qquad C=\nabla^3\psi.
\]

On the nondegenerate quotient, the Christoffel operator of the Amari
\(\alpha\)-connection in natural coordinates is

\[
\Gamma^{(\alpha)}(u,v)
=\frac{1-\alpha}{2}G^{-1}C(u,v,\cdot).
\]

Therefore a vector \(v\) transported along \(h(t)=h+t\varepsilon a\) satisfies

\[
P^{(\alpha)}_{h\to h+\varepsilon a}v
=v-\varepsilon\frac{1-\alpha}{2}
G(h)^{-1}C_h(a,v,\cdot)+O(\varepsilon^2).
\]

#### Proof

In natural coordinates, the exponential coefficients vanish and the
Levi--Civita coefficients with the final index lowered are
\(\Gamma^{LC}_{ij,k}=C_{ijk}/2\). Under the convention

\[
g(\nabla^{(\alpha)}_XY,Z)
=g(\nabla^{LC}_XY,Z)-\frac\alpha2C(X,Y,Z),
\]

raising the final index gives the displayed Christoffel operator. Parallel
transport obeys

\[
\dot v(t)+\Gamma^{(\alpha)}(\dot h(t),v(t))=0.
\]

A first-order expansion at \(t=0\) gives the result. \(\square\)

Thus

\[
\begin{array}{c|c|c}
\alpha & \text{connection} & \text{first-order correction to a fixed vector}\\
\hline
1 & e & 0\\
0 & LC & -\tfrac12G^{-1}C(a,v,\cdot)\\
-1 & m & -G^{-1}C(a,v,\cdot)
\end{array}
\]

A fitted \(\alpha\) estimates the scalar correction coefficient inside this
canonical family; it is not an unrestricted learned connection. This is a
connection/Christoffel correction. Curvature is detected only by noncommuting
covariant derivatives or transport around paths/loops.

## 8. Exact price paid by the flat connections

### Theorem 8: exponential and mixture transports are Fisher-dual

For tangent vectors \(u,v\in T_p\Delta_+\),

\[
g_r(P^m_{p\to r}u,P^e_{p\to r}v)=g_p(u,v).
\]

#### Proof

Let \(s_i=v_i/p_i\). Since \(\sum_i u_i=0\),

\[
g_r\left(u,r(s-\mathbb E_rs)\right)
=\sum_i u_i(s_i-\mathbb E_rs)
=\sum_i u_is_i
=g_p(u,v).
\]

\(\square\)

They preserve the cross-pairing, not their own Fisher lengths.

The formulas in Section 7 are the **ambient full-simplex** connections. For a
lower-dimensional decoded exponential family, the ambient exponential connection
restricts to the family, but ambient mixture or LC paths generally do not. The
intrinsic mixture connection uses expectation coordinates
\(\eta=\nabla\psi(h)=\mathbb E_pW_Y\), and intrinsic LC transport uses the decoder
metric \(G(h)\) and cubic tensor \(C(h)\). Ambient and intrinsic experiments must be
reported separately.

With the convention

\[
g(\nabla^{(\alpha)}_XY,Z)
=g(\nabla^{LC}_XY,Z)-\frac\alpha2C(X,Y,Z),
\]

where \(C\) is the Amari--Chentsov cubic tensor,

\[
\nabla^{(\alpha)}g=\alpha C.
\]

If \(U,V\) are \(\alpha\)-parallel along \(\gamma\), then

\[
\frac d{dt}g(U,V)
=\alpha C(\dot\gamma,U,V).
\]

Thus the integrated cubic tensor measures the exact Fisher-length/angle distortion paid for flatness. For the categorical simplex, \(C\) is nonzero; for example, in mixture coordinates and distinct \(a,b\),

\[
C_{aab}=-\frac1{p_V^2}.
\]

### Corollary 8.1: exact Fisher defect of hidden translation

On the decoded exponential family in natural hidden coordinates, let

\[
\tau_a(h)=h+a.
\]

Its differential leaves coordinate vectors unchanged, but

\[
(\tau_a^*G)_h=G(h+a).
\]

Because \(G=\nabla^2\psi\) and \(C=\nabla^3\psi\),

\[
\left.\frac d{dt}\right|_{t=0}(\tau_{tu}^*G)_h(v,w)
=C_h(u,v,w).
\]

The finite squared-length defect is exactly

\[
\|v\|_{G(h+a)}^2-\|v\|_{G(h)}^2
=\int_0^1 C_{h+ta}(a,v,v)\,dt.
\]

Thus ordinary hidden translation is exponential-affine but is a local Fisher
isometry in direction \(u\) only where \(C(u,\cdot,\cdot)=0\). This is the
decoder-specific quantitative form of nonmetricity; it is stronger than merely
saying that “vector addition ignores curvature.”

## 9. Exact curvature of a trained linear softmax decoder

Let

\[
\psi(h)=\log\sum_y\exp(b_y+w_y^\top h).
\]

Let the random vector \(W_Y\) take value \(w_y\) with probability \(p_h(y)\), and write

\[
\mu=\mathbb E_pW_Y,
\qquad z_y=w_y-\mu.
\]

### Metric choice is not neutral: centered logits versus predictive distributions

A recent technical report, Mir's [*Representation Geometry Fields*](https://www.researchgate.net/publication/408490280_Representation_Geometry_Fields_Pullback_Metrics_Induced_by_Transformer_Computation_Mathematical_Specification),
uses the Euclidean metric on logits modulo uniform shifts. If

\[
\Pi=I-\frac1V\mathbf 1\mathbf 1^\top,
\]

then the pullback through the affine unembedding is

\[
G_{\mathrm{logit}}=W^\top\Pi W.
\]

This matrix is constant, so its nondegenerate visible subspace is intrinsically
flat. That result is correct for that codomain metric. It does **not** imply that
the Fisher pullback is flat. The Fisher metric is instead

\[
G_{\mathrm F}(h)
=W^\top\left(\operatorname{diag}p_h-p_hp_h^\top\right)W,
\]

which normally varies with \(h\). The two metrics coincide only up to a constant
factor at the uniform distribution, where

\[
\operatorname{diag}p-pp^\top=\frac1V\Pi.
\]

The Fisher choice is singled out when distance means local predictive
distinguishability. For a perturbation \(\delta h\),

\[
\begin{aligned}
D_{\mathrm{KL}}(p_h\|p_{h+\delta h})
&=\psi(h+\delta h)-\psi(h)-\nabla\psi(h)^\top\delta h\\
&=\frac12\delta h^\top G_{\mathrm F}(h)\delta h
+O(\|\delta h\|^3).
\end{aligned}
\]

For every observed target token \(y\), the affine-head negative log-likelihood
also satisfies the exact identity

\[
\nabla_h^2[-\log p_h(y)]=G_{\mathrm F}(h),
\]

because its \(y\)-dependent logit term is affine in \(h\). Equivalently, the KL
expansion is the second-order excess cross-entropy when \(p_h\) is the reference
conditional distribution. These facts identify the local loss/prediction
quadratic form; by Theorem 2 they still do not identify a unique Euclidean metric
on hidden coordinates. Thus the two pullback constructions answer different
questions:

- \(G_{\mathrm{logit}}\) measures squared change in centered logits;
- \(G_{\mathrm F}\) measures second-order KL/cross-entropy change in predicted
  probabilities.

The apparent contradiction "the final affine head is flat" versus "the decoded
family can be curved" is therefore a metric-choice distinction. If intervention
coordinates are taken before a final LayerNorm rather than at the input to the
affine head, its Jacobian must also be included in either pullback.

### Theorem 9: metric and cubic tensor are vocabulary cumulants

\[
G(h)=\nabla^2\psi(h)
=\sum_yp_yz_yz_y^\top
=W^\top(\operatorname{diag}p-pp^\top)W,
\]

and

\[
C_{ijk}(h)=\partial_i\partial_j\partial_k\psi(h)
=\sum_yp_yz_{y,i}z_{y,j}z_{y,k}.
\]

The lowered Levi--Civita Christoffel tensor in natural coordinates is

\[
\Gamma_{ijk}^{LC}=\frac12C_{ijk}.
\]

#### Proof

Derivatives of a finite exponential-family log-partition function are cumulants of its sufficient statistic. Directly, \(\partial_i p_y=p_yz_{y,i}\); differentiating the mean gives the covariance, and differentiating the covariance gives the third central moment. The Hessian metric identity gives \(\Gamma_{ijk}=C_{ijk}/2\). \(\square\)

### Exact relation to prior scalar softmax-cumulant probes

For a hidden direction \(u\), let the scalar random variable

\[
S_u(Y)=(Wu)_Y=u^\top w_Y,\qquad Y\sim p_h.
\]

Then

\[
\kappa_2(S_u)=G_h(u,u),
\qquad
\kappa_3(S_u)=C_h(u,u,u).
\]

Viswanathan and Park's entropy expansion therefore uses the same broad kind of
softmax cumulant primitive. Their observables are scalar self-contractions of
token-to-barycenter logit displacements. The curvature calculation below requires
mixed contractions \(C(u,v,\cdot)\), raising an index with \(G^{-1}\), and combining
several contractions into a coordinate-invariant Riemann tensor. Moreover, a
prompt-level output-logit displacement need not lie in the tangent image of \(W\)
for a lower-dimensional decoder family. The novelty claim is consequently about
the mixed-tensor Riemannian construction, not about introducing cumulants to LLM
analysis.

### Theorem 10: sectional curvature from three contracted third cumulants

For directions \(u,v\), define covectors

\[
c_{uv}=C(u,v,\cdot),\quad
c_{uu}=C(u,u,\cdot),\quad
c_{vv}=C(v,v,\cdot).
\]

Under the sign convention calibrated so the full categorical simplex has positive curvature \(1/4\),

\[
K_h(u,v)
=\frac{
\tfrac14\left[
c_{uv}^\top G^{-1}c_{uv}
-c_{uu}^\top G^{-1}c_{vv}
\right]
}{
(u^\top Gu)(v^\top Gv)-(u^\top Gv)^2
}.
\]

#### Proof

For a Hessian metric \(G_{ij}=\psi_{ij}\), fourth derivatives cancel from the Riemann tensor. The classical formula in [Totaro, *The Curvature of a Hessian Metric*](https://arxiv.org/abs/math/0401381) is, in his index convention,

\[
R_{ijkl}
=-\frac14G^{ab}
\left(C_{jla}C_{ikb}-C_{ila}C_{jkb}\right).
\]

Contracting with the two-plane \(u,v\) gives the displayed numerator. \(\square\)

### Corollary 10.1: curvature is a commutator of cubic-tensor operators

Define the \(G\)-self-adjoint endomorphism \(A_u\) by

\[
G(A_uv,w)=C(u,v,w).
\]

With the same curvature convention,

\[
R(u,v)=-\frac14[A_u,A_v].
\]

Consequently, the decoded Fisher metric is flat on a region if and only if

\[
[A_u,A_v]=0
\]

for every pair of tangent directions \(u,v\) throughout that region. In
particular, a nonzero third cumulant does **not** by itself prove curvature; the
relevant obstruction is the noncommutativity of the induced cubic-tensor
operators.

#### Proof

Raise one index in the Hessian-metric curvature formula of Theorem 10. The two
quadratic cubic-tensor terms become the two operator products in the commutator.
The sign is fixed by the saturated-simplex calibration below. \(\square\)

### Corollary 10.2: linear and curved are compatible statements

Suppose the effective decoded family is regular on a neighborhood and
\(K_h(u,v)\ne0\) for some plane. Then:

1. hidden arithmetic still composes predictions exactly according to the flat
   exponential law of Theorem 7;
2. the Fisher--Levi-Civita connection is not flat on that neighborhood;
3. no commuting, spanning family of hidden translations can simultaneously act
   as Fisher isometries there.

#### Proof

Item 1 is Theorem 7. Item 2 is the definition of nonzero sectional curvature.
Item 3 follows from Theorem 4, or equivalently from Corollary 8.1 together with
the commutator identity in Corollary 10.1. \(\square\)

This resolves the original apparent contradiction: a representation can be
globally additive for one affine connection and intrinsically curved for the
metric-compatible Levi--Civita connection.

Computationally,

\[
c_{uv}=\sum_yp_y(z_y^\top u)(z_y^\top v)z_y.
\]

One metric construction and three linear solves are sufficient. No finite differences, manifold fitting, Christoffel differentiation, or GPU is required.

### Saturated-simplex validation

For logits \((\theta_1,\theta_2,0)\), take

\[
w_1=(1,0),\quad w_2=(0,1),\quad w_3=(0,0).
\]

The formula reduces algebraically to

\[
c_{12}^\top G^{-1}c_{12}
-c_{11}^\top G^{-1}c_{22}
=\det G,
\]

so \(K=1/4\). The implementation recovers this to numerical precision, including probabilities as close to the boundary as \((1-2\varepsilon,\varepsilon,\varepsilon)\) with \(\varepsilon=10^{-9}\).

The curvature of a lower-dimensional decoded family need not equal \(1/4\); it may be positive, zero, or negative.

## 10. Holonomy and a quantitative semantic lower bound

For a small rectangular loop with sides \(\varepsilon X\) and \(\delta Y\),

\[
P_{\partial\Box}
=I\pm\varepsilon\delta R(X,Y)
+O\!\left(\varepsilon\delta(\varepsilon+\delta)\right).
\]

Curvature zero implies local path independence. Global path independence additionally requires trivial global holonomy; simple connectedness is a standard sufficient condition for a flat connection.

### Proposition 11: holonomy lower-bounds every path-independent prediction

Let two paths on an oriented two-dimensional predictive surface connect \(p\) to \(q\) and enclose a contractible region \(D\). Their relative Levi--Civita transport is a rotation through

\[
\Omega=\int_DK\,dA\pmod{2\pi}.
\]

For \(v\in T_pM\),

\[
\|P_1v-P_2v\|
=2\|v\|\left|\sin\frac\Omega2\right|.
\]

For any single path-independent prediction \(w\in T_qM\), the triangle inequality gives

\[
\max\{\|w-P_1v\|,\|w-P_2v\|\}
\ge
\|v\|\left|\sin\frac\Omega2\right|.
\]

For a small loop this is approximately

\[
\frac{\|v\|}{2}|K|\operatorname{Area}(D).
\]

This is a quantitative obstruction, not merely the statement “the space is curved.” Its ingredients are classical; applying the bound to controlled semantic composition is a project-level proposition.

Ambient Fisher transport around output distributions measures the known sphere curvature. It becomes model-specific only through the chosen vertices. Intrinsic model curvature requires transport inside \(\mathcal M_{\mathrm{dec}}\) or another explicitly specified predictive surface.

## 11. Closest prior art and what remains

| Topic already occupied | Closest source | Remaining distinction |
|---|---|---|
| Information geometry of softmax representations; linear-representation critique; \(e/m\) interpolation and steering | Park et al., [*The Information Geometry of Softmax: Probing and Steering*](https://arxiv.org/abs/2602.15293), ICML 2026 | It does not study Fisher--LC curvature, parallel transport, or holonomy. |
| Euclidean quotient-logit pullback geometry and final-affine-layer flatness | Mir, [*Representation Geometry Fields*](https://www.researchgate.net/publication/408490280_Representation_Geometry_Fields_Pullback_Metrics_Induced_by_Transformer_Computation_Mathematical_Specification), July 2026 technical report; [code](https://github.com/sirraya-labs/Representation-Geometry-Fields) | It pulls back a fixed Euclidean metric on centered logits, \(W^\top\Pi W\), not the probability-dependent Fisher metric. It explicitly leaves full curvature computation for future work. This is a non-peer-reviewed technical report. |
| Pullback Fisher metric at transformer intermediate activations | Wang and Zhao, [*FishBack*](https://arxiv.org/abs/2605.17231) | It studies local optimal steering and metric spectra, not the Riemann tensor or LC holonomy. |
| LM-head Fisher pullback and predictive geometry from NLL | [*Next-Token Prediction as Implicit Spectral Contrastive Learning*](https://openreview.net/pdf?id=xoeGFVZCKS) | No curvature or parallel transport analysis. |
| Softmax-entropy cumulants as layer- and checkpoint-level LLM probes | Viswanathan and Park, [*Probing Geometry of Next Token Prediction Using Cumulant Expansion of the Softmax Entropy*](https://arxiv.org/abs/2510.04285), HiLD 2025 poster | It expands entropy/KL using scalar logit-deviation cumulants and tracks them in GPT-2/Pythia. It does not form the unembedding third-cumulant tensor, the Riemann tensor, semantic sectional curvature, or LC transport. We cannot claim the first use of cumulants or checkpoint cumulants in LLM geometry. |
| Fisher semantic manifolds, Christoffels, geodesics, and curvature language in LLMs | Mabrok, [*Latent Semantic Manifolds in Large Language Models*](https://arxiv.org/abs/2603.22301) | Its empirical curvature uses local-PCA/extrinsic proxies, not the exact intrinsic Fisher Riemann tensor. |
| Fisher-weighted local charts and holonomy in LLM interpretation | Javidnia, [*A Gauge Theory of Superposition*](https://arxiv.org/abs/2603.00824) | Its chart transports use fitted maps/Procrustes machinery, not Fisher--LC transport. |
| Representation holonomy and its evolution in training | Sevetlidis and Pavlidis, [*Gauge-invariant representation holonomy*](https://arxiv.org/abs/2601.21653) | It uses whitening and local Procrustes transport, primarily in vision models. |
| An LLM-specific connection, parallel transport, curvature, and holonomy | Modenbach, [*A geometric relation of the error introduced by sampling...*](https://arxiv.org/abs/2605.04899) | Its connection is a bespoke \(\mathfrak{so}(n)\)-valued construction, not Fisher--LC. |
| Holonomy in Gemma feature planes | Richards, [*Do Active SAE Feature Planes Carry More Holonomy?*](https://arxiv.org/abs/2607.20522) | It explicitly uses restricted-Jacobian transport rather than differentiating the metric or constructing the LC connection. |
| Pulling Fisher geometry through probabilistic decoders | Arvanitidis et al., [*Pulling Back Information Geometry*](https://proceedings.mlr.press/v151/arvanitidis22b.html), AISTATS 2022 | General precedent; not next-token checkpoint dynamics or connection selection. |
| Probability-simplex embeddings and \(\alpha\)-geometry | Volpi and Malagò, [*Natural Alpha Embeddings*](https://arxiv.org/abs/1912.02280) | Static conditional embeddings, not contextual autoregressive transfer. |
| Parallel-transport analogies | Tifrea et al., [*Poincaré GloVe*](https://arxiv.org/abs/1810.06546) | Hyperbolic word embeddings, not predictive Fisher geometry. |
| Hidden representation identifiability | Roeder et al., [*On Linear Identifiability of Learned Representations*](https://proceedings.mlr.press/v139/roeder21a.html) | Establishes why the non-identifiability observation is not new. |
| Failure of a naive token-manifold assumption | Robinson et al., [*Token embeddings violate the manifold hypothesis*](https://arxiv.org/abs/2504.01002) | Motivates defining a continuous decoder family rather than assuming a smooth natural-state cloud. |

## 12. Novelty verdict

### Claims that must not be made

- “First probabilistic representation of words or LLMs.”
- “First Fisher geometry of hidden states.”
- “First transformer pullback metric, softmax-gauge quotient, or visible subspace.”
- “First use of softmax cumulants or first checkpoint study of predictive cumulants.”
- “First parallel transport for analogies.”
- “First holonomy in an LLM.”
- “First curvature analysis across Pythia training.”
- “Curvature proves linear representations do not exist.”
- “Cross-entropy uniquely learns a hidden metric.”

### Plausibly differentiated contribution

1. Specialize the classical Hessian-metric curvature identity to an exact third-cumulant **Riemann sectional-curvature** estimator for a trained autoregressive softmax decoder. The qualifier is essential: cumulant probes of LLM logits already exist.
2. Measure Fisher--LC sectional curvature in controlled semantic planes across open pretraining checkpoints and seeds.
3. Compare three connection hypotheses on the same held-out semantic transfer task.
4. Measure the exact cost each connection pays: LC path dependence, \(e/m\) Fisher nonmetricity, and finite-domain failures.
5. Test whether curvature or integrated holonomy predicts composition failure after controlling for entropy, published softmax-entropy cumulant probes, Fisher edge lengths, plane conditioning, and random planes.

As of 2026-08-04, the search found no work containing this exact combination.
Confidence that it is unoccupied is moderate, not absolute: a literature search
cannot prove absence, and a reviewer-level citation-network search is still required
before submission.

The safest contribution sentence is:

> Prior work has separately tracked scalar softmax-logit cumulants during pretraining and constructed Euclidean pullback metrics on softmax-invariant logit quotients. This project identifies the mixed predictive cumulant tensors of the probability-weighted Fisher pullback, contracts them into intrinsic Levi--Civita curvature, validates the corresponding small-loop holonomy, and tests these coordinate-invariant quantities across autoregressive pretraining checkpoints and controlled semantic composition tasks.

## 13. CPU pilot result, not yet a scientific claim

The implementation was run on Pythia-14M at four checkpoints, using three hand-written factorial prompt families and eight random Fisher-orthonormal planes per base context.

The exact linear solves had maximum relative residual \(1.46\times10^{-16}\). At initialization, semantic sectional curvatures were approximately \(0.0010\) to \(0.0014\), while the metric condition number was about \(1.27\). By the final checkpoint, condition numbers were approximately \(3.1\times10^3\) to \(4.2\times10^3\), semantic curvatures ranged from \(-0.0445\) to \(0.0211\), and matched random-plane means were negative.

This proves only that the estimator is numerically usable and returns model-dependent values distinct from the ambient simplex's automatic \(1/4\). Three prompts, one seed, and a 14M model cannot support a training or semantic conclusion.

The hard-prompt output analogy also exposed a domain issue: in all 24 tested orientations, exponential composition was feasible, while unscaled mixture and ambient Fisher--LC analogies left their valid domains. This is consistent with Theorem 7 and motivates a local or fixed-coarse-graining design for the full comparison.

The transport-only diagnostic behaved as theory requires: LC log-length distortion
was below \(1.3\times10^{-16}\). At the final checkpoint, mean absolute log-length
distortion was \(0.0877\) for exponential transport and \(0.4389\) for mixture
transport. These pilot magnitudes are not population estimates.

## 14. Current thesis

> Final hidden-state addition at a linear softmax decoder is exactly the flat exponential-affine composition law. That law is globally feasible and path independent, but it is not Fisher-metric-compatible. Levi--Civita transport preserves predictive distinguishability but can be curved, path dependent, and only locally feasible. The research question is which tradeoff actual semantic composition follows, and whether the exact decoder curvature predicts when exponential vector arithmetic fails.
