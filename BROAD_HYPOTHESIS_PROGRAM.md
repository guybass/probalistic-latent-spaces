# From the MSc test back to the broad hypothesis

**Date:** 5 August 2026
**Purpose:** explain precisely how the connection-selection experiment supports, falsifies, or leaves unresolved the original hypothesis

## 1. The original hypothesis needs one correction

The motivating intuition was:

> Language-model latent representations are fundamentally probabilistic rather than linear; vector addition is not intrinsically valid, and semantic transformations should be compared by parallel transport.

Taken literally, this is too strong.

1. Transformer hidden states are elements of an ambient vector space $\mathbb R^d$ by construction.
2. At a linear head, $h\mapsto Wh+b$ is affine, and $h\mapsto\operatorname{softmax}(Wh+b)$ maps every finite $h$ to a valid positive probability distribution.
3. Therefore $h+u$, subtraction, and interpolation are mathematically defined. A translated state may not be produced naturally by a text prompt, but it remains a valid decoder intervention.
4. Hidden addition is exactly affine for the exponential connection of the decoder's exponential family.

So the project should **not** claim that linear representations do not exist or that vector addition is meaningless.

The corrected broad hypothesis is:

> **Predictive-geometry hypothesis.** Although model states have an ambient linear structure, a single context-independent displacement with a fixed Euclidean metric is not generally sufficient to represent semantic transformations while preserving local predictive distinguishability. The next-token distribution induces a Fisher geometry; in regions of nonzero Fisher--Levi-Civita curvature, no global path-independent vector addition can also act by Fisher isometries. Actual semantic transformations should therefore be tested as local vector fields whose invariance depends on a connection.

This version is both stronger mathematically and genuinely falsifiable.

## 2. “Probabilistic” does not replace “linear”

The correct structure is layered:

\[
\text{ambient state }h\in\mathbb R^d
\xrightarrow{\quad Wh+b\quad}
\text{logits}
\xrightarrow{\quad\mathrm{softmax}\quad}
p_h(\text{next token})\in\Delta_+^{V-1}.
\]

The vector space and statistical manifold coexist.

- The vector space supplies coordinates, addition, neural computation, and the flat exponential connection.
- The distribution map supplies an operational notion of distinguishability through KL divergence.
- The Hessian of local KL gives the pullback Fisher metric

\[
G(h)=W^\top\bigl(\operatorname{diag}p_h-p_hp_h^\top\bigr)W.
\]

- The metric varies with $h$ even though the decoder is affine.
- The Levi--Civita connection of this metric can be curved, while the exponential and mixture connections remain flat.

Thus the research question is not “linear space or manifold?” It is:

> **Which geometric structure correctly expresses sameness of a semantic transformation across contexts?**

### Audit of the original reasoning

| Original intuition | Verdict | Precise correction |
|---|---|---|
| Conditional token predictions form a probabilistic manifold. | Correct after specifying a smooth family. | Use the affine-softmax decoder family or an explicitly defined pullback family, not the discrete cloud of naturally occurring prompts by itself. |
| A global representation implies zero curvature. | False in this form. | A sphere has a global faithful embedding in Euclidean space and remains intrinsically curved. Flatness is a property of a chosen connection, not of mere global representability. |
| A faithful embedding that preserves an information metric must be flat. | False. | Isometric embeddings preserve intrinsic curvature; they do not eliminate it. The Fisher simplex embeds by square roots into a sphere of radius two and has LC curvature $1/4$. |
| Next-token cross-entropy learns a “conditional Shannon metric.” | Needs correction. | Cross-entropy/KL induces the Fisher metric as its local second-order form. Training learns the conditional map and decoder parameters; it does not force ambient Euclidean distance to equal Fisher distance. |
| Hidden vectors are not sumable because probabilities are not a vector space. | False at the affine head. | Hidden/logit addition is globally defined and corresponds to the $e$-affine law. Probability-coordinate addition is a different operation and generally invalid without renormalization. |
| Parallel transport is therefore required. | Empirical, not automatic. | LC transport is required only if Fisher-metric compatibility is the semantic invariant. Constant $e$-components or mixture-covector invariance may fit better. |

This audit does not destroy the hypothesis. It removes the ontological overclaim and leaves a sharper operational claim that the experiments can actually reject.

## 3. The un-narrowed framework: connection-relative linearity

The broader conceptual contribution is to separate four objects that are often conflated.

### 3.1 Support geometry

Let $\mathcal S$ denote states actually produced by natural prompts. It may be a sparse cloud, a stratified set, or a smooth manifold locally; none should be assumed without evidence.

At the final affine head, define the larger **intervention manifold**

\[
\mathcal I=\mathbb R^d/N,
\qquad
N=\{u:Wu\in\operatorname{span}(\mathbf 1)\}.
\]

Every point of $\mathcal I$ defines a decoded distribution even if it is not naturally produced by text. The exact geometry in this project is primarily the geometry of $\mathcal I$, sampled at points from $\mathcal S$.

### 3.2 Predictive metric

The decoder map $F:\mathcal I\to\Delta_+^{V-1}$ induces

\[
g=F^*g_{Fisher}.
\]

This metric says when two infinitesimal interventions are equally distinguishable by their next-token distributions. Other metrics answer other questions; the Fisher choice is operational, not ontological.

### 3.3 Transport connection

A metric alone does not specify which semantic vector at one context is “the same” vector at another. That comparison needs a connection. Exponential, LC, mixture, and fitted-alpha transports preserve different quantities.

Importantly, “ordinary vector reuse” is not transport-free. It is exactly parallel transport under the ambient flat connection and, at the affine softmax head, under the exponential connection.

### 3.4 Semantic transformation field

For operation $T$, define a sampled section of the tangent bundle,

\[
X_T(h)=h(Tc)-h(c),
\qquad h=h(c).
\]

This leads to a connection-relative definition of linear representation.

> **Definition (connection-relative linearity).** A semantic transformation $T$ is $\nabla$-linear on a region if its field is covariantly constant there:
>
> \[
> \nabla X_T=0.
> \]

Approximate linearity is measured by a normalized energy such as

\[
\mathcal E_\nabla(T)
=
\frac{
\int\|\nabla X_T\|_g^2\,d\rho
}{
\int\|X_T\|_g^2\,d\rho
},
\]

or its local graph estimator. A constant hidden vector is $e$-linear. An LC-parallel field is metric-linearly transported. A mixture-parallel field is dual-linear.

This reframes the original binary question:

\[
\text{“Are representations linear?”}
\quad\longrightarrow\quad
\text{“Under which connection, on which support, and for which operation is }\nabla X_T\approx0\text{?”}
\]

The likely novel umbrella is the combined program:

> Separate support, predictive metric, and transport connection; represent a semantic operation as a tangent-bundle section; define global linearity by covariant constancy; and test its curvature/holonomy obstruction and steering consequences.

The individual ingredients have substantial prior art. Priority should be claimed only for this exact operational synthesis and its connection-selection experiment, with “to our knowledge” language.

A suitable umbrella title is:

> **Metric, Support, and Connection: A Theory of Connection-Relative Linearity in Language Models**

The current literature-audit confidence that this complete decomposition and its joint steering test are unoccupied is approximately 0.75. The narrower alpha-selection experiment remains the higher-confidence novelty claim.

## 4. What the mathematics already proves

These statements do not require an LM experiment.

### Theorem-level fact A: exact predictive Fisher geometry

For the affine softmax family

\[
p_h(y)=\exp\left(w_y^\top h+b_y-\psi(h)\right),
\]

the Fisher metric and cubic tensor are

\[
G=\nabla^2\psi,
\qquad
C=\nabla^3\psi.
\]

This proves that the decoder family has a probability-dependent intrinsic metric. It does **not** prove that semantic behavior follows its Levi--Civita connection.

### Theorem-level fact B: flatness depends on the connection

In natural hidden coordinates,

\[
\Gamma^{(\alpha)}(a,v)
=
\frac{1-\alpha}{2}G^{-1}C(a,v,\cdot).
\]

The exponential connection ($\alpha=1$) and mixture connection ($\alpha=-1$) are flat. Levi--Civita ($\alpha=0$) preserves Fisher lengths and angles but is generally curved. Therefore “the family is dually flat” does not mean “the Fisher Riemannian manifold has zero curvature.”

For the standard alpha family, the complete tradeoff is

\[
T^{(\alpha)}=0,
\qquad
\nabla^{(\alpha)}g=\alpha C,
\qquad
R^{(\alpha)}=(1-\alpha^2)R^{LC}.
\]

| Connection | Curvature | Fisher nonmetricity | Quantity kept constant under transport |
|---|---|---|---|
| Exponential, $\alpha=1$ | zero | $+C$ | natural-coordinate vector components |
| Levi--Civita, $\alpha=0$ | generally nonzero | zero | Fisher inner products, lengths, and angles |
| Mixture, $\alpha=-1$ | zero | $-C$ | metric-lowered/expectation-coordinate covectors |

There is no geometry-only reason to select LC. It is uniquely metric compatible, but the empirical semantic invariant may instead be a natural vector or a dual covector.

### Theorem-level fact C: curvature obstructs metric-compatible global addition

If a smooth abelian addition law acts transitively by Fisher isometries, the Fisher metric is flat. Consequently, nonzero Levi--Civita curvature rules out a global addition law that is simultaneously:

1. context independent;
2. path independent;
3. Fisher-metric preserving.

Logit/hidden addition remains valid; it pays for flatness through Fisher nonmetricity.

### Theorem-level fact D: local field residual measures connection-parallelness

For a sampled semantic field $V$ and a short edge $h\to h+\varepsilon a$,

\[
V(h+\varepsilon a)-P^{(\alpha)}V(h)
=
\varepsilon\nabla^{(\alpha)}_aV(h)+O(\varepsilon^2).
\]

This proves that held-out transport residual is a consistent local test of the claim that a semantic transformation is parallel under the chosen connection.

### Theorem-level fact E: an interior alpha cannot usually be global

On a dually flat family,

\[
R^{(\alpha)}=(1-\alpha^2)R^{LC}.
\]

If the curvature operators have no common null direction, a nonzero field cannot be globally alpha-parallel for $|\alpha|<1$. Therefore an LC or intermediate-alpha semantic law should be claimed locally and with a declared path, not as a universal global vector.

## 5. What remains empirical

The theorems do not identify the semantic connection. The following claims can be supported or falsified only with model data:

1. semantic transformations form reproducible fields rather than noise;
2. a field is closer to parallel under $e$, LC, $m$, or a fitted alpha;
3. transported vectors predict held-out transformations;
4. using transport as an intervention improves target behavior;
5. curvature or holonomy predicts failures of ordinary vector composition;
6. these structures develop during pretraining rather than being present at random initialization;
7. conclusions replicate across prompts, seeds, checkpoints, and model sizes.

This is where the MSc connection-selection experiment sits: it tests the central bridge between exact predictive geometry and semantic behavior.

## 6. How the selected experiment proves or disproves the hypothesis

For transformation $T$, observe

\[
V_T(h_i)=h(Tc_i)-h(c_i)
\]

at many base contexts. Fit the connection parameter on training contexts and evaluate transport on held-out contexts.

The experiment cannot prove a universal ontology of all latent spaces. It can decisively test a scoped operational statement:

> For this model family, representation location, and transformation class, is a semantic effect better modeled as a constant global vector or as a connection-transported local tangent vector?

### Outcome table

| Held-out result | What it supports | What it falsifies or weakens |
|---|---|---|
| Exponential transport wins; alpha-hat is near $1$ | Ordinary natural-coordinate vector addition is the best local semantic law for the tested field. | The strong claim that parallel transport beyond constant hidden components is necessary. It does not remove the Fisher metric; it shows semantics chose the flat connection. |
| LC wins; alpha-hat is near $0$ | Context variation is quantitatively explained by Fisher-metric-compatible transport. Constant hidden components are not the correct invariant. | The global constant-vector hypothesis for the tested operation. |
| Mixture wins; alpha-hat is near $-1$ | The invariant is closer to a fixed expectation-coordinate covector than a fixed hidden vector. Probabilistic dual geometry matters. | The LC-specific claim. “Use parallel transport” survives, but “LC must be the transport” fails. |
| A stable interior alpha wins | Semantic invariance follows a local compromise between affine flatness and metric compatibility. Curvature predicts unavoidable path dependence. | Both a universal $e$ law and a universal LC law. |
| Fitted alpha varies by operation | Different semantic transformations preserve different invariants. There is no single universal semantic connection. | Any one-connection theory of all semantics. |
| No canonical or fitted alpha generalizes, while a learned vector field does | Context dependence is real, but predictive Fisher connection terms do not explain it. | The claim that decoder Fisher geometry is sufficient to explain semantic heterogeneity. |
| All methods are indistinguishable because $C$ and edge effects are tiny | The experiment is locally unidentifiable or underpowered. | Nothing substantive; this is an inconclusive result, not evidence for flatness. |
| Results occur equally at `step0` and trained checkpoints | The structure is architectural, decoder-induced, or stimulus-induced rather than learned through pretraining. | The claim that pretraining learned the observed connection profile. |
| Results strengthen systematically across checkpoints and replicate across seeds | Pretraining organizes the predictive semantic field in the measured way. | The null that the result is initialization or prompt noise. |

### What would specifically support the original intuition

The strongest support requires all of the following:

1. semantic fields vary across contexts more than matched noise and shuffled controls;
2. $e$-transport is reliably worse than a non-$e$ connection on held-out contexts;
3. the geometric correction improves actual interventions, not only vector alignment;
4. correction size or curvature predicts where untransported addition fails beyond distance and norm controls;
5. small-loop transport or composition exhibits the path dependence predicted by curvature;
6. the effect develops through pretraining and replicates across seeds.

Then the defensible conclusion is:

> Ambient vector addition exists, but it is not the empirically correct global equivalence rule for these semantic transformations. Predictive information geometry supplies a better local transport law, and curvature obstructs extending it to a path-independent global vector representation.

### What would disprove the strong version

The strong version is rejected if, after adequate power and controls:

1. constant $e$-transport predicts held-out fields as well as or better than every connection-aware alternative;
2. transported interventions do not improve semantic transfer;
3. curvature and local connection-defect statistics add no predictive value beyond Fisher distance, entropy, and vector norm;
4. semantic composition has no reproducible path/order effect beyond sampling noise;
5. checkpoint results do not differ from initialization.

That result would not disprove the existence of predictive Fisher geometry. It would show that this geometry is descriptively real but operationally unnecessary for the tested semantic transformations.

## 7. The expanded evidence program

The narrow alpha-index experiment is the keystone, not the whole program. The broad hypothesis has five connected empirical layers.

### Study I: geometry exists and is model-specific

Measure at controlled contexts and checkpoints:

- variation of $G(h)$;
- semantic-plane LC sectional curvature;
- matched random-plane curvature;
- cubic connection-defect scalar $\chi$;
- differences from `step0` and across seeds.

This distinguishes the known curvature of the ambient categorical simplex from learned structure in the decoder subfamily and selected semantic planes.

**Evidence supplied:** whether the exact predictive geometry is nonconstant, curved, semantic-direction selective, and training dependent.

**Not supplied:** whether semantics follows that geometry.

### Study II: connection selection for semantic fields

Fit the semantic alpha-index on local training edges and evaluate $e$, LC, $m$, and fitted-alpha transport on held-out concepts/templates.

**Evidence supplied:** which notion of “the same transformation” best explains observed field variation.

**This is the minimal MSc contribution.**

### Study III: causal transfer intervention

At target base state $h_j$, compare:

\[
h_j+sV_i,
\qquad
h_j+sP^{(\alpha)}_{i\to j}V_i,
\qquad
h_j+sV_j,
\]

at matched target Fisher norm. Measure target log-odds, off-target KL, anti-steering frequency, and distance to the locally observed target effect. Include matched random, opposite-sign, and scalar-rescaling controls.

**Evidence supplied:** whether connection-aware transport is causally useful rather than merely a better descriptive fit.

### Study IV: composition and holonomy

Choose two controlled operations $A$ and $B$. Compare both textual orders where linguistically meaningful and compare the corresponding transported intervention loops. Test whether

\[
P_{\partial\square}-I
\approx
-\operatorname{Area}\,R(A,B)
\]

predicts observed order/composition discrepancy after matching edge lengths and loop area.

**Evidence supplied:** whether the curvature obstruction has an observable semantic consequence: path-dependent composition.

This study must not equate linguistic noncommutativity with curvature automatically. Syntax, prompt differences, and model nonlinearities are alternative causes and require factorial controls.

### Study V: pretraining dynamics

Repeat Studies I--III at initialization, early, middle, and final PolyPythia checkpoints across seeds.

**Evidence supplied:** whether the geometry/semantic coupling is learned during next-token pretraining rather than merely induced by the softmax architecture.

### Study VI: joint failure decomposition

Let $X_s$ and $X_t$ be locally observed semantic interventions. Raw vector reuse is exponential/ambient transport. For any candidate semantic connection,

\[
X_t-P^e_{s\to t}X_s
=
\underbrace{X_t-P^\nabla_{s\to t}X_s}_{\text{field nonparallelism}}
+
\underbrace{\left(P^\nabla_{s\to t}-P^e_{s\to t}\right)X_s}_{\text{connection mismatch}}.
\]

This suggests three experimentally distinguishable sources of steering failure:

1. **Support error:** the intervened point leaves the high-density set of naturally realized activations.
2. **Metric error:** the intervention causes unnecessary output KL or off-target predictive change.
3. **Transport error:** a source semantic vector is not the appropriate direction at the target context under the candidate connection.

Existing work largely studies these separately: [Manifold Steering](https://arxiv.org/abs/2605.05115) emphasizes support, [FishBack](https://arxiv.org/abs/2605.17231) emphasizes predictive metric cost, and [Hu et al.](https://arxiv.org/abs/2607.04525) together with [Steering Vector Fields](https://arxiv.org/abs/2602.01654) emphasize state-dependent transformations. The broader program tests their incremental and joint prediction of transfer failure.

Support error is the least exact component because the natural state set is sampled and may not be a smooth density manifold. It should be measured with several declared proxies and never confused with exact decoder-manifold membership.

Intermediate transformer layers are an optional later extension. They require pulling the Fisher metric through the remaining network Jacobian and differentiating that pullback to obtain connection terms. That is substantially more expensive and should not be required for the first complete thesis.

## 8. Gated proof ladder

The word “proof” should be reserved for the theorem-level implications. Empirical support becomes progressively stronger through these gates.

| Gate | Requirement | If passed | If failed |
|---|---|---|---|
| 0. Numerical validity | Synthetic curvature, coordinate invariance, stable solves, RK4 convergence, and tensor-to-holonomy agreement | The implementation measures the stated geometry. | Stop semantic interpretation. |
| 1. Transformation validity | Trained models show the predeclared semantic/log-odds manipulation beyond `step0` and shuffled controls. | The field arrows correspond to a behavior the model encodes. | Change stimuli/model before drawing geometric conclusions. |
| 2. Structured geometry | Semantic curvature/defect differs from matched random directions and training initialization after entropy/conditioning controls. | Geometry is structured and potentially learned. | Claim only generic decoder geometry. |
| 3. Held-out connection prediction | One transport lowers held-out field error across templates and seeds. | The chosen connection is a better descriptive law of semantic transfer. | The predictive Fisher connection family does not explain the field at tested scale. |
| 4. Causal intervention | Transported source vectors improve target behavior over constant, scalar, random, and opposite-sign controls. | Connection-aware transport is operationally useful. | Descriptive alignment is not sufficient evidence of semantic relevance. |
| 5. Curvature-specific semantics | Tensor curvature predicts numerical holonomy and controlled order/path effects beyond loop area and linguistic interaction. | Curvature, not merely a varying metric, has an observed semantic consequence. | Connection variation may matter, but curvature-specific claims remain unsupported. |
| 6. Training dynamics | Effects co-develop with manipulation strength across checkpoints and replicate across seeds. | Evidence that pretraining organizes geometry and semantics together. | Treat the effect as architectural or stimulus dependent. |

The minimum persuasive thesis passes Gates 0--3. The strongest version of the original hypothesis requires Gates 4--6.

### Connection effects are not automatically curvature effects

This distinction is essential:

- The local cubic/Christoffel correction can change source-to-target transfer along one path even in a flat geometry or flat connection.
- Curvature is specifically detected by disagreement between paths, loop holonomy, or operation order after controlling for the loop.
- Therefore a successful alpha fit supports **connection-relative semantics**. It supports **curvature-relevant semantics** only when the independent holonomy/composition test also succeeds.

## 9. Final verdict on the original hypothesis

### Already disproved as originally worded

- Hidden representations are not vectors: false; they are vectors in the architecture.
- Addition does not remain in the decoder family: false for a finite affine-softmax intervention.
- A global faithful representation forces zero curvature: false; intrinsic curvature survives isometric embedding.
- Dually flat means LC-flat: false; it refers to the exponential and mixture connections.
- Levi--Civita is automatically the correct semantic transport: not established.

### Already proved after correction

- The predictive decoder induces an exact Fisher metric.
- Hidden addition is exponential-affine but generally not Fisher-metric preserving.
- Nonzero LC curvature obstructs global Fisher-isometric vector translation.
- Parallelness and linearity are connection relative.
- Curvature obstructs global path-independent interior-alpha transport under the stated nondegeneracy condition.

### The central open claim

> Do actual semantic transformation fields behave more like constant exponential vectors, constant mixture covectors, LC-parallel fields, or operation-dependent connection-linear sections—and does the winning geometry predict intervention and composition failures?

The alpha-index experiment answers the first part. Causal steering answers operational relevance. Holonomy/composition answers whether curvature itself matters. Checkpoints answer whether the coupling is learned.

The expanded project therefore does not abandon the MSc novelty. It uses that result as the first identifiable test inside a broader theory of **connection-relative linearity**.

## 10. Beyond the first thesis

Once the final-head program is established, the framework expands naturally:

1. **Layerwise pullback connections:** propagate predictive Fisher geometry to intermediate activations and include second derivatives of the downstream map in the connection.
2. **Prediction horizon:** replace next-token behavior with $P(Y_{1:T}\mid h)$ and test whether preferred metrics/connections change with semantic horizon.
3. **Gauge and quotient structure:** compare only quantities invariant under behavior-preserving reparameterizations and quotient decoder-null fibers.
4. **Learned connections:** if every fixed alpha fails, estimate a low-complexity connection field and test torsion, nonmetricity, curvature, and held-out transport separately.
5. **Semantic holonomy atlas:** identify operations and regions whose holonomy fixes a direction versus rotates it, giving a local map of where global concept vectors can or cannot exist under a connection.

These are broader research directions, not requirements for showing whether the present hypothesis survives its first decisive tests.

## Related project documents

- [MSC_NOVELTY_PROPOSAL.md](MSC_NOVELTY_PROPOSAL.md): exact alpha estimator, closest literature, and CPU protocol.
- [PROOFS_AND_NOVELTY.md](PROOFS_AND_NOVELTY.md): full theorem and curvature audit.
- [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md): curvature, connection, composition, and checkpoint controls.
- [RESEARCH_NOTES.md](RESEARCH_NOTES.md): development of the original and audited hypotheses.
