# Training regimes for connection-relative representations

**Status:** proposed causal experiments; not yet implemented
**Constraint:** CPU-first, matched-compute comparisons

## 1. The training question

The observational project asks which connection semantic transformations already follow. The training project asks a stronger causal question:

> Can a model be trained so a controlled semantic transformation becomes approximately parallel under a chosen predictive connection, and does that improve held-out transfer or composition at matched language-model quality?

This is more meaningful than merely measuring curvature. A representation can become geometrically simpler without becoming behaviorally better, and an optimizer can change representation coordinates without changing the model's function.

The success criterion is therefore:

\[
\text{behavioral generalization gain}
\quad\text{at matched NLL/perplexity and compute},
\]

not “lower geometric loss” by itself.

## 2. Core geometric loss

For a semantic operation $T$, take two base contexts $c_i,c_j$ and their transformed forms. At the state entering the affine LM head, define

\[
h_i=h(c_i),
\qquad
X_i=h(Tc_i)-h(c_i),
\]

\[
h_j=h(c_j),
\qquad
X_j=h(Tc_j)-h(c_j),
\qquad
a_{ij}=h_j-h_i.
\]

At $h_i$, let

\[
G_i=\operatorname{Cov}_{p_i}(W_Y),
\qquad
A_{i,a}v=G_i^{\dagger}C_i(a,v,\cdot).
\]

For $\beta=(1-\alpha)/2$, first-order alpha transport is

\[
\widetilde X_{i\to j}^{(\alpha)}
=
X_i-\beta A_{i,a_{ij}}X_i.
\]

The symmetric normalized consistency loss is

\[
\ell_{\alpha}(i,j)
=
\frac{
\frac12
\|X_j-\widetilde X_{i\to j}^{(\alpha)}\|_{G_j}^2
+
\frac12
\|X_i-\widetilde X_{j\to i}^{(\alpha)}\|_{G_i}^2
}{
\operatorname{sg}\!\left[
\|X_j\|_{G_j}^2+\|X_i\|_{G_i}^2+\varepsilon
\right]
}.
\]

Here $\operatorname{sg}$ denotes stop-gradient. Stopping the scale prevents the model from lowering the normalized objective by manipulating its denominator.

The complete objective is

\[
\mathcal L
=
\mathcal L_{NLL}
+\lambda_{geom}\,\mathbb E[\ell_\alpha]
+\lambda_{strength}\,\mathcal L_{strength}.
\]

$\mathcal L_{strength}$ prevents the trivial solution $X_T\approx0$. In the synthetic grammar, ordinary next-token supervision supplies most of this constraint; an explicit margin on the operation's target log-odds is retained as a diagnostic and optional penalty.

## 3. Matched training conditions

The first causal comparison should use the same architecture, token order, optimizer, number of tokens, batch construction, and seeds.

| Regime | Transport assumption | Geometry loss |
|---|---|---|
| R0: language model | none beyond NLL | $0$ |
| R1: Euclidean constant vector | ambient/e-flat | $\|X_j-X_i\|_2^2$ |
| R2: Fisher-weighted e | $\alpha=1$ | $\|X_j-X_i\|_{G_j}^2$ |
| R3: LC consistency | $\alpha=0$ | $\ell_0$ |
| R4: mixture consistency | $\alpha=-1$ | $\ell_{-1}$ |
| R5: matched random correction | control | same correction norm as R3, random Fisher direction |

R2 is essential. If R3 beats R1 but not R2, the gain may come from Fisher weighting rather than LC transport.

Do not make a jointly learned alpha the primary training condition. It introduces an extra degree of freedom and can absorb scale or approximation error. First compare the fixed canonical connections, then fit alpha after training on one split and evaluate it on another. A trainable operation-specific alpha is a secondary experiment only after the fixed comparison succeeds.

## 4. CPU experiment A: tiny controlled language model

### Data

Generate a small grammar with known transformations and held-out compositions:

- singular/plural subject number;
- present/past temporal cue;
- affirmative/negative polarity;
- lexical categories with train/test-disjoint nouns and verbs;
- template paraphrases held out as complete families.

At least two transformations should commute symbolically, for example number and temporal cue. This provides four factorial corners and a known ground truth for compositional generalization.

The confirmation split holds out:

1. lexical items;
2. template families;
3. selected combinations of otherwise observed factors;
4. longer dependency distances than training.

### Model

A practical starting configuration is:

- 2--4 decoder-only transformer layers;
- hidden dimension 64;
- 4 attention heads;
- vocabulary 65 in the primary saturated-head experiment;
- sequence length at most 32;
- 1--5 million training tokens;
- four fixed random seeds.

This keeps exact full-vocabulary $G$ and cubic contractions feasible on CPU.

### Geometry-locked primary model

Use a fixed saturated softmax head

\[
W=
\begin{bmatrix}
I_d\\
0
\end{bmatrix},
\qquad b=0,
\qquad V=d+1.
\]

The transformer learns the natural logits $h$, but cannot game the regularizer by changing decoder rank or rotating a learned head. The Fisher metric remains exact and identifiable in the interior. For $d=64$, this gives a 65-token synthetic vocabulary.

Because this is the saturated categorical family, its LC sectional curvature is the classical constant $1/4$. This experiment isolates the **transport-law causal question**; it cannot demonstrate that curvature magnitude itself was learned. A learned minimal head is a secondary robustness condition, while model-specific curvature learning is studied with Pythia checkpoints.

For every random seed, save one initialization and branch it into R0--R5. Paired branches must see identical NLL examples in identical order. This substantially improves the causal comparison over independently initialized arms.

### Batch construction

Each geometric batch contains blocks

\[
(c_i,Tc_i,c_j,Tc_j)
\]

from the same semantic operation but different lexical/template contexts. Language-model examples and geometric blocks are sampled separately so every regime sees the same NLL tokens.

Only the final state entering the LM head is regularized in the first study. This preserves the exact exponential-family formulas and avoids ambiguous intermediate-layer geometry.

## 5. Stable optimization protocol

Exact evaluation and stable training have different numerical requirements.

### During training

- Use a declared damped solve $G_{\tau}=G+\tau\lambda_{max}I$.
- Report $\tau$ and tune it only on the pilot split.
- Recompute the connection operator at every geometric batch.
- In the primary stable implementation, compute the connection correction and target metric from the current batch, then stop-gradient both for that optimizer step. Gradients still flow through the explicit source and target semantic vectors in the residual. This is an alternating surrogate, not the exact gradient of one fixed geometric objective.
- Validate a small exact-gradient implementation on the tiny model before attempting it at larger scale.
- Clip only optimizer gradients, not geometric measurements.

An intermediate implementation may stop-gradient only probabilities, centered decoder rows, and the solve while retaining differentiation through tensor contractions with $a$ and $X$. Compare it against the fully detached version on the tiny model before using it as a scientific condition.

### During evaluation

- Use float64 geometry.
- Use the undeformed Fisher metric on the numerically resolved quotient/subspace.
- Report rank and threshold sensitivity rather than silently adding a ridge.
- Compare first-order training transport with converged RK4 transport on a held-out subset.
- Use the full vocabulary for the tiny model.

### Output-level alternative for finite transformations

The tangent loss is a local approximation. For finite prompt changes, also form

\[
\widehat p_j^T
=
p_\theta\!\left(h_j+\widetilde X_{i\to j}^{(\alpha)}\right)
\]

and minimize or evaluate

\[
\mathcal L_{\alpha}^{output}
=
D_{KL}\!\left(
\operatorname{sg}[p_\theta(\cdot\mid Tc_j)]
\,\|\,
\widehat p_j^T
\right).
\]

The stop-gradient target prevents both distributions from moving together. The Fisher tangent loss is the local quadratic approximation to this output-level discrepancy; agreement between them is an important validity check.

## 6. Hyperparameter selection without favoring a geometry

Choose $\lambda_{geom}$ separately for each regime on a pilot split to match one of two fair budgets:

1. **NLL-matched:** choose the strongest regularization whose validation NLL is within a predeclared tolerance of R0;
2. **Pareto comparison:** report the frontier of semantic transfer versus validation NLL over the same lambda grid.

Use the same grid and number of pilot trials for R1--R5. Freeze it before the confirmation seeds.

Do not select lambda using the final connection residual alone. That would guarantee the geometry looks successful even if behavior degrades.

## 7. Primary outcomes

### Behavioral outcomes

1. held-out next-token accuracy for the controlled transformation;
2. compositional generalization on unseen factor combinations;
3. long-distance agreement accuracy;
4. target log-odds change under a transported intervention;
5. off-target KL and anti-steering frequency;
6. ordinary validation NLL/perplexity.

### Geometric outcomes

1. held-out $e$, LC, and $m$ transport residuals;
2. post-hoc semantic alpha-index;
3. local defect scalar $\chi$;
4. semantic versus random-plane sectional curvature;
5. tensor-predicted and numerically integrated holonomy;
6. metric rank and condition number.

### Primary causal contrasts

\[
\text{R3}-\text{R2}
\]

tests whether LC transport adds value beyond the same Fisher evaluation metric.

\[
\text{R3}-\text{R5}
\]

tests whether the signed geometric correction matters beyond correction magnitude.

\[
\text{R1}-\text{R0}
\]

tests the standard global-vector consistency intervention.

All uncertainty is computed over independently trained seeds and held-out template/lexical blocks, not token positions or graph edges.

## 8. What counts as a meaningful training result

A strong positive result requires all of:

1. the target regime reduces its held-out transport residual;
2. it improves at least one preregistered behavioral generalization outcome;
3. validation NLL is matched or the result lies on a better Pareto frontier;
4. the gain replicates across at least three seeds;
5. it beats R2 and R5, not merely R0;
6. the post-hoc preferred connection shifts in the predicted direction;
7. the result survives exact RK4 evaluation and numerical-threshold checks.

The following are not meaningful by themselves:

- lower curvature;
- lower training geometric loss;
- a prettier two-dimensional representation plot;
- a changed alpha-index without held-out transfer;
- improved training accuracy with worse held-out NLL;
- a result from one seed.

## 9. Interpretation matrix

| Training result | Interpretation |
|---|---|
| R3 improves LC residual and behavior over R2/R5 | Causal support that metric-compatible transport can produce better semantic transfer. |
| R2 performs best | Fisher weighting helps, but curvature/LC correction is unnecessary. |
| R1 performs best | The task rewards a conventional global vector representation. |
| R4 performs best | Dual/expectation-coordinate invariance is the useful training bias. |
| Each regime lowers only its own residual | Geometry is shapeable, but no behavioral advantage has been shown. |
| Every geometry hurts NLL or generalization | Connection regularization is the wrong inductive bias or is too approximate. |
| Geometry changes without behavioral change | Likely gauge/representation selection rather than functional improvement. |
| LC gains disappear under exact transport | First-order or damping artifacts caused the apparent result. |

## 10. CPU experiment B: Pythia-14M adapter fine-tuning

Run only after the synthetic study establishes a stable implementation.

1. Freeze the tokenizer, LM head, and most transformer weights.
2. Train a small low-rank adapter or the final block on 5,000--20,000 controlled paired contexts.
3. Compare R0, R2, and R3 first; add R4 only if conditioning permits.
4. Use three seeds and identical example order.
5. Evaluate on unseen lexical items, templates, and operation combinations.
6. Measure ordinary LM loss on an untouched text sample to detect degradation.

Freezing the LM head keeps the predictive geometry reference stable while the adapter learns where contexts land in that geometry. This is cheaper and easier to interpret than full-model fine-tuning.

The strongest result would be an LC-trained adapter that improves held-out semantic transfer and lowers off-target KL over an NLL-matched Fisher-e adapter. An $e$-trained adapter winning would be equally informative evidence against the LC-specific hypothesis.

## 11. Do not directly minimize curvature first

Curvature is not inherently bad. It encodes path dependence and may be required for useful context sensitivity. Directly forcing $R^{LC}\to0$ can:

- collapse meaningful semantic distinctions;
- encourage metric degeneracy;
- select an arbitrary representation gauge;
- improve a geometric statistic without improving behavior.

Train the property actually desired—held-out covariant consistency of a declared semantic operation—and measure curvature as an explanatory variable. Curvature regularization becomes justified only if independent experiments first show that holonomy predicts a specific harmful composition error.

## 12. Novelty boundary for training

The generic idea of penalizing covariant derivatives is established prior art. [Parallel Field Regularization](https://proceedings.neurips.cc/paper/2011/hash/bc6dc48b743dc5d013b1abaebd2faed2-Abstract.html) explicitly regularizes a vector field toward $\nabla V=0$; [Multi-task Vector Field Learning](https://proceedings.neurips.cc/paper/2012/hash/a5e00132373a7031000fd987a3c9f87b-Abstract.html) extends this to multiple task fields; and [Parallel Vector Field Embedding](https://jmlr.org/beta/papers/v14/lin13a.html) develops the construction further.

Euclidean semantic parallelogram training is also established by [Analogy-preserving Semantic Embedding](https://proceedings.mlr.press/v28/juhwang13.html), while [Natural Alpha Embeddings](https://arxiv.org/abs/1912.02280) already applies alpha-geometry to static conditional word distributions. Recent [Semantic Tube Prediction](https://arxiv.org/abs/2602.22617) regularizes consecutive LLM hidden displacements toward local collinearity. These are required conceptual baselines, not claims to rediscover.

Therefore do not claim that parallel-field regularization, covariant consistency, or the energy $\int\|\nabla X\|^2$ is new.

The defensible contribution is the exact combination of:

1. an autoregressive model's predictive categorical Fisher pullback;
2. explicit comparison of Amari $e$, LC, $m$, and fitted-alpha connections;
3. contextual linguistic transformation fields;
4. full-vocabulary tangent and output-level objectives;
5. matched causal training arms and held-out intervention/composition evaluation.

Safe wording is:

> Prior work has separately regularized Riemannian fields toward parallelism, trained Euclidean analogy/equivariance structure, and applied alpha-geometry to static conditional embeddings. To our knowledge, no existing work treats a contextual linguistic intervention as a tangent-vector field on an autoregressive model's predictive softmax manifold and trains that field toward parallelism under a selected Amari alpha-connection.

## 13. Decision sequence

1. Replicate the existing checkpoint pilot across seeds and many prompts.
2. Run the tiny synthetic training comparison R0--R5.
3. Require behavioral evidence before expanding.
4. Run the Pythia-14M adapter comparison R0/R2/R3.
5. Only then study trainable alpha, holonomy penalties, intermediate layers, or longer predictive horizons.

This sequence produces a useful negative result at every stopping point and requires no GPU.
