# Combined Red-Team Audit: Round 5

Date: 2026-08-17  
Repository state audited: commit `983121ec5af65d28367553f1432d977ce99358df`  
Scope: mathematics, manuscript, protocol, code, tests, empirical artifacts,
statistics, reproducibility, and public claims.

This report combines two adversarial reviews:

1. the Opus 5 review supplied in the review transcript; and
2. an independent repository-wide audit with executable numerical probes.

It does not silently promote one reviewer's assertion into a fact. Findings are
labeled as follows:

- **CONFIRMED**: reproduced from the current checkout or independently derived;
- **CONDITIONAL**: correct only under a stated statistical or design
  interpretation;
- **ATTRIBUTED**: reported by Opus 5, but its exact reproduction artifact was
  not present in this checkout;
- **CLEARED**: specifically attacked and no defect was found. This is evidence,
  not a proof of absence.

## Remediation status after the audit

The findings below are preserved as the historical attack record. This table
is the current disposition after the 17 August 2026 remediation pass and keeps
the claim domains separate.

| Domain | Resolved in this pass | Still open or externally blocked |
|---|---|---|
| Mathematics | All 21 formal results have explicit proofs; the LC certificate's \(\lambda^{-2}\)/\(\lambda^{-3}\) scaling and archived conditioning ratios are disclosed | No instantiated trained-model certificate because the required pathwise envelopes were never measured; no formal proof-assistant certification |
| Manuscript | Adjacent 2026 work is cited and distinguished; stale implementation/status language and identity-versus-measurement wording are corrected | PDF rebuild and visual inspection require a LaTeX engine not installed here |
| Protocol/specification | Pure-relative primary refinement with zero dimensionful absolute floor, exact-JVP audit, distinct rank/conditioning/chart-scale gates, immutable packet provenance, typed certified-versus-sampled bounds, crossed/nested statistics, numerical success thresholds, feasibility, multiplicity, and failure rules are explicit | A confirmatory prompt manifest and simulation-based seed count require study inputs that do not yet exist |
| Core geometry code | Stable chord-based Fisher distance/log, stable exponential log, and fail-closed boundary/overflow behavior replace the silent collapse and `nan` paths | Extreme inputs that are not representable in float64 are deliberately rejected, not claimed accurate |
| Distillation code | D1--D4 and D6--D9 are repaired; packet schema v4 validates provenance, tensor semantics, gate thresholds, and accepted status | Real-model packet generation and student training remain unimplemented |
| Tests | All adversarial regressions in Section E were added; 100 tests pass and Ruff passes | Hosted CI is configured but has not yet produced a remote run record |
| Empirical artifacts | New smoke results use `pythia-smoke-2`, immutable snapshot commits, distinct model/control seed fields, tokenizer/outcome/design/environment hashes, and repository dirty state; synthetic recovery now uses independent closed-form targets | The archived pilot remains a legacy execution-only artifact and cannot acquire provenance it never recorded; no substantive multi-seed result exists |
| Statistics | Concrete coverage, NLL, behavioral-effect, Fisher-alignment, chart-shrinkage, certificate, feasibility, intention-to-treat, hierarchy, and 80%-power rules are frozen in the protocol | Power cannot be numerically instantiated until engineering-pilot variance components exist; confirmatory inference is blocked until a versioned sampling frame exists |
| Reproducibility/governance | `pyproject.toml`, exact validated lock files, and Windows/Linux CI were added; editable packaging and the result schema were exercised locally | The author must choose a software license; that legal choice is not inferred by the audit |
| Public claims | README, overview, and protocol now separate mathematical identities, numerical validation, and empirical measurements | There is still no confirmatory empirical claim to report |

## Executive verdict at discovery time

The table below records the state that produced the findings. It is superseded
for current code status by the remediation table above, but retained so the
severity of the original failures is not rewritten after the fact.

| Domain | Verdict | What survives | What fails now |
|---|---|---|---|
| Mathematics | **Standing, not certified** | No theorem-level contradiction was found by either audit; all 21 formal results now have explicit proofs | Finite-scale usefulness remains unestablished in the trained-model regime |
| Manuscript | **Strong theory draft, not submission-ready** | Claims are mostly disciplined and conditional; proof completeness is repaired | Stale PDF, incomplete related work, and no complete trained-model certificate |
| Protocol | **Thoughtful but not executable as written** | It states many correct gates and falsification conditions | Several gates, metadata requirements, and statistical claims are not enforced by code |
| Core geometry code | **Correct in ordinary interior regimes; unsafe near numerical boundaries** | Moderate-regime identities and the archived pilot's large effects are not obviously invalidated | Fisher and exponential maps silently collapse or emit `nan` on valid open-simplex inputs |
| Distillation code | **Not safe for scientific use** | Tensor assembly works in ordinary fixtures | It can penalize pure gauge, accept inaccurate packets, trust a false JVP, and deserialize an accepted negative metric |
| Tests | **Healthy regression suite with material blind spots** | 79 tests pass | Passing tests do not exercise the failure regimes that matter most |
| Empirical evidence | **Execution only** | The Pythia pipeline ran and deterministic identities were observed | No multi-seed, preregistered, continuous-tangent, behavior-linked result exists |
| Statistics | **Pilot quality** | Conditional randomization logic is stated carefully | Replication units, multiplicity, effect thresholds, power, and sampling frame are not confirmatory |
| Reproducibility | **Partial** | Source and archived JSON are available | Environment, exact model snapshots, result schema, and Git provenance are not pinned |

At discovery time, the six months of work were **not destroyed by a discovered mathematical
counterexample**. They are at risk of being oversold or empirically invalidated
because the then-current measurement and packet pipeline could produce silently wrong
numbers. The defensible status is therefore: **promising conditional theory,
competent moderate-regime geometry, unsafe distillation core, and no substantive
empirical result yet**.

---

## A. Mathematics

### A0. Scope of the mathematical clearance

**ATTRIBUTED / PARTIALLY CONFIRMED.** Opus 5 reports independently re-deriving
all 16 results in its theorem table, including:

- the constants in `D_LC`;
- both transport-bound branches;
- the Bernoulli coefficient `C_{theta theta theta} = 2 cot(theta)`;
- `||D^2 psi[u,v]||^2 = (1/4) E[S(u)^2 S(v)^2]`; and
- the curvature convention `R = -(1/4)[A_u,A_v]` used by the code.

The independent audit likewise found no contradiction in the pullback Fisher
formula, the Amari connection formulas, the KL counterexample, naturality,
the square-root Levi--Civita stability argument, the nonzero-alpha Bernoulli
separation, the Sobolev interpolation route, or the conditional-variance
identity.

However, `paper/main.tex` currently contains **21** theorem, proposition, and
corollary environments, not 16. The missing Opus theorem table and derivation
notes are not committed. The strongest defensible statement is therefore:

> Two adversarial reviews found no mathematical defect in the central results;
> this is not a complete, independently archived proof audit of every formal
> statement.

Do not write “all theorems are certified correct.” Write “no error was found in
two independent adversarial derivations,” and archive the derivation ledger if
that assurance is important.

### A1. Missing proof for two-model convergence

**CONFIRMED.** `cor:two-model-convergence` is stated at
`paper/main.tex:754-763` without a proof. A KL triangle inequality is neither
available nor needed. The proof is the operator-norm triangle inequality after
comparing each model to the common target:

\[
\|P^{A,n}_\gamma-P^{B,n}_\gamma\|_{\mathrm{op}}
\le
\|P^{A,n}_\gamma-P^*_{\gamma}\|_{\mathrm{op}}
+
\|P^*_{\gamma}-P^{B,n}_\gamma\|_{\mathrm{op}}.
\]

Apply the preceding risk-to-transport theorem to each summand, using the
uniform hypotheses to absorb the two constants into one. For the aligned
version, first pull model B back by `Phi_n` and use the same argument. The
corollary appears true; the omission is a proof-completeness defect, not a
counterexample.

**RESOLVED 2026-08-17.** `paper/main.tex` now includes this proof explicitly.
The stricter follow-up audit also found three immediate corollaries whose proofs
were implicit rather than enclosed in proof environments. Explicit proofs were
added for `cor:intrinsic-field`, `cor:connection-decomposition`, and
`cor:cross-model-semantic`. All 21 theorem/proposition/corollary environments
now have explicit proofs.

### A2. The square-root LC theorem may be finite-scale vacuous

**CONFIRMED AS A RISK; ATTRIBUTED FOR THE EXACT NUMBER.** The theorem is valid,
but its displayed connection constant contains `lambda^-2`:

\[
D_{LC}
=\lambda^{-1}(B_1\delta_2+B_2\delta_1)
+2B_1^2B_2\lambda^{-2}\delta_1.
\]

The smallest archived final-checkpoint metric eigenvalue is
`1.1988998311190128e-05`, versus approximately `2.77e-03` at initialization.
That makes the inverse-square factor roughly 53,700 times worse before changes
in the other envelopes are considered.

Opus 5 reports a bound of `7.4e8` at the final checkpoint for `delta=1e-3`,
versus `0.43` at initialization, and says a 10% certificate would need
`delta approximately 1.4e-13`. The exact `7.4e8` calculation cannot be replayed
from committed artifacts: the pilot did not store the required uniform
along-path `B1`, `B2`, `delta1`, `delta2`, and path-length inputs. The existing
resolution document itself acknowledges that omission.

The correct conclusion is not “the theorem is wrong” or even “it is proved
vacuous for every trained model.” It is:

> Vocabulary and minimum-token-probability dependence were removed, but rank
> conditioning remains. The repository has not shown a nonvacuous trained-model
> certificate at its own observed Fisher scales.

Put an instantiated sensitivity table in Limitations, with every input and
whether it is pointwise, sampled-along-path, or uniformly certified.

**MATHEMATICAL DISCLOSURE RESOLVED 2026-08-17; MEASUREMENT OPEN.** The theorem
and appendix now expand the metric-compatible branch as

\[
L_\gamma\left[
\frac{B_1^2(B_1\delta_2+B_2\delta_1)}{\lambda^2}
+\frac{2B_1^4B_2\delta_1}{\lambda^3}
\right],
\]

and report the archived pointwise inverse-power diagnostics. The final
checkpoint's worst `lambda^-1`, `lambda^-2`, and `lambda^-3` factors are about
`231`, `5.34e4`, and `1.23e7` times their initialization values. A complete
certificate remains open because `B2`, `delta1`, `delta2`, and pathwise extrema
were not measured.

### A3. Mathematical cleared-list

**CLEARED.** The following attacks did not produce a defect:

- exact pullback Fisher metric and affine-softmax score moments;
- exact predictive naturality under the stated shared-image assumptions;
- KL-to-geometry counterexample arithmetic;
- lower-index LC identity in square-root coordinates;
- both Duhamel transport branches, including the `M_T + M_S` exponent;
- categorical-sphere LC parallel transport and metric compatibility;
- constant curvature `1/4` of the full categorical simplex;
- Amari `+/- alpha` duality pairing;
- Bernoulli nonzero-alpha separation and the `2 cot(theta)` cubic;
- cubic commutator curvature assembly and sign convention;
- Hilbert-valued Sobolev interpolation route, conditional on its stated
  uniform assumptions; and
- the population conditional-variance/Pythagorean decomposition.

---

## B. Manuscript

### B1. The manuscript's strongest part is its conditional scope

**CLEARED.** The paper generally distinguishes theorem from experiment,
ambient decoder geometry from support-intrinsic geometry, and a structural
identity from a semantic claim. Neither audit found a hidden theorem saying
that LC transport must win empirically.

### B2. Required manuscript repairs

1. **RESOLVED:** explicit proofs now cover all 21 formal results, including
   `cor:two-model-convergence`.
2. **CONFIRMED:** rebuild the committed PDF. `README.md:15` explicitly says
   the PDF is mathematically stale and `paper/main.tex` is authoritative.
3. **RESOLVED:** the appendix now gives the explicit \(\lambda^{-1}\),
   \(\lambda^{-2}\), and \(\lambda^{-3}\) sensitivity table and states which
   pathwise inputs remain unmeasured.
4. **RESOLVED:** `PROJECT_OVERVIEW.md` now distinguishes the implemented
   model-free packet core from the unimplemented real-model builder, trainer,
   and evaluator.
5. **RESOLVED:** public summaries now separate identities, numerical checks,
   and empirical measurements; see Section I.

### B3. Novelty and related-work risk

**RESOLVED REVIEW RISK; NOT A PRIORITY CLAIM.** The related-work section now
discusses and distinguishes two closely adjacent 2026 preprints:

- [The Shape of Beliefs: Geometry, Dynamics, and Interventions along
  Representation Manifolds of Language Models' Posteriors](https://arxiv.org/abs/2602.02315);
- [Latent Semantic Manifolds in Large Language Models](https://arxiv.org/abs/2603.22301).

The current contribution may still be distinct: the stability/separation
package, cross-model transport consequences, and semantic-sufficiency
obstruction are more specific than merely observing Fisher geometry. But the
paper's revised related-work paragraph now states that distinction explicitly.
Several proofs combine standard ingredients—pullback/naturality,
rational continuity, Duhamel/Gronwall, Pinsker plus Sobolev, and Hilbert
projection—so theorem correctness alone does not establish novelty.

### B4. Submission verdict

The manuscript is a serious theory draft. The proof gap, finite-scale
mathematical disclosure, and related-work comparison are repaired. It is not
ready for submission until the stale PDF is rebuilt, and it still cannot claim an
instantiated trained-model certificate. None of those items destroys the
mathematical program.

---

## C. Protocol and specification

The protocol was stricter than the implementation at discovery time. C1--C5
below are retained as the defect record; all five are resolved in the current
packet schema and protocol.

### C1. Historical: relative derivative-refinement tolerance was not implemented

**CONFIRMED AT DISCOVERY; RESOLVED.** `PREDICTIVE_CONNECTION_DISTILLATION.md:581-584` requires a
preregistered relative tolerance. `_max_relative_difference` at
`src/predictive_geometry/distillation.py:232-240` divides by
`max(1, ||tensor||)`. Every tensor with norm below one is therefore judged on
an absolute scale. This directly contradicts `build_packet`'s docstring at
lines 253-259.

### C2. Historical: required packet provenance was not represented

**CONFIRMED AT DISCOVERY; RESOLVED.** The specification at
`PREDICTIVE_CONNECTION_DISTILLATION.md:205-218` requires immutable model
revision, tokenizer/outcome hashes, chart and domain identifiers, context,
dtype, and quantization metadata. `ConnectionPacket` at
`src/predictive_geometry/distillation.py:47-75` stores none of the model,
tokenizer, outcome-map, chart-domain, context, or inference-dtype provenance.
It also does not store `metric_floor` or `refinement_rtol`, so acceptance cannot
be independently recomputed after serialization.

### C3. Historical: certified versus sampled constants were prose-only

**CONFIRMED AT DISCOVERY; RESOLVED.** `packet_transport_bound` says its continuum inputs must come from
a certificate or be clearly labeled empirical estimates, but accepts bare
floats and stores no provenance/status. `sobolev_grid_audit` is correctly called
an audit rather than a certificate in its docstring, yet the surrounding packet
format cannot prevent a sampled modulus from being presented downstream as a
certified bound.

### C4. The rank gate is a chart-unit gate

**CONDITIONAL AT DISCOVERY; RESOLVED.** `build_packet` rejected when the smallest raw metric eigenvalue
is below a fixed `metric_floor`. Rescaling the chart coordinates rescales the
metric and can flip an intrinsically full-rank packet from accepted to rejected.
Calling the rejection reason `"rank"` is therefore mathematically inaccurate;
it means “below an absolute numerical floor in the declared chart units.”

If chart scale is frozen in advance, an absolute floor can be a legitimate
numerical contract. If scale is learned, compared across arms, or not stored in
packet provenance, the gate is gameable and conflicts with the protocol's rule
that chart shrinkage is a failure. The implementation should separately report:

- exact/numerical rank or a relative eigenvalue ratio;
- absolute sensitivity in the frozen chart units; and
- the chart-scale contract.

### C5. Statistical-unit language is not defensible as written

**CONFIRMED AT DISCOVERY; RESOLVED.** `PREDICTIVE_CONNECTION_DISTILLATION.md:883-888` called training
seed, prompt template, lexical family, and semantic operation “independent
units.” They are normally crossed or nested factors, not mutually independent
replicates. A template contains lexical realizations; operations recur across
templates; every arm is evaluated under the same training seed. The proposed
bootstrap over seeds and prompt families does not automatically represent this
dependence.

The protocol should define the target population and use a crossed multilevel
model, a multiway cluster bootstrap, or a randomization procedure justified by
the actual assignment mechanism.

---

## D. Code

### D0. Severity summary

This table records discovery severity. D1--D9 are resolved in the current
checkout; the original mechanisms remain below as reproducible rationale for
the regression tests.

| ID | Severity | Status | Failure |
|---|---:|---|---|
| D1 | Critical | Confirmed | Centered-logit Jacobian loss centers the wrong axis for the repository's declared layout |
| D2 | Critical | Confirmed | Packet refinement gate is absolute below unit norm and accepts badly inaccurate tensors |
| D3 | Critical | Confirmed | A supplied “exact” JVP is trusted without consistency checking |
| D4 | Critical | Confirmed | Packet serialization preserves checksums but not semantic validity or provenance |
| D5 | High | Confirmed | Simplex maps silently collapse or emit `nan` near valid open-simplex boundaries |
| D6 | High | Conditional | Finite-fiber sufficiency attribution is biased for population estimation |
| D7 | High | Confirmed | Sobolev grid audit is biased and silently returns nonsense when under-resolved |
| D8 | Medium | Conditional | Fixed metric floor is chart-scale dependent and mislabeled as rank |
| D9 | Medium | Confirmed | Transport bound raises `OverflowError` on valid finite inputs |

### D1. Centered-logit Jacobian loss removes the wrong gauge

**CONFIRMED.** `_jet_tensors` explicitly requires logit Jacobians shaped
`(chart_dim, outcomes)` at `distillation.py:205-206`. For that layout, outcome
centering is across `axis=1`. `centered_logit_jacobian_loss` instead subtracts
the mean over `axis=0` at lines 521-529.

Executable counterexample: let `T` be any `2 x 3` logit Jacobian and let

```text
S = T + [[ 1.0,  1.0,  1.0],
         [-0.2, -0.2, -0.2]].
```

`S` and `T` describe the identical predictive derivative because the added
rowwise constants are pure softmax gauge. The current loss returns
`2.1599999999999997`, not zero.

This can systematically penalize the Jz control arm for gauge noise and make a
connection arm look better by construction. It affects the prospective
distillation experiment, not the archived Pythia curvature pilot.

### D2. The refinement gate can be anti-informative

**CONFIRMED.** A high-frequency Bernoulli family with step `h=0.1` was chosen so
the stencil aliases the derivative. The analytic metric is `4.0e-6`; the packet
reports `2.161518584923522e-6`, a true relative error of
`0.4596203537691194`. Nevertheless:

```text
accepted                  True
reported_refinement_error 2.161518584923522e-6
refinement_rtol           1.0e-3
```

The tensor is accepted because the denominator is forced to one. Opus 5 reports
an even worse chart-scale sweep—56% true error while the recorded refinement
error fell to `6.6e-16`—but that exact sweep script is not present. The
independent counterexample is sufficient to confirm the mechanism.

Use an explicit `atol + rtol * scale` contract, store both tolerances, report
per-tensor errors, and include an anti-aliasing or independent-step check.
Pure relative error also needs a declared zero-tensor policy; replacing the
current floor with division by a near-zero norm would create a different bug.

**FOLLOW-UP RESOLVED.** The first repair made the diagnostic relative but left
`refinement_atol=1e-6`, which could still dominate small first-kind tensors and
produce a false acceptance. The primary default is now zero (as is the JVP
audit's absolute floor), and an analytic scale-\(10^{-4}\) Bernoulli regression
with more than 10% first-kind relative error is rejected. The zero-tensor policy
is deliberately fail closed: if relative convergence of a structurally zero
tensor cannot be resolved, the packet is rejected rather than accepted under an
arbitrary chart-unit constant.

### D3. An incorrect exact JVP can pass every gate

**CONFIRMED.** The builder validates only the supplied JVP's dtype, shape, and
finiteness. It does not compare it with finite differences of the logits or
with the probability-difference cubic audit.

For a Bernoulli logistic family with `p=0.3`, the analytic cubic is `0.084`.
Supplying a zero JVP with otherwise correct logits produces:

```text
accepted             True
primary cubic        0.0
probability audit    0.08399993175002503
refinement_error     2.367383372670062e-11
```

The packet gate therefore certifies a primary tensor that is known to be
wrong. Exact autodiff is not infallible at an API boundary: wrong axis order,
wrong point, detached computation, or a zero-filled fallback all satisfy the
current checks.

### D4. The packet checksum proves integrity, not validity

**CONFIRMED.** The writer checks that stored eigenvalues match the stored
metric, but neither writer nor reader requires an accepted metric to be
positive definite or to clear the threshold under which it was accepted.
A manually constructed packet with `metric=[[-1]]`, eigenvalues `[-1]`, and
`accepted=True` survives `packet_to_dict` followed by `packet_from_dict` and
returns accepted.

The schema also omits the metadata required by Section C2. A checksum answers
“was this payload altered?” It does not answer “did this packet come from the
declared model/chart?” or “does its accepted flag follow the declared gate?”

### D5. Simplex numerics fail on valid inputs

**CONFIRMED.** These are numerical defects, not mathematical defects in the
underlying connections.

1. `fisher_distance` and `_sphere_log` use `arccos` of a dot product. For
   `p=(0.5,0.5)` and `q=(0.5+1e-8,0.5-1e-8)`, both the distance and the Fisher
   log return exactly zero.
2. Even `fisher_distance(p,p)` can return a positive noise floor of roughly
   `3e-8` for ordinary long normalized vectors because the coefficient is not
   exactly one.
3. For `p=(1-e,e)`, `q=(e,1-e)`, `e=1e-20`, the exponential log-exp round trip
   returns `(0.5,0.5)`, with maximum error `0.5`.
4. For the smallest positive float in one component and a finite tangent,
   exponential exp and parallel transport return `[nan, nan]` with runtime
   warnings instead of rejecting an unsafe calculation.

The archived Pythia prompt pairs had good log-exp reconstruction and their
large composition errors are not explained away by these probes. The failures
do contaminate near-zero closure/distance claims and make the general-purpose
API unsafe near the simplex boundary.

### D6. Sufficiency decomposition: exact identity, biased population attribution

**CONDITIONAL BUT IMPORTANT.** The implementation uses the same-sample fiber
mean. It gives an exact empirical Pythagorean split, which is why the additivity
test passes. If the rows are IID samples used to estimate population terms,
the components are biased.

For a balanced fiber of size `r`, with true conditional variance `sigma^2` and
true squared mismatch `m^2`:

\[
E[\widehat{\mathrm{insufficiency}}]
=(1-1/r)\sigma^2,
\qquad
E[\widehat{\mathrm{mismatch}}]
=m^2+\sigma^2/r.
\]

At `r=2`, `sigma^2=2`, and `m^2=1`, the expected reported terms are `1` and
`2`, although the population terms are `2` and `1`. The total remains unbiased
and exactly additive, so the attribution can be inverted while the current
test remains green.

If the finite rows exhaust the intended discrete population, there is no bias.
The function must therefore declare its estimand. For population inference,
use cross-fitting, an ANOVA correction, or a hierarchical estimator and report
fiber sample sizes.

### D7. Sobolev grid audit is biased and silently under-resolved

**CONFIRMED.** The routine takes forward differences and then applies a
trapezoid rule on successively shorter grids. For three samples of `f(x)=x` on
`[0,1]`, the `H^1` squared norm is exactly `4/3`, while the function returns
`0.875`. With one sample equal to `17` and requested order four, it returns
`0.0`.

The routine does not robustly validate positive finite spacing, finite values,
or that at least `order+1`/`order+2` points exist for the declared derivative
scheme. It is correctly labeled an audit rather than a certificate, but an
audit must still fail closed when it has no derivative information.

### D8. Metric floor and chart scale

See C4. The numerical behavior is real; the scientific severity depends on
whether chart units are frozen and stored. At minimum, rename the reason,
separate absolute and relative conditioning gates, and serialize the contract.

### D9. Transport-bound overflow

**CONFIRMED.** `packet_transport_bound(1000,1,1,1,0,0,0)` passes input
validation and then raises `OverflowError: math range error` at the exponential.
A bound routine should return `inf` with an explicit overflow status or compute
in log space; a valid finite input should not crash an evaluation run.

### D10. Code cleared-list

**CLEARED.** Neither audit broke:

- the corrected `output_axis` handling in tensor operator norms;
- the algebraic `M_T + M_S` transport exponent;
- ordinary-regime sphere parallel transport;
- holonomy against the closed-form sphere result;
- curvature assembly and the Amari sign convention;
- Fisher-Haar QR sampling;
- spectrum-block leverage preservation under the declared band rule;
- whole-context-block cubic shuffling behavior;
- the Weyl serialization eigenvalue bound;
- strict open-simplex validation in the packet compatibility path; or
- float64 logit enforcement at the packet API boundary.

---

## E. Tests

At the audited checkout, all 79 committed unit tests passed and Ruff passed,
yet the tests were structurally unable to detect the main failures. The
remediation adds every regression requested below; the current suite has 100
passing tests and Ruff passes.

| Defect | Existing nominal coverage | Why it stays invisible | Required regression |
|---|---|---|---|
| Wrong Jacobian centering axis | `test_jacobian_versus_gram_distinction` | Fixture uses an `(8,2)` layout inconsistent with `_jet_tensors`' `(chart_dim,outcomes)` contract and asserts only that loss is positive | Add pure rowwise logit gauge and require exactly zero; validate shape/layout explicitly |
| Sufficiency attribution bias | `test_sufficiency_decomposition_is_exactly_additive` | Plug-in bias moves mass between the two terms and cancels in the total | Simulate known population fibers over repeated samples and test component expectations or label empirical estimand only |
| Absolute refinement gate | Packet refinement tests | Moderate tensors do not expose `max(1, norm)`; no scale/aliasing adversary | Scale the same chart family over many orders and include high-frequency stencil aliasing |
| False exact JVP | Exact-JVP fixtures | JVP agrees with the logits by construction | Inject a finite, correctly shaped wrong JVP and require rejection |
| Invalid accepted packet | Serialization round trip | Fixtures start from valid builder output | Serialize a negative or below-threshold metric marked accepted and require rejection |
| Fisher small-distance collapse | Interior round trip | Points are well separated | Test `1e-8` and smaller separations against a stable reference |
| Exponential boundary failure | Ordinary interior round trip | Probabilities are not extreme | Test `1e-20` and subnormal components; require accuracy or explicit failure |
| Sobolev bias/under-resolution | 201-point linear grid, `places=2` | Fine-grid `O(h)` boundary bias is hidden by a two-decimal assertion | Test coarse exact polynomials and reject too few samples |
| Bound overflow | One moderate finite bound | Exponential does not approach overflow | Test large finite exponent and require `inf`/status, not exception |

Notably, `field.py` already tested chart-scale invariance for its fitted field
objective. The repaired packet API now separates algebraic rank, relative
conditioning, and absolute chart-scale gates and records the chart contract.

---

## F. Experiments and empirical artifacts

### F1. There is no substantive empirical result

**CONFIRMED.** The stored Pythia run contains:

- one Pythia-14M model training run;
- four checkpoints;
- three hand-written prompt factorials; and
- eight Fisher-Haar controls per context.

The protocol itself labels this execution-only. Eight controls give a minimum
plus-one tail rank of `1/9`; the controls are not spectrum matched; conditioning
changes by orders of magnitude; and no behavioral transfer outcome is tested.
The real-model packet builder, student trainer, and evaluator listed at
`PREDICTIVE_CONNECTION_DISTILLATION.md:1184-1193` are unimplemented.

No finding in this report converts the pilot into negative evidence. It remains
an engineering record, not evidence for or against semantic geometry.

### F2. “One seed” versus four recorded seeds

**CONFIRMED AS AMBIGUOUS METADATA, NOT FOUR MODEL REPLICATES.** The archived
files contain different `random_seed` values across checkpoints. Those are
control/RNG seeds; the weights are checkpoints from one model training run.
Calling them simply `random_seed` invites readers to mistake four computational
seeds for four independent model seeds. Store separate fields:

- `model_training_seed` or an immutable model-run identifier;
- `control_seed` and seed derivation scheme; and
- `checkpoint_revision`/snapshot hash.

**RESOLVED FOR NEW OUTPUTS.** Schema `pythia-smoke-2` uses those distinct
fields and explicitly marks checkpoints as repeated measures from one model
run. The historical file remains unchanged and legacy-labeled in the protocol.

### F3. Result schema drift and incomplete provenance

**CONFIRMED.** Archived JSON uses `random_planes_per_context`; current source
emits `control_planes_per_context` plus newer threshold and banding fields.
There is no result-schema version or migration. Current output metadata records
symbolic model/revision and settings but not:

- repository Git SHA and dirty state;
- exact Hugging Face snapshot commit;
- tokenizer artifact hash;
- Python/platform/package versions; or
- an environment lock hash.

The local cache contains exact snapshot commits, but the JSON does not. The
historical control draws also cannot be reproduced by current code because
hashed control seeding was introduced later; the protocol discloses this
correctly.

**RESOLVED FOR NEW OUTPUTS.** The current schema stores repository SHA/dirty
state, resolved model and tokenizer snapshot commits, tokenizer and outcome-map
hashes, Python/platform/package versions, exact lock-file hashes, design hash,
and base/per-control seeds. A cached one-checkpoint smoke run exercised the
schema and resolved both model and tokenizer to the same immutable commit.

### F4. Synthetic recovery is circular validation

**CONFIRMED.** `experiments/synthetic_connection_recovery.py` generates targets
with the same `analogy` machinery later used by `evaluate_quadrilateral` to
score recovery. This is a useful regression smoke test. It is not independent
validation of the geometry or evidence that the method recovers a real
semantic operation.

**RESOLVED AS A VALIDATION CLAIM.** The script now generates mixture,
exponential, and Fisher targets from independent closed-form formulas and labels
the output a synthetic regression rather than semantic evidence.

### F5. Float32-weight hypothesis was attacked and cleared

**ATTRIBUTED / CLEARED.** Opus 5 reports that using float32 model weights at
condition number roughly `4.2e4` changed sectional curvature by only
`1.6e-08`. The exact probe is not committed, but this is consistent with the
independent finding that archived deterministic quantities reproduce closely.
There is no present evidence that float32 weights explain the pilot effects.

### F6. Relation of new numerical failures to old pilot numbers

**CONFIRMED.** Actual archived prompt pairs had accurate ordinary-regime
log-exp reconstruction, so the large reported composition errors are not
artifacts of D5. However, the distance noise floor is on the same scale as
some reported near-zero orientation closures. Treat those near-zero values as
numerical identity checks, not model measurements.

---

## G. Statistics

### G1. The planned replication structure is not yet confirmatory

**CONFIRMED.** The protocol requests at least five Pythia seeds and at least 30
base prompts for confirmation, while the distillation plan explicitly calls
four seeds preliminary. Even five training seeds may be weak for heterogeneous
arm-by-template interactions. The repository contains no simulation-based
power or interval-width analysis showing what effects this design can resolve.

**DESIGN RULE RESOLVED; NUMERICAL POWER OPEN.** The fixed five-seed/30-prompt
rule has been removed. Confirmation now requires a simulation-based count with
at least 80% power after multiplicity correction. The calculation cannot be
instantiated before engineering-pilot variance components exist.

### G2. More random planes do not create more scientific replicates

**CONFIRMED.** Moving from 8 to 999 spectrum-matched controls improves Monte
Carlo tail-rank resolution under the conditional group-invariance null. It does
not create independent model seeds, prompts, operations, or training runs.
The plane randomization p-value is valid only under the explicitly declared
conditional invariance null; otherwise it is descriptive.

### G3. Missing frozen numerical decision rules

**CONFIRMED.** The protocol says to preregister practical significance and
matched-NLL/Pareto rules but does not provide concrete values for:

- minimum behavioral effect size;
- NLL-match tolerance;
- minimum packet-acceptance coverage;
- treatment/penalty for infeasible transport;
- maximum tolerated chart shrinkage;
- acceptable certificate looseness; or
- stopping and exclusion rules for failed model seeds.

Without numerical values frozen before seeing results, the “positive result”
criteria retain analyst degrees of freedom.

**RESOLVED.** The protocol now freezes 90% overall/80% per-stratum packet
coverage, 0.01 nats/token NLL matching, a 0.20 baseline-seed-SD behavioral
effect with positive lower 95% interval, Fisher-alignment singular values in
`[0.5,2]`, at most 10% relative transport-certificate looseness, at least 90%
of D1 intervention length per stratum, a two-percentage-point feasibility
noninferiority margin, and intention-to-treat handling of failed seeds.

### G4. Multiplicity is only partially addressed

**CONFIRMED.** Holm adjustment is discussed for a narrow curvature family, but
the full program contains H1-H6, multiple checkpoints, contexts, operations,
outcomes, connection choices, D0-D6/Jz/Jpsi arms, banding sensitivities, and
behavioral metrics. The confirmatory hierarchy and gatekeeping order are not
fully frozen.

**RESOLVED FOR THE CONFIRMATORY DISTILLATION CLAIM.** Behavioral transfer,
transport defect, and feasibility are gatekept in that order; D3--D2 and
D4--D3 use Holm familywise 0.05, and remaining comparisons are exploratory or
falsification checks.

### G5. Prompt generalization lacks a sampling frame

**CONFIRMED.** The templates are fixed/hand-selected, but the population to
which a template bootstrap is supposed to generalize is not defined. A
bootstrap cannot manufacture population generality from a convenience sample.
Define the prompt/operation frame, selection process, and whether inference is
finite-population or superpopulation.

**CLAIM BOUNDARY RESOLVED; MANIFEST OPEN.** The protocol now limits the
historical prompts to finite convenience-set inference and blocks confirmation
until a versioned manifest records the target frame, selection mechanism, and
split. It forbids superpopulation language absent a probability sample. The
manifest itself remains a study input, not something the code audit can invent.

---

## H. Reproducibility and software engineering

### H1. What currently works

**CONFIRMED.** On the audited checkout:

- 79 unit tests pass;
- Ruff passes;
- the synthetic scripts run in the ordinary tested regimes;
- installed packages satisfy `pip check`; and
- static TeX citation/reference scans found no missing citation keys,
  undefined references, duplicate labels, or unused bibliography entries.

No LaTeX engine is installed, so the manuscript could not be rebuilt or
visually inspected from source in this environment.

### H2. Environment is not pinned

**CONFIRMED.** `requirements.txt` contains only `numpy>=2.0`.
`requirements-model.txt` contains lower bounds for Transformers and
Safetensors and instructs the user to install CPU PyTorch separately. There is
no lockfile, `pyproject.toml`, `environment.yml`, container definition, or CI
workflow. The current working environment is not the same thing as a
reproducible environment.

**RESOLVED.** `pyproject.toml`, exact core and CPU-model lock files, and a
Windows/Linux GitHub Actions workflow are now present. Editable packaging,
`pip check`, and the locked local versions were validated.

### H3. Repository packaging/governance is incomplete

**CONFIRMED.** No LICENSE is present, so reuse rights are unclear. There is no
package metadata or automated continuous-integration record. These do not
affect theorem truth, but they materially weaken an external reproducibility
claim.

**PARTIALLY RESOLVED.** Package metadata and CI configuration are present. No
software license has been selected; only the author can make that legal choice,
so the README explicitly denies an implied reuse grant until a license is added.

---

## I. Public claims

### I1. LC length distortion is an identity check, not an empirical discovery

**CONFIRMED.** Levi--Civita parallel transport is metric compatible by
definition. Reporting mean absolute log-length distortion near machine zero is
a valuable implementation validation. Placing it beside empirical composition
errors without labeling it as an identity check risks implying that the model
was observed to prefer LC length preservation.

### I2. Exponential orientation closure is also structurally forced here

**CONFIRMED.** At an affine softmax head, exponential coordinates are flat and
the two additive composition orders commute. Near-zero exponential orientation
closure in the pilot is therefore another implementation identity, not evidence
that training learned semantic commutativity. Mixture closure is likewise flat
when both compositions remain feasible. Fisher/LC path dependence is the
nontrivial geometric quantity.

### I3. Recommended public separation

Every summary should use three explicitly labeled lists:

1. **Mathematical identities:** LC length preservation, affine-head
   exponential flatness/closure, categorical curvature identities.
2. **Numerical validation:** residuals, reconstruction accuracy, tests, stencil
   convergence, and identity errors.
3. **Empirical measurements:** held-out behavior, composition error, curvature
   on preregistered planes, cross-model/seed effects, and uncertainty.

The current pilot has items in the first two lists. It does not yet have a
confirmatory item in the third.

**RESOLVED IN PUBLIC TEXT.** README, overview, and protocol now use this
separation and label LC length preservation and flat-connection closure as
implementation-identity checks.

---

## J. Disposition of the pre-run priority order

1. **Complete:** D1--D9 code repairs and adversarial boundary behavior.
2. **Complete:** explicit sufficiency estimands and the repaired Sobolev audit.
3. **Complete:** all Section E regressions and local CI-equivalent validation.
4. **Source complete / build blocked:** related work, proof completeness, and
   finite-scale disclosure are repaired; the PDF still needs a LaTeX engine.
5. **Complete for new artifacts:** environment, model snapshots, packet schema,
   and result schema are versioned and hashed.
6. **Specified / inputs open:** multilevel estimand, practical thresholds, and
   multiplicity are frozen; the prompt manifest and power calculation await
   study inputs.
7. **Next scientific step:** only after those inputs exist, run the multi-seed
   continuous-tangent behavioral experiment.

## Final disposition

- **Mathematics:** no defect found; all formal results now have explicit proofs;
  independent certification still absent.
- **Paper:** serious theory draft; source repairs are complete, but the stale PDF
  still prevents submission-ready status.
- **Code:** the confirmed silently-wrong-number paths are repaired and covered by
  adversarial regression tests; prospective distillation results still require
  the unimplemented real-model pipeline.
- **Experiments:** execution-only pilot; no substantive semantic evidence.
- **Statistics:** numerical decision rules are specified, but confirmatory power
  and population inference remain blocked on pilot variance inputs and a frozen
  prompt manifest.
- **Research value:** not destroyed. The theoretical program survives, but the
  empirical and distillation claims remain prospective until the preregistered
  study actually runs.
