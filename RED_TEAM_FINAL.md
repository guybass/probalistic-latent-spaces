# Final repository red-team audit

**Audit date:** 17 August 2026  
**Scope:** current working tree, including uncommitted and untracked files  
**Method:** theorem/proof and reference audit, independent numerical probes,
schema-tampering probes, boundary and scale attacks, test/CI execution, artifact
inspection, protocol/statistical review, and full visual rendering of the shipped PDF.

## Executive verdict

The mathematical program survives this pass. I found no new false theorem, wrong
constant, or sign error in `paper/main.tex`. The current source contains 21 formal
results and 21 proofs, with no missing references or citations.

The packet implementation does **not** yet survive. Three high-severity defects can
produce or preserve an accepted but scientifically invalid training packet:

1. commensurate finite-difference stencils can alias a smooth logit field and accept
   tensors that are wrong by orders of magnitude;
2. float32 serialization can turn an accepted SPD metric into a singular stored
   metric while deserialization still reports `accepted=True`; and
3. a packet can sample outside its declared chart bounds and still be serialized as
   reproducible.

There are also two integrity gaps, two Fisher-simplex domain defects, a contradiction
in the confirmatory multiplicity plan, and reproducibility/editorial blockers. The
100-test suite is green because none of the new attacks is represented.

**Release recommendation:** theory manuscript: conditionally defensible after a fresh
TeX build and citation correction. Packet/training artifact pipeline: **no-go** until
D1-D3 and C1 are fixed and regression-tested. No empirical connection-distillation
claim is currently supported, which the repository mostly states correctly.

---

## A. Mathematics

### A0. Cleared

- 21 theorem/proposition/corollary environments and 21 proof environments.
- 76 labels; no duplicates and no undefined `ref`/`eqref` targets.
- 20 bibliography entries; every key is cited and every citation key exists.
- The two-model convergence corollary now correctly uses the operator-norm triangle
  inequality through the common limiting transport, not a nonexistent KL triangle
  inequality.
- The Duhamel signs in `paper/main.tex` and
  `PREDICTIVE_CONNECTION_DISTILLATION.md` agree with their respective definitions of
  the coefficient matrix.
- The square-root LC bound still exposes the claimed lambda inverse powers; the text
  correctly refuses to instantiate a certificate from a spectral floor alone.
- Randomized moderate-interior simplex checks (500 cases, dimensions 2-19) gave:
  maximum Fisher `Exp(Log)` absolute error `3.61e-16`, maximum LC-transport isometry
  relative error `3.35e-15`, and no failed identity check.

### A1. No new mathematical defect found

This is not a machine-checked proof and not a substitute for referee review. It is a
negative red-team result: I found no counterexample to the stated theorems or their
displayed constants in this pass.

---

## B. Manuscript and bibliography

### B1. [Medium] One venue record is not adequately supported

`paper/references.bib:194-202` records Hu, Niu, and Varma as a main-conference ICML
2026 paper in PMLR volume 306, and `paper/README.md:36` says that venue was verified.
The arXiv record exposes no journal reference, while the first author's publication
page identifies the work as an **ICML Mechanistic Interpretability workshop
spotlight**. The official ICML 2026 downloads index contains the Park et al. paper but
does not return this title.

Until an actual PMLR landing page is located, use `@misc` with the arXiv identifier
and an accurately qualified workshop note. Do not claim main-proceedings publication
from a conference-formatted preprint alone.

Primary records:

- https://arxiv.org/abs/2607.04525
- https://paleuna.github.io/
- https://icml.cc/Downloads/2026

### B2. [Release blocker] The shipped PDF is stale

`output/pdf/predictive_geometric_agreement.pdf` was created on 8 August 2026; the
current `paper/main.tex` was modified on 17 August. The 23-page PDF renders without an
obvious clipping or collision defect, but it omits the two newest references and at
least some later proof/remediation text. No TeX engine is installed, so the current
source has not been compiled.

### B3. [Minor] Submission checklist language is stale

`paper/README.md:36` says “the four 2026 entries,” while the bibliography now contains
six 2026 entries. More importantly, “all ... venues verified” is too strong until B1
is resolved.

---

## C. Protocol and statistics

### C1. [High for a confirmatory claim] Success criteria outrun multiplicity control

The protocol says a positive result must beat D1, Jz, Jpsi, and D2
(`EXPERIMENT_PROTOCOL.md:637-639`; PCD `:932`) and, when run, D5 (PCD `:949`). But the
frozen familywise procedure covers only D3-D2 and D4-D3, and explicitly labels every
other comparison exploratory (`EXPERIMENT_PROTOCOL.md:721-725`). A confirmatory
positive claim therefore depends on comparisons that the same protocol calls
exploratory.

Choose one coherent rule before data inspection. For example, define a gatekept
family containing every contrast required for the final claim, or narrow the final
claim to the two Holm-controlled contrasts. The behavioral/transport/feasibility
conjunction should also be stated explicitly as an intersection-union rule or given a
separate endpoint multiplicity procedure.

### C2. [High] “Independent” finite-difference validation is not independent enough

The protocol freezes only the nested grid `(h, h/2, h/4)` and calls the JVP comparison
independent. D1 below shows that a smooth signal can vanish on every point in that
grid. A refinement plateau on commensurate stencils is evidence of self-consistency,
not a certificate of derivative accuracy.

Require exact autodiff derivatives for production packets where possible, use them to
form the metric and score moment directly, and audit with at least one preregistered
incommensurate or randomized step. If finite differences remain primary, state a
bandwidth/derivative-envelope assumption; no finite stencil family certifies an
arbitrary smooth black-box function.

### C3. [Medium] The nonzero-tolerance sensitivity protocol cannot be represented

The protocol requires a tensor-specific recorded roundoff derivation for any nonzero
absolute floor. The packet stores one shared `refinement_atol` for `G`, `L`, and `C`,
only the worst aggregate diagnostics, and no calculation/source field. The primary
zero floor is represented correctly; the advertised sensitivity analysis is not.

### C4. Cleared

- Independent units, crossed/nested prompt factors, seed-first resampling,
  intention-to-treat failures, packet-coverage floors, intervention-shrinkage gates,
  and the simulation-based power requirement are now stated carefully.
- The historical pilot is explicitly execution-only and is not presented as semantic
  evidence.
- The confirmatory study remains honestly blocked on a versioned prompt/operation
  manifest, variance components, and a simulation-derived seed count.

---

## D. Code and numerical correctness

### D1. [High] Nested-stencil aliasing accepts grossly wrong geometry

Relevant code: `src/predictive_geometry/distillation.py:265`, `:399`, and `:428`.

Use a one-dimensional three-logit family

```text
l(z) = W z + a sin(8 pi z / h) v,
W = (0, 1, 3), v = (1, -0.2, -0.8), h = 0.02, a = 0.001.
```

The oscillatory term vanishes at every central point used by `h`, `h/2`, and `h/4`.
With no supplied JVP, the packet is accepted with:

```text
refinement_tolerance_ratio = 2.4131e-05
packet metric              = 1.5555555544
analytic metric            = 0.2617012034
relative metric error      = 4.9440       (494%)
packet cubic               = 0.7407407407
analytic cubic             = 0.0296561081
```

A deliberately wrong declared JVP equal to `W` also passes with JVP tolerance ratio
zero, because the audit reuses the same aliased finest step. This is not merely
truncation error: it is exact sampling alias.

Fix: compute `G` and `C` from genuine autodiff logit JVPs; use HVPs/autodiff or an
incommensurate derivative audit for `L`; disallow production packets without a
noncommensurate or exact-derivative path. Add this exact adversarial fixture.

### D2. [High] Float32 round-trip can make an accepted metric singular

Relevant code: `distillation.py:1385-1398` and `:1534-1538`.

An accepted source metric

```text
[[1, 1],
 [1, 1 + 4e-10]]
```

has `lambda_min = 2.000000165e-10`, above the default `1e-10` metric floor and
relative-conditioning gate. Float32 serialization stores `[[1,1],[1,1]]`, whose
minimum eigenvalue is zero. Deserialization nevertheless returns `accepted=True`
because it checks the stored pre-quantization eigenvalue against the singular matrix
using the quantization error as tolerance, then gates the stored eigenvalue rather
than the actual matrix.

Fix: recompute eigenvalues from the serialized float32 metric and apply every rank,
conditioning, and scale gate to those values. Stronger: require
`lambda_min(source) - ||Delta G||_2` to clear the floor. Otherwise retain the metric
in float64.

### D3. [High] Reproducible packets may use out-of-chart stencil points

Relevant code: `distillation.py:399` and `:1312`.

With declared bounds `[0,1]`, center `z=0.01`, and `h=0.02`, the builder evaluates
the teacher at `z=-0.01`. The central point is inside the chart, so the accepted
packet serializes with `reproducible=True`. The theorem's declared compact domain and
the packet's provenance therefore do not cover the derivative calculation.

Fix: before any teacher call, require every axial and mixed stencil point at every
level to lie within `chart_bounds`; otherwise reject with a distinct `chart_boundary`
reason or use a declared one-sided boundary stencil with its own error analysis.

### D4. [Medium] Rejected-packet semantics can be forged

Relevant code: `distillation.py:1324-1327`.

Semantic validation returns immediately for every rejected packet. Replacing an
actually accepted full-rank packet with `accepted=False, rejection_reason="rank"`
survives checksumming, serialization, and deserialization even though its recorded
eigenvalue clears the rank gate. This permits corrupted rejection-category and
coverage summaries to look schema-valid.

Fix: recompute the gate result and ordered rejection reason from the stored fields for
both accepted and rejected packets. For reasons that require unavailable refinement
levels, store the level summaries needed to verify them.

### D5. [Medium] The promised rejection audit channel is incomplete

The builder raises before constructing a packet for invalid/underflowed simplex
values. Strict serialization also refuses nonfinite rejected packets and refers to a
“separate audit channel,” but no such schema or writer exists. A pipeline that counts
only serialized packets can silently omit exactly the catastrophic failures required
in the acceptance denominator.

Fix: add a versioned `PacketFailureAudit` record containing provenance, center, step,
attempted stencil coordinates, failure class, and safe finite diagnostics. Test that
every attempted central point yields exactly one success or failure record.

### D6. [Medium] Fisher exponential accepts a path that leaves the manifold

Relevant code: `src/predictive_geometry/simplex.py:133-145` and `:179`.

At `p=(0.5,0.5)`, `u=(2 pi,-2 pi)`, `fisher_exp(p,u)` returns `p` after a full sphere
rotation. The intervening great-circle path crosses out of the positive square-root
orthant, so the Riemannian exponential on the open simplex is no longer defined.
Endpoint-only positivity is insufficient.

Fix: compute the first positive-orthant boundary time along the sphere geodesic and
reject any requested unit-time path that reaches it.

### D7. [Low/medium] Tiny valid Fisher operations are misclassified or erased

Relevant code: `simplex.py:120` and `:138`.

For `p=(0.5,0.5)` and a representable `q-p` of about `1e-15`, `fisher_log` raises
“undefined at antipodes” because `sin(theta)<1e-14`, even though the points are nearly
identical. For `u=(3e-15,-3e-15)`, `fisher_exp` silently returns `p`, losing a
representable tangent.

Fix: distinguish `theta -> 0` from `theta -> pi`; use stable `theta/sin(theta)` and
`sinc` expansions. Only the exact zero tangent should return unchanged.

### D8. Cleared

- Jacobian outcome-axis centering, unbiased sufficiency estimand, dimensionless
  conditioning gate, corrected relative refinement diagnostic, provenance checksum,
  Duhamel sign, tensor unfolding, transport bound growth factor, Haar QR correction,
  curvature assembly, and score-moment cubic all survived this pass.
- The three synthetic drivers complete successfully.

---

## E. Tests and CI

### E1. Current results

- `100` unit tests: pass.
- Ruff `0.14.2`: pass.
- `pip check`: pass.
- Python compilation of `src`, `tests`, and `experiments`: pass.
- `git diff --check`: pass, apart from line-ending conversion warnings.

### E2. [High coverage gap] Missing adversarial test classes

Add tests for:

1. an oscillation vanishing simultaneously on `h`, `h/2`, and `h/4`;
2. post-quantization SPD/rank/conditioning gates;
3. full-stencil containment in chart provenance bounds;
4. recomputation of rejected-packet reasons;
5. one failure-audit record per attempted packet;
6. LC exponential path containment, not endpoint containment;
7. representable near-zero `Log`/`Exp` behavior.

### E3. [Medium reproducibility gap] CI does not test the claimed support/locks

`.github/workflows/ci.yml` tests only Python 3.12 although `pyproject.toml` declares
Python >=3.10. CI installs open constraints via `pip install -e ".[test]"` and ignores
both exact lock files. The locks also pin versions but not artifact hashes.

Fix: test at least 3.10 and 3.12 (or narrow `requires-python`), add one locked job, and
generate hash-locked requirements for release reproduction.

---

## F. Empirical artifacts and reproducibility

### F1. [Medium] Control seeds depend on a symbolic request, not the resolved model

`stable_control_seed` hashes the requested revision (`pythia_cpu_smoke.py:193,255`),
while the immutable commit is discovered later (`:236`). Rerunning a result from its
recorded commit instead of the original symbolic tag changes every control draw. The
per-design seed is recorded but the CLI cannot replay it directly.

Fix: resolve first and derive seeds from the resolved model commit, tokenizer commit,
design hash, and control definition version. Alternatively support a replay manifest
containing the recorded per-design seeds.

### F2. [Low] Threading is used but not recorded

`--threads` changes Torch execution (`pythia_cpu_smoke.py:525,570`) but is absent from
the result payload. Record Torch and BLAS thread counts and deterministic-algorithm
settings.

### F3. Standing blockers, correctly disclosed

- All four committed Pythia JSON files are legacy unversioned artifacts; the results
  README correctly labels them as one model run and execution-only.
- No current `pythia-smoke-2` result is committed.
- Real-model packet generation, student training, behavioral evaluation, the prompt
  manifest, and the confirmatory power simulation are not implemented.
- The current checkout is dirty. The corrected driver therefore refuses a scientific
  artifact unless the nonreproducible override is used.
- Most remediation is uncommitted, and `.github/`, `pyproject.toml`, both lock files,
  and the Round-5 report are untracked. This is not a releasable immutable state.
- No software license exists. README correctly says source availability is not reuse
  permission.

### F4. Cleared

- The installed Python 3.12 environment matches the pinned NumPy, Torch CPU,
  Transformers, Safetensors, and Ruff versions.
- The driver records repository/driver/lock hashes, runtime versions, immutable model
  and tokenizer revisions, tokenizer/outcome hashes, and distinct model/control seed
  semantics.
- The archived artifacts parse as strict JSON and are not silently upgraded to the
  new schema.

---

## G. Public claims

### G1. [Must revise before release] Packet-core validation is overstated

README lines 47-50 and 72-75 call the schema validated and the JVP/refinement audits
independent. D1-D5 refute that release-level wording. Replace it with “model-free
prototype with passing regression tests” until the high-severity findings close.

### G2. Cleared

- The pilot is not presented as semantic evidence.
- LC length preservation and flat-connection closure are correctly identified as
  implementation identities, not empirical measurements.
- Real-model distillation remains explicitly prospective.
- The manuscript limitations correctly separate vocabulary independence from a
  numerically useful finite-scale certificate.

---

## Ordered remediation

1. **D1:** exact/autodiff packet geometry plus noncommensurate derivative audit.
2. **D2:** gate the actual serialized metric (or serialize it in float64).
3. **D3:** enforce full-stencil chart containment.
4. **C1:** make the confirmatory contrast family match the claimed success rule.
5. **D4-D5:** recomputable rejection semantics and a complete failure-audit schema.
6. **D6-D7:** correct Fisher exponential-domain and small-angle handling.
7. **F1/E3:** immutable seed derivation, locked multi-version CI, recorded threading.
8. **B1-B2:** correct the venue record and rebuild/inspect the PDF.
9. Commit the complete candidate state, rerun all probes from a clean checkout, and
   only then regenerate any scientific artifact.

## Bottom line

The six months of mathematics are not on the floor. Theorems and proof structure
survived. The dangerous layer is the numerical packet certification: it can currently
accept aliased geometry, preserve a singular serialized metric as accepted, and claim
chart provenance that does not cover its own stencil. Those failures are fixable, but
they are release blockers because they attack the exact bridge from the sound theory
to future empirical claims.
