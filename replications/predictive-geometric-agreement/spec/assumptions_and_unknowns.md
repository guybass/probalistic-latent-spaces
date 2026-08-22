# Assumptions and unknowns

## Fixed policies

- The manuscript under `../../paper/` is reference-only.
- Existing files under `../../src/`, `../../experiments/`, `../../tests/`, and
  `../../results/` cannot provide baseline replication evidence.
- The run environment is Windows/PowerShell and local CPU unless the manifest
  is explicitly revised.
- Numerical acceptance thresholds are fixed in the matrix and per-target config
  before a candidate run is inspected.

## Open scientific questions

- The paper groups a long stability chain into named theorems and deferred
  lemmas. The audit must decide whether each displayed constant is sharp,
  conservative, or merely sufficient without silently changing norms.
- `PGA-03` needs an independent coordinate chart and a boundary sequence that
  stays strictly inside the simplex.
- `PGA-05` must distinguish pointwise rank floors from a uniform floor along a
  path; a pointwise check cannot discharge the stronger hypothesis.
- `PGA-06` must verify the exact admissible Sobolev indices on bounded domains
  and compact manifolds, not only on a periodic box.
- `PGA-07` requires an explicitly declared noninjective quotient. Coarsening the
  outputs changes the predictive map and its image geometry and must be modeled
  as such.
- `PGA-08` depends on availability and immutable identification of the Pythia
  snapshot. If unavailable, the target remains unmatched with the transport or
  resource failure recorded.

## PGA-01 numerical convention

The grid calculation estimates the supremum-KL rate over `[-1,1]`; it is not
used as a proof that no between-grid maximum exists. Acceptance combines the
exact metric identities with a fitted log-log rate within the interval declared
in `config/pga-01.json`.
