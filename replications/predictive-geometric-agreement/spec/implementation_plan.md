# Implementation plan

All independent runner code lives under `code/`, configuration under `config/`,
and generated evidence under `artifacts/`. Existing implementation files in the
repository root are reference-prohibited by the default author-code policy.

## Execution order

1. `PGA-01` regular KL-to-geometry counterexample.
2. `PGA-02` boundary/nonzero-alpha counterexample.
3. `PGA-03` saturated-simplex curvature.
4. `PGA-04` exact naturality audit.
5. `PGA-05` finite-scale stability audit.
6. `PGA-06` risk-to-transport audit.
7. `PGA-07` semantic obstruction examples.
8. `PGA-08` independent Pythia diagnostic rerun.

Exactly one matrix row remains `ACTIVE`. Each runner must be invoked through
`track-run`; its output is then bound to code, configuration, seed, and this
specification through `register-target-artifact`. Comparison evidence is
recorded before a row moves to `MATCHED`.

## Dependencies and gates

- `PGA-01` through `PGA-07`: repository Python environment and NumPy only.
- `PGA-08`: the separately locked model environment, network access for an
  immutable model snapshot, and CPU runtime; no cached author result may count
  as candidate evidence.
- Numeric runners exit nonzero when their predeclared tolerance fails.
- Structural proof audits emit a machine-readable obligation list and cannot
  pass with an unresolved load-bearing step.
- The report must describe deviations and unmatched targets; the whole-paper
  completion gate remains false until every row is independently matched.
