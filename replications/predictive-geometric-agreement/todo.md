# TODO

## Current phase
independent-replication

## Active target
PGA-02

## Acceptance gates
- [x] Inventory the LaTeX source and paper assets.
- [x] Enumerate the load-bearing claim groups in `spec/reproduction_matrix.csv`.
- [x] Declare a claim-specific acceptance mode for every target.
- [x] Replace the scaffold placeholders in `spec/*.md`.
- [x] Keep exactly one ACTIVE target until all rows are terminal.
- [x] Register the PGA-01 artifact through the run wrapper and provenance files.
- [x] Register PGA-01 with paper-method evidence (`code_path`, `config_path`, `paper_trace_path`, `method_components`, `implementation_summary`).
- [x] Add PGA-01 comparison evidence and report coverage.
- [ ] Apply the same evidence chain to every remaining claimed artifact.
- [ ] `validate-completion` passes only when the whole paper is genuinely complete

## Open unknowns
- The proof-chain audit granularity may expand if a supporting lemma has an
  independent failure mode.
- The immutable model snapshot required by PGA-08 may require a network-enabled
  model environment.

## Completed
- Bootstrap completed.
- Paper source inventory and hash index completed.
- Windows-safe PowerShell run recording added and attributed.
- PGA-01 matched: exact Fisher constants agree within `3.5e-18`, the fitted
  uniform-grid KL slope is `-1.9979`, and the nonzero metric gap is retained.
