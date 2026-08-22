# Harness provenance and scope

## What was evaluated

The motivating preprint is Damon Falck et al., *Training AI Scientists to
Replicate Research*, arXiv:2608.13331v1 (13 August 2026). It reports three
relevant ideas:

- bounded figure-replication tasks rather than unconstrained paper generation;
- task-specific judging across visual fidelity, claim support, method fidelity,
  resource use, and scientific integrity;
- a simple agent harness with an auditable workspace and a coding agent as a
  tool.

The trained 27B Faraday model, Replica task set, judge implementation, and
training stack were not released with v1. They are therefore not vendored or
represented as available here. Reproducing the training recipe would also
require infrastructure far beyond this CPU-first project.

## Practical harness selected

The executable scaffold comes from the closely related, released
Paper-replication workflow by Atharva Hans and Ilias Bilionis:

- paper: arXiv:2607.02134v2, *Coding-agents can replicate scientific machine
  learning papers*;
- source: https://github.com/PredictiveScienceLab/paper-replication-paper;
- upstream license: Apache License 2.0;
- retrieved: 19 August 2026.

Its evidence contract is a good fit for this project: explicit targets,
single-active-target progress, immutable run records, code/config/source
provenance, claim-specific comparison evidence, and a fail-closed completion
gate. Those requirements complement the repository's existing numerical gates,
falsification criteria, and provenance checks.

The upstream Codex skill is installed at
`~/.codex/skills/paper-replication`. The bootstrap generated the files in this
case study. Upstream assumes `/bin/zsh` for tracked commands and generates Bash
entrypoints. This local copy selects PowerShell on Windows and adds
`scripts/reproduce_all.ps1`; scientific validation behavior is otherwise
unchanged.

## Connection policy

This workspace initially audits the project's own manuscript independently.
The paper source at `../../paper/` is reference material. Existing author code
is forbidden as baseline replication evidence by the manifest's default policy.
Changing that policy is a scientific-design decision, not a convenience switch.

The harness is ready for claim enumeration and independent reproduction work,
but it is not complete and makes no new scientific claim yet.
