# Predictive Geometric Agreement

This is an independent, LaTeX-first replication workspace for the paper in
`../../paper/main.tex`. It is the first repository connection for the
Paper-replication Codex harness.

The source paper and its assets are reference-only. The default policy forbids
using this repository's existing author implementation under `../../src/`,
`../../experiments/`, or `../../tests/` as replication evidence. New independent
implementations belong inside this case study. See `UPSTREAM.md` for provenance,
scope, and the Windows compatibility patch.

## Core commands

```powershell
..\..\.venv\Scripts\python.exe scripts\paper_harness.py status --project-dir .
..\..\.venv\Scripts\python.exe scripts\paper_harness.py inspect-paper --project-dir .
..\..\.venv\Scripts\python.exe scripts\paper_harness.py validate-completion --project-dir .
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\reproduce_all.ps1
```

## Expected workflow

1. Inspect the paper sources and inventory the TeX tree.
2. Fill out the spec files under `spec/`.
3. Keep `spec/reproduction_matrix.csv` and `todo.md` in sync, including per-target `acceptance_mode`.
4. Mark exactly one ACTIVE target at a time.
5. Use `track-run`, `register-target-artifact`, and comparison evidence before moving any target to MATCHED.
6. Register baseline targets only after a real paper-method implementation exists under project code, config, and spec trace files.
7. Use exact visual matching only for targets that truly require it; convergence curves and other stochastic summaries should usually use numeric or structural acceptance.
8. Move a target to MATCHED only after the artifact exists, provenance is recorded, and `report/main.tex` embeds it.
9. Treat the paper as done only when `validate-completion` passes.

The scaffold is intentionally incomplete: no scientific claim is matched merely
because the harness exists or its structural checks run.

## Case study id

- Paper slug: `predictive-geometric-agreement`
