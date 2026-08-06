# Paper

`main.tex` is the arXiv-style theory manuscript:

> **From Predictive Agreement to Geometric Agreement: Stability of Fisher Connections and Semantic Transport in Neural Representations**

The paper develops six connected results:

1. exact predictive-image equivalence preserves Fisher--Amari geometry and intrinsic predictive fields;
2. cross-entropy/KL convergence alone does not imply geometric convergence;
3. general alpha-transport is stable under explicit probability-coordinate and rank regularity;
4. square-root/Hellinger regularity gives vocabulary-independent Levi--Civita metric, transport, and curvature stability, with a metric-compatible transport bound that avoids unavoidable exponential growth; the same regularity provably fails for fixed nonzero-alpha connections unless the raised third score moment is also controlled;
5. integrated KL plus uniform Hilbert-valued Sobolev regularity implies Levi--Civita transport convergence without a token-probability floor;
6. conditional Fisher variance is the irreducible obstruction to descending a semantic operation through a noninjective predictive map and vanishes at an injective head.

## Build

From this directory, run a LaTeX engine with BibTeX support. For example:

```powershell
tectonic main.tex
```

After building and checking references, copy the verified PDF to `../output/pdf/predictive_geometric_agreement.pdf`.
