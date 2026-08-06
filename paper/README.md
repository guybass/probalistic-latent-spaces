# Paper

`main.tex` is the arXiv-style theory manuscript:

> **From Predictive Agreement to Geometric Agreement: Stability of Fisher Connections and Semantic Transport in Neural Representations**

The paper makes four narrow claims:

1. predictive-map agreement propagates quantitatively to Fisher metrics, Amari alpha-connections, parallel transport, and curvature under explicit regularity assumptions;
2. cross-entropy convergence alone does not imply geometric convergence;
3. cross-entropy plus uniform Sobolev regularity implies parallel-transport convergence at an explicit rate;
4. conditional Fisher variance is the irreducible obstruction to representing a semantic operation as a field on the predictive quotient.

It deliberately does **not** claim the first probabilistic latent geometry, Fisher pullback, semantic vector field, or representation holonomy.

## Build

From this directory, run a LaTeX engine with BibTeX support. For example:

```powershell
tectonic main.tex
```

The verified repository PDF is written to `../output/pdf/predictive_geometric_agreement.pdf`.
