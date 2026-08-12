# Paper

`main.tex` is the arXiv-style theory manuscript:

> **From Predictive Agreement to Geometric Agreement: Stability of Fisher Connections and Semantic Transport in Neural Representations**

The paper develops six connected results:

1. exact predictive-image equivalence preserves Fisher--Amari geometry and intrinsic predictive fields;
2. cross-entropy/KL convergence alone does not imply geometric convergence;
3. general alpha-transport is stable under explicit probability-coordinate and rank regularity;
4. square-root/Hellinger regularity gives vocabulary-independent Levi--Civita metric, transport, and curvature stability, with both metric-compatible and intrinsic mixed-norm transport bounds; paired Bernoulli maps show sharply that the same regularity fails for every fixed nonzero-alpha connection unless the raised third score moment is also controlled;
5. integrated KL plus uniform Hilbert-valued Sobolev regularity implies Levi--Civita transport convergence without a token-probability floor;
6. conditional Fisher variance is the irreducible obstruction to descending a semantic operation through a noninjective predictive map and vanishes at an injective head.

## Build

From this directory, run a LaTeX engine with BibTeX support. For example:

```powershell
tectonic main.tex
```

After building and checking references, copy the verified PDF to `../output/pdf/predictive_geometric_agreement.pdf`.

## arXiv submission checklist

Status of each requirement, verified August 2026:

**Done in the manuscript**

- [x] Conventional theorem/proof structure (`amsthm`), numbered environments, complete proofs, related-work section, limitations section.
- [x] `\pdfoutput=1` in the first five preamble lines, as arXiv's pdflatex detection requires.
- [x] MSC 2020 classes (53B12, 62B11, 68T07) and keywords after the abstract.
- [x] Abstract is 1,648 characters, under arXiv's 1,920-character metadata limit; no display math in it.
- [x] All bibliography entries verified to exist with matching titles, authors, and venues, including the four 2026 entries (arXiv:2602.15293 published at ICML 2026 PMLR; arXiv:2607.04525 published at ICML 2026 PMLR 306; arXiv:2601.21653 accepted at ICLR 2026; arXiv:2605.17231 preprint).
- [x] The code-availability URL matches the actual git remote (`probalistic-latent-spaces` is the repository's real spelling).

**To do at submission time**

- [ ] **Include `main.bbl`.** arXiv does not run BibTeX. Build locally, then upload `main.tex` together with the generated `main.bbl` (with tectonic: `tectonic --keep-intermediates main.tex`). Submitting only `main.tex` + `references.bib` will fail to produce citations.
- [ ] **Category.** Recommended: primary `cs.LG`, cross-list `math.DG` and `stat.ML`. The closest related work (softmax information geometry, representation holonomy, latent-geometry alignment) all sits in `cs.LG`. A `math.ST`-primary submission is defensible if a math-side endorsement is easier to obtain, but expect moderators to add the `cs.LG` cross-list either way.
- [ ] **Endorsement.** A first submission in `cs.LG` may require endorsement from an eligible `cs.LG` author. The endorsement code is issued during submission; any of the cited authors' groups or a colleague with prior `cs.LG` papers can endorse.
- [ ] **License.** The minimal arXiv non-exclusive license keeps the most options open for later journal submission; choose CC BY only if certain.
- [ ] **Metadata.** Enter title and abstract in plain text with `$...$` inline math only; author as "Guy Basson"; optionally add an `\thanks{}` contact email to the author line before building.
- [ ] **Final local compile check.** No LaTeX engine is installed on this machine; before submission run the build on a machine with tectonic or TeX Live and confirm zero errors and no missing-reference warnings.
