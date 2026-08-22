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
- [x] An engine-guarded `\pdfoutput=1` is in the first five preamble lines, so arXiv's pdfLaTeX detection is preserved while the documented Tectonic/XeTeX build remains valid.
- [x] MSC 2020 classes (53B12, 62B11, 68T07) and keywords after the abstract.
- [x] Abstract is 1,648 characters, under arXiv's 1,920-character metadata limit; no display math in it.
- [x] All bibliography entries verified to exist with matching titles, authors, and publication status, including all six 2026 entries. arXiv:2602.15293 records an ICML 2026 journal reference; arXiv:2601.21653 is accepted at ICLR 2026; arXiv:2605.17231, arXiv:2607.04525, arXiv:2602.02315, and arXiv:2603.22301 are cited as preprints. No venue is inferred merely from an arXiv posting.
- [x] The code-availability URL matches the actual git remote (`probalistic-latent-spaces` is the repository's real spelling).
- [x] A clean Tectonic build generated `main.bbl` and a 31-page PDF. The final log has no undefined references, undefined citations, duplicate-label warnings, overfull boxes, or TeX errors; all 31 rendered pages were visually inspected.
- [x] The final source audit found 98 unique labels, 149 resolved references, 20 cited bibliography entries with no missing or unused keys, and 36 balanced proof environments.
- [x] The verified release PDF is copied to `../output/pdf/predictive_geometric_agreement.pdf`; `main.bbl` is retained beside the source for arXiv submission.

**To do at submission time**

- [ ] **Category.** Recommended: primary `cs.LG`, cross-list `math.DG` and `stat.ML`. The closest related work (softmax information geometry, representation holonomy, latent-geometry alignment) all sits in `cs.LG`. A `math.ST`-primary submission is defensible if a math-side endorsement is easier to obtain, but expect moderators to add the `cs.LG` cross-list either way.
- [ ] **Endorsement.** A first submission in `cs.LG` may require endorsement from an eligible `cs.LG` author. The endorsement code is issued during submission; any of the cited authors' groups or a colleague with prior `cs.LG` papers can endorse.
- [ ] **License.** The minimal arXiv non-exclusive license keeps the most options open for later journal submission; choose CC BY only if certain.
- [ ] **Metadata.** Enter title and abstract in plain text with `$...$` inline math only; author as "Guy Basson"; optionally add an `\thanks{}` contact email to the author line before building.
- [ ] **Submission-time rebuild.** Repeat the clean build immediately before upload so the submitted PDF and `main.bbl` are guaranteed to match the final source.
