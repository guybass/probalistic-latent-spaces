# Targets

The replication unit is a load-bearing claim group, not every displayed
equation. Supporting lemmas are audited with the theorem chain that consumes
them. The matrix currently records eight groups:

1. `PGA-01`: independently evaluate the regular Bernoulli construction. Verify
   the reported Fisher metrics at zero and the uniform `O(n^-2)` KL rate.
2. `PGA-02`: verify the boundary Bernoulli identities `g=1`,
   `Gamma^LC=0`, `C=2 cot(theta)`, and divergence of every fixed nonzero-alpha
   coefficient.
3. `PGA-03`: recover sectional curvature `1/4` for the saturated categorical
   simplex, including an approach-to-boundary stress sequence.
4. `PGA-04`: audit exact predictive naturality and intrinsic-field descent,
   including the stated quotient/rank assumptions.
5. `PGA-05`: audit the probability-coordinate and square-root stability chains,
   with special attention to norm alignment, index raising, and rank factors.
6. `PGA-06`: audit the KL-plus-Sobolev interpolation and embedding chain used to
   obtain transport convergence.
7. `PGA-07`: independently construct injective and noninjective examples for the
   semantic conditional-variance obstruction.
8. `PGA-08`: rerun the Pythia-14M pointwise conditioning diagnostic from model
   weights and compare the reported spectral ranges and inverse powers.

The manuscript contains no external result-figure assets. Its one numerical
display is the inline Pythia diagnostic table in Appendix A.4. Acceptance rules
were fixed in `reproduction_matrix.csv` before candidate artifacts were created.

Ambiguities that can change a scientific conclusion are recorded in
`assumptions_and_unknowns.md`; they may not be silently resolved inside runner
code.
