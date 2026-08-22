# Math audit

The manuscript treats a smooth predictive map `p: M -> int(Delta)` as the
primary object. Pulling back categorical Fisher geometry gives

`g_ij = sum_a (partial_i p_a)(partial_j p_a)/p_a`

and the Amari--Chentsov cubic is the corresponding third score moment. A
declared alignment must compare predictive images and tangent maps; raw hidden
coordinates are not treated as identified.

## PGA-01 trace

For `r(x)=1/3+x/20` and
`r_n(x)=r(x)+sin(nx)/(100n)`, all probabilities remain interior and

`r_n'(0)=3/50`, while `r'(0)=1/20`.

The Bernoulli pullback metric therefore gives

- `g_n(0)=81/5000`,
- `g_*(0)=9/800`.

The perturbation is uniformly `O(1/n)` and Bernoulli KL is locally quadratic on
the common interior range, so its uniform KL is `O(1/n^2)`. `PGA-01` checks the
two exact metric constants and estimates the rate on a dense fixed grid. The
grid is evidence for the rate, while the analytic interior/quadratic argument
remains the reason the supremum statement is valid.

## Remaining proof obligations

- `PGA-02`: derive the score values and all connection coefficients directly
  from the Bernoulli angle parameterization.
- `PGA-03`: derive the radius-two sphere isometry and distinguish ambient
  Euclidean flatness from intrinsic sphere curvature.
- `PGA-04`: check every naturality statement under an explicit diffeomorphism
  and identify where constant rank is used.
- `PGA-05`: trace every tensor norm, inverse-metric factor, and transport
  propagator in both exponential and metric-compatible branches.
- `PGA-06`: verify the Hellinger--KL, interpolation, and Hilbert-valued embedding
  exponents and their dimension dependence.
- `PGA-07`: check that conditional variance is informative only after a genuine
  noninjective predictive quotient is declared.
- `PGA-08`: keep pointwise spectral diagnostics separate from the uniform
  pathwise hypotheses required by the finite-scale theorem.

Finite differences, regression fits, or numerical quadrature are supporting
checks only. They do not replace proof obligations stated above.
