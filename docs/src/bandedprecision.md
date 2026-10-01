# Banded precision Gaussians

`BandedPrecision(b)` constrains the **precision** (inverse covariance) to have
half-bandwidth `b` in the given feature order. It does not constrain the covariance
bandwidth. Feature ordering is part of the model.

```@example banded
using StructuredGaussianMixtures, Distributions, Random, LinearAlgebra
X = randn(MersenneTwister(42), 8, 200)
g = fit(BandedPrecision(2), Exact(regularization=1e-5), X)
logpdf(g, X[:, 1])
```

The single-Gaussian path supports observation weights and continuation:

```@example banded
state = workspace(BandedPrecision(2), Exact(), g)
fit!(state, Exact(), X; weights=ones(size(X, 2)))
state.report.status
```

Independent mixture components use the same covariance fitter:

```@example banded
mixture = fit(MixtureSpec(BandedPrecision(2), 3), EM(maxiter=4), X;
              rng=MersenneTwister(3))
size(responsibilities(mixture, X))
```

## Fitting and numerical behavior

The band graph is decomposable. `Exact` fits each centered feature by weighted
regression on its preceding `b` features, then assembles the precision from the
conditional regression coefficients and residual variances. This is the exact
constrained Gaussian MLE when `regularization=0` and the required local scatter
matrices are positive definite. A positive regularization replaces the empirical
scatter `S` with `S + regularization*I` before solving; it does **not** add a ridge
to the final covariance. Equivalently it adds a trace penalty on precision.
The report records the unpenalized data log likelihood, not the penalized objective.
Consequently regularized mixture iterations do not promise monotonic unpenalized
likelihood. Rank-deficient unregularized local problems raise a factorization
error (or produce a failed outer EM report).

The fit stores only scatter bands, not a dense scatter matrix. Work is
O(n*p*b + p*b^3) for nonzero bandwidth, with the diagonal case O(n*p).
The representation stores precision and Cholesky bands in O(p*(b+1)) space.
Factorization takes O(p*(b+1)^2), and each score, sample, or precision solve takes
O(p*(b+1)). No optional package is required.

## Conditioning and marginalization

```@example banded
conditional = predict(g, X[1:3, 1]; input_indices=1:3, output_indices=4:8)
marginal_model = marginal(g, [1, 4, 8])
(mean(conditional), size(cov(marginal_model)))
```

When every variable outside the output is observed, conditioning uses a principal
precision submatrix and band solves, returning `BandedPrecisionMvNormal`.
Keeping output indices in their original order preserves a bandwidth no larger
than the original; arbitrary reordering can increase it.

When some variables are neither observed nor requested, they must be marginalized.
The implementation solves for the needed covariance columns and returns a dense
`MvNormal`; it does not incorrectly treat those variables as observed. Marginals
also use selected solves and return `MvNormal`, since their precision need not
remain banded. For q selected coordinates this uses O(p*q) temporary storage.
Calling `cov`, `var`, or `invcov` explicitly materializes dense matrices.

The constructor copies its inputs; accessors return independent arrays. Stored
fields are implementation details and must not be mutated because the precision
factor is cached.

```@docs
BandedPrecision
BandedPrecisionMvNormal
```
