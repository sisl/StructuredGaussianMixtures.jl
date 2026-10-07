# StructuredGaussianMixtures.jl
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://sisl.github.io/StructuredGaussianMixtures.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://sisl.github.io/StructuredGaussianMixtures.jl/dev)
[![codecov](https://codecov.io/gh/sisl/StructuredGaussianMixtures.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/sisl/StructuredGaussianMixtures.jl)



### Overview

Fit weighted single Gaussians and Gaussian mixtures with separate structure and
method specifications. Observations are columns of a features × samples matrix.
Fitted models support `logpdf`, sampling, marginalization and conditional `predict`.

```julia
using StructuredGaussianMixtures, Random
X = randn(MersenneTwister(1), 10, 500)
spec = MixtureSpec(LowRankDiagonal(2), 4)
method = EM(covariance_method=CovarianceEM(maxiter=20), n_init=3)
gmm = fit(spec, method, X; rng=MersenneTwister(2))
posterior = predict(gmm, [0.2, -0.1], [1, 2], [3, 4])

# A single Gaussian uses the covariance solver directly.
gaussian = fit(FullCovariance(), Exact(), X)

# Explicit initialized state for continuation; fit! never reinitializes.
state = workspace(spec, method, gmm)
fit!(state, method, X)
state.report
```

- `FullCovariance()` and `DiagonalCovariance()` use `Exact()` covariance updates.
- `LowRankDiagonal(r)` uses `CovarianceEM()` without forming a dense covariance.
- `LatentCovariance(r)` with `Tied(:F,:D)` uses `PCAEM(latent_method=EM(...))`.
  It retains the PCA loading and component latent covariances in `LatentMvNormal`.
- All fitting paths accept `weights=...` and fresh fits accept `rng=...`.
- Native `EM` provides convergence reports, restarts and weighted updates.
  `responsibilities(gmm, X)` returns posterior membership probabilities;
  `predict` continues to mean conditional prediction.

Install with `Pkg.add("StructuredGaussianMixtures")`. See the
[fitting guide](docs/src/fitting.md) for the API, supported combinations,
initialization and migration from the former EM/FactorEM/PCAEM interface.

Sparse precision Gaussians support sparse scoring and conditioning without an
optimization dependency. To fit graphical lasso models, use the default
native graphical-lasso solver, or optionally supply a conic optimizer through the
`Convex` extension; see
[the sparse precision guide](docs/src/sparseprecision.md).
