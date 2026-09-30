# Explicit Latent Gaussians

`LatentMvNormal` retains a loading basis separately from its latent covariance:

```math
x = \mu + F A_{\mathrm{factor}} z + \epsilon,\quad z\sim N(0,I),\quad
\epsilon\sim N(0,\operatorname{Diag}(d)).
```

Thus the observation covariance is `F * A_factor * A_factor' * F' + Diagonal(d)`.
`A_factor` is any square covariance factor; it need not be symmetric or triangular.
Positive finite residual variances are required. Singular `A_factor`, zero latent
dimension, and latent dimensions larger than the observation dimension are valid.

```@docs
LatentMvNormal
loading
latent_covariance_factor
latent_covariance
```

## Usage and ownership

```@example latent
using StructuredGaussianMixtures, Distributions, Random, LinearAlgebra
rng = MersenneTwister(4)
μ = zeros(8)
F = randn(rng, 8, 2)
d = fill(0.2, 8)
A_factor = [1.0 0.0; 0.3 0.7]
g = LatentMvNormal(μ, F, d, A_factor)
X = rand(rng, g, 20)
logpdf(g, X) # One small factorization is shared across this batch.
```

The constructor copies inputs into `Float64` arrays, and `mean`, `loading`,
`latent_covariance_factor`, and `diagonal` return copies. Derived marginals and conditionals
also own their parameters. Fields are implementation details and should not be
mutated. Equal loadings across components describe equal values, not an enforced
fitting constraint. No persistent cache can become stale after array mutation.

`length` and `size` describe the observed dimension; `rank(g)` reports the stored
latent dimension (not the numerical rank of `F * A_factor`). `low_rank_factor(g)` returns
`F * A_factor`. `cov(g)` explicitly materializes the dense observation covariance.

## Marginalization and conditioning

```@docs
marginal(::LatentMvNormal, ::Union{Vector{Int},AbstractRange})
predict(::LatentMvNormal, ::AbstractVector, ::Union{Vector{Int},AbstractRange}, ::Union{Vector{Int},AbstractRange})
```

```@example latent
m = marginal(g, [8, 2, 1])
c = predict(g, [0.5, -0.2], [1, 3], [2])
@assert c isa LatentMvNormal
@assert loading(c) == loading(g)[[2], :]
mean(c), cov(c)
```

Indices must be unique, in bounds and disjoint between inputs and outputs.
Outputs must be nonempty. Empty inputs return the target marginal. Coordinates
in neither set are marginalized, not implicitly conditioned on. The conditional
remains `LatentMvNormal`, even when its observed dimension is smaller than its
latent dimension. These validation rules apply to both structured types. LRD continues to return
a dense `MvNormal` when the output dimension is no larger than its stored latent dimension.

For observed indices `O`, let `effective_factor_O = F_O * A_factor` and
`M = I + effective_factor_O' * D_O^-1 * effective_factor_O`. If `M = R * R'` is a Cholesky factorization,
the conditional latent factor is `A_factor / R'`. The target loading rows and residual
variances remain unchanged. The mean uses the same factorization and triangular
solves. Neither an explicit inverse nor a dense observation covariance is needed.
Mixture prediction updates component weights using the observed marginal densities.

## PCAEM migration

`fit(PCAEM(...), X)` now returns a mixture of `LatentMvNormal` components;
`FactorEM` still returns `LRDMvNormal` components. PCAEM estimates a shared PCA
basis `P`, shared reconstruction-residual variances `d`, and reduced-space means
`m_k` and covariances `A_k`. Its components retain
`μ_k = μ_global + P * m_k`, `F_k = P`, and `A_factor_k * A_factor_k' = A_k` explicitly.
Each component owns copies of the common values to prevent accidental cross-component
mutation. This representation change adds no new shared-parameter optimization.

Update dispatch/type assertions that assumed PCAEM returned `LRDMvNormal`.
To absorb the latent factor explicitly when the latent dimension is smaller than
the observed dimension (as required by the existing LRD constructor):

```@example latent
lrd = LRDMvNormal(mean(g), low_rank_factor(g), diagonal(g))
@assert cov(lrd) ≈ cov(g)
```

For other dimensions, `MvNormal(mean(g), Symmetric(cov(g)))` is a dense fallback.
Absorbing `A_factor` discards the separate loading/latent-covariance interpretation.
Existing `predict`, `marginal`, `logpdf` and sampling calls continue to work.

## Computational costs

With observed dimension `p`, latent dimension `r`, and `n` observations, preparing
`F * A_factor` and its small Cholesky costs `O(p*r^2 + r^3)`. Batched scoring then costs
`O(n*(p*r + r^2))` and uses `O(n*(p+r))` temporary storage. Preparation is local
to each scoring call. Sampling applies `A_factor` and `F` directly and costs
`O(n*(r^2+p*r))`. Marginal and conditional outputs copy their parameters.
Local runtime and allocation measurements are summarized in the pull request;
benchmark scripts and generated results are not part of the package.

## Common interface and compatibility

Both structured types support `mean`, `cov`, `var`, `length`, `size`, vector and
matrix `logpdf`, batched `logpdf!`, sampling with `rand`/`rand!`, `marginal`, and
`predict`. `var(g)` computes marginal variances without constructing `cov(g)`.

| Accessor | `LRDMvNormal` | `LatentMvNormal` |
|---|---|---|
| `loading(g)` | Copy of `F` | Copy of `F` |
| `latent_covariance(g)` | Sized diagonal identity | `A_factor * A_factor'` |
| `latent_covariance_factor(g)` | Sized diagonal identity | Copy of `A_factor` |
| `low_rank_factor(g)` | Copy of `F` | `F * A_factor` |
| `diagonal(g)` | Copy of residual variances | Copy of residual variances |
| `rank(g)` | Stored latent dimension | Stored latent dimension |

LRD's positional constructor remains `LRDMvNormal(μ, F, D)`, and its `.F` field is retained; prefer `loading(g)` or `low_rank_factor(g)` for public
access. Both constructors copy inputs, and all array-valued accessors return
independent values. Code that previously mutated an LRD by modifying constructor
inputs or accessor results must instead construct a new distribution. Package
fitting routines explicitly update owned fields internally. LRD now also rejects
nonfinite means/loadings/variances and invalid, repeated or overlapping prediction
indices. Empty conditioning inputs are allowed; empty output/marginal sets are not.

Both types use `F` for the loading matrix. `A_factor` stores a factor of the
latent covariance and is exposed through `latent_covariance_factor`. This factor
is not necessarily triangular, so it is named `A_factor` rather than `A_chol`.

```@example latent
@assert latent_covariance(lrd) == Matrix{Float64}(I, 2, 2)
@assert latent_covariance(g) ≈ A_factor * A_factor'
@assert var(g) ≈ diag(cov(g))
scores = zeros(size(X, 2))
logpdf!(scores, g, X)
@assert scores ≈ logpdf(g, X)
```
