# Explicit Latent Gaussians

`LatentMvNormal` retains a loading basis separately from its latent covariance:

```math
x = \mu + LBz + \epsilon,\quad z\sim N(0,I),\quad
\epsilon\sim N(0,\operatorname{Diag}(d)).
```

Thus the observation covariance is `L * B * B' * L' + Diagonal(d)`.
`B` is any square covariance factor; it need not be symmetric or triangular.
Positive finite residual variances are required. Singular `B`, zero latent
 dimension, and latent dimensions larger than the observation dimension are valid.

```@docs
LatentMvNormal
loading
latent_factor
```

## Usage and ownership

```@example latent
using StructuredGaussianMixtures, Distributions, Random, LinearAlgebra
rng = MersenneTwister(4)
μ = zeros(8)
L = randn(rng, 8, 2)
d = fill(0.2, 8)
B = [1.0 0.0; 0.3 0.7]
g = LatentMvNormal(μ, L, d, B)
X = rand(rng, g, 20)
logpdf(g, X) # One small factorization is shared across this batch.
```

The constructor copies inputs into `Float64` arrays, and `mean`, `loading`,
`latent_factor`, and `diagonal` return copies. Derived marginals and conditionals
also own their parameters. Fields are implementation details and should not be
mutated. Equal loadings across components describe equal values, not an enforced
fitting constraint. No persistent cache can become stale after array mutation.

`length` and `size` describe the observed dimension; `rank(g)` reports the stored
latent dimension (not the numerical rank of `L * B`). `low_rank_factor(g)` returns
`L * B`. `cov(g)` explicitly materializes the dense observation covariance.

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
latent dimension. These validation rules apply to the new latent type; existing
LRD behavior is unchanged.

For observed indices `O`, let `F_O = L_O * B` and
`M = I + F_O' * D_O^-1 * F_O`. If `M = R * R'` is a Cholesky factorization,
the conditional latent factor is `B / R'`. The target loading rows and residual
variances remain unchanged. The mean uses the same factorization and triangular
solves. Neither an explicit inverse nor a dense observation covariance is needed.
Mixture prediction updates component weights using the observed marginal densities.

## PCAEM migration

`fit(PCAEM(...), X)` now returns a mixture of `LatentMvNormal` components;
`FactorEM` still returns `LRDMvNormal` components. PCAEM estimates a shared PCA
basis `P`, shared reconstruction-residual variances `d`, and reduced-space means
`m_k` and covariances `A_k`. Its components retain
`μ_k = μ_global + P * m_k`, `L_k = P`, and `B_k * B_k' = A_k` explicitly.
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
Absorbing `B` discards the separate loading/latent-covariance interpretation.
Existing `predict`, `marginal`, `logpdf` and sampling calls continue to work.

## Computational costs

With observed dimension `p`, latent dimension `r`, and `n` observations, preparing
`L * B` and its small Cholesky costs `O(p*r^2 + r^3)`. Batched scoring then costs
`O(n*(p*r + r^2))` and uses `O(n*(p+r))` temporary storage. Preparation is local
to each scoring call. Sampling applies `B` and `L` directly and costs
`O(n*(r^2+p*r))`. Marginal and conditional outputs copy their parameters.
See `benchmark/latent.jl` for reproducible runtime and allocation measurements.
