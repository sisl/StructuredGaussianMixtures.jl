# LRDMvNormal

This page documents the `LRDMvNormal` class, which represents low-rank plus diagonal multivariate normal distributions.

## Overview

The `LRDMvNormal` distribution represents a multivariate normal distribution with covariance matrix of the form:

$$\Sigma = FF' + D$$

where:

- $F$ is a low-rank factor matrix of size $m \times r$ where $r \ll m$
- $D$ is a diagonal matrix
- The full covariance matrix is never explicitly formed for efficiency

This structure is particularly useful for high-dimensional data where $m \gg n$ (number of features much larger than number of samples).

## Constructor

```@docs
LRDMvNormal
```

### Usage

```julia
using StructuredGaussianMixtures

# Create a 10-dimensional distribution with rank-3 structure
μ = randn(10)           # Mean vector
F = randn(10, 3)        # Low-rank factor (10×3)
D = ones(10)            # Diagonal vector
dist = LRDMvNormal(μ, F, D)
```

## Distribution Interface

### Basic Properties

```@docs
Distributions.length(::LRDMvNormal)
Distributions.size(::LRDMvNormal)
Distributions.mean(::LRDMvNormal)
Distributions.cov(::LRDMvNormal)
```

### Low-Rank Structure Access

```@docs
rank(::LRDMvNormal)
low_rank_factor(::LRDMvNormal)
diagonal(::LRDMvNormal)
```

## Probability and Sampling

### Log Probability Density

```@docs
Distributions.logpdf(::LRDMvNormal, ::AbstractVector)
```

### Random Sampling

```@docs
Distributions._rand!(::AbstractRNG, ::LRDMvNormal, ::VecOrMat)
Distributions._rand!(::AbstractRNG, ::LRDMvNormal, ::AbstractVector)
```

## Simple Examples

### Creating and Using LRDMvNormal

```julia
using StructuredGaussianMixtures

# Create a low-rank distribution
n_features = 100
rank = 5
μ = randn(n_features)
F = randn(n_features, rank)
D = ones(n_features) * 0.1

dist = LRDMvNormal(μ, F, D)

# Basic properties
println("Dimension: ", length(dist))
println("Rank: ", rank(dist))
println("Mean: ", mean(dist)[1:5])  # First 5 elements

# Generate samples
samples = rand(dist, 1000)
println("Sample shape: ", size(samples))

# Compute log probability
log_prob = logpdf(dist, samples[:, 1])
println("Log probability: ", log_prob)
```

### Efficient Operations

```julia
# The covariance matrix is never explicitly formed
# This is efficient even for high dimensions
n_features = 1000
rank = 10

μ = randn(n_features)
F = randn(n_features, rank)
D = ones(n_features) * 0.1

dist = LRDMvNormal(μ, F, D)

# This is efficient - no O(n³) operations
log_prob = logpdf(dist, randn(n_features))

# Sampling is also efficient
samples = rand(dist, 100)
```

### Accessing Components

```julia
# Get the low-rank structure
F_matrix = low_rank_factor(dist)
D_vector = diagonal(dist)
rank_val = rank(dist)

println("F shape: ", size(F_matrix))
println("D length: ", length(D_vector))
println("Rank: ", rank_val)

# Full covariance (only for small dimensions!)
full_cov = cov(dist)
println("Full covariance shape: ", size(full_cov))
```

## Mathematical Background

### Efficient Log-Likelihood Computation

The log probability density is computed efficiently using the matrix inversion lemma:

$$(FF' + D)^{-1} = D^{-1} - D^{-1}F(I + F'D^{-1}F)^{-1}F'D^{-1}$$

This avoids forming the full covariance matrix and reduces computational complexity from O(m³) to O(r³) where r ≪ m.

### Determinant Computation

The determinant is computed efficiently as:

$$\det(FF' + D) = \det(D) \cdot \det(I + F'D^{-1}F)$$

### Sampling

Samples are generated using the decomposition:

$$X = \mu + FZ_1 + \sqrt{D}Z_2$$

where $Z_1$ and $Z_2$ are independent standard normal random vectors.

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Log-likelihood | O(r³) | Uses matrix inversion lemma |
| Sampling | O(mr) | Efficient decomposition |
| Memory | O(mr) | Stores F and D, not full covariance |

### Memory Usage

For a distribution with $m$ features and rank $r$:
- **Storage**: $O(mr + m) = O(mr)$ for F and D
- **Traditional**: $O(m²)$ for full covariance matrix
- **Savings**: $O(m²/mr) = O(m/r)$ for typical $r \ll m$

### When to Use

- **High-dimensional data** where $m \gg n$
- **Low-rank structure** in the data
- **Memory constraints** preventing full covariance storage
- **Efficient sampling** requirements

## Integration with GMMs

The `LRDMvNormal` distribution is used internally by `PCAEM` and `FactorEM` methods:

```julia
# PCAEM creates LRDMvNormal components
gmm = fit(PCAEM(3, 5), data)
for comp in components(gmm)
    println("Component rank: ", rank(comp))
end

# FactorEM also creates LRDMvNormal components
gmm = fit(FactorEM(3, 5), data)
for comp in components(gmm)
    println("Component rank: ", rank(comp))
end
``` 