# Fitting Methods

This page describes the three main fitting methods implemented in StructuredGaussianMixtures.jl: EM, PCAEM, and FactorEM.

## Overview

All fitting methods implement the `GMMFitMethod` interface and can be used with the `fit` function:

```julia
gmm = fit(fitmethod, data)
```

## GMMFitMethod Interface

```@docs
GMMFitMethod
```

## EM: Standard Expectation Maximization

The EM method fits Gaussian Mixture Models with full covariance matrices using standard Expectation Maximization.

### Constructor

```@docs
EM
```

### Usage

```julia
using StructuredGaussianMixtures

# Basic usage
fitmethod = EM(3)
gmm = fit(fitmethod, data)

# With custom parameters
fitmethod = EM(5; method=:rand, kind=:full, nInit=100, nIter=20)
gmm = fit(fitmethod, data)
```

### Fit Method

```@docs
StructuredGaussianMixtures.fit(::EM, ::Matrix)
```

## PCAEM: Mixture of Probabilistic Principal Component Analysis

PCAEM fits a GMM in PCA-reduced space and transforms back to the original space, effectively learning low-rank covariance structures.

### Constructor

```@docs
PCAEM
```

### Usage

```julia
# Basic usage
fitmethod = PCAEM(3, 2)
gmm = fit(fitmethod, data)

# With custom parameters
fitmethod = PCAEM(5, 3; gmm_method=:rand, gmm_nInit=100)
gmm = fit(fitmethod, data)
```

### Fit Method

```@docs
StructuredGaussianMixtures.fit(::PCAEM, ::Matrix)
```

## FactorEM: Mixture of Factor Analyzers

FactorEM directly fits GMMs with covariance matrices constrained to the form Σ = FF' + D, where F is a low-rank factor matrix and D is diagonal.

### Constructor

```@docs
FactorEM
```

### Usage

```julia
# Basic usage
fitmethod = FactorEM(3, 2)
gmm = fit(fitmethod, data)

# With custom parameters
fitmethod = FactorEM(5, 3; initialization_method=:rand, nInit=10, nIter=20)
gmm = fit(fitmethod, data)

# With weighted data
weights = ones(size(data, 2))  # Equal weights
gmm = fit(fitmethod, data, weights)
```

### Fit Methods

```@docs
StructuredGaussianMixtures.fit(::FactorEM, ::Matrix)
StructuredGaussianMixtures.fit(::FactorEM, ::Matrix, ::Vector)
```

## Method Comparison

### When to Use Each Method

| Method | Best For | Covariance Structure | Computational Cost |
|--------|----------|---------------------|-------------------|
| **EM** | Low-dimensional data, full covariance needed | Full | O(m³) |
| **PCAEM** | High-dimensional data, dimensionality reduction | Low-rank + diagonal | O(r³) where r < m |
| **FactorEM** | High-dimensional data, direct low-rank fitting | Low-rank + diagonal | O(r³) where r < m |

## Simple Examples

### Basic Fitting

```julia
using StructuredGaussianMixtures

# Generate some data
data = randn(2, 1000)

# Fit with different methods
gmm_em = fit(EM(3), data)
gmm_pca = fit(PCAEM(3, 1), data)
gmm_factor = fit(FactorEM(3, 1), data)

# Evaluate
println("EM log-likelihood: ", mean(logpdf(gmm_em, data)))
println("PCAEM log-likelihood: ", mean(logpdf(gmm_pca, data)))
println("FactorEM log-likelihood: ", mean(logpdf(gmm_factor, data)))
```

### Weighted Fitting

```julia
# Create weights based on data values
weights = [data[1, i] > 0 ? 1.0 : 0.5 for i in 1:size(data, 2)]

# Fit with weights (only FactorEM supports this)
gmm_weighted = fit(FactorEM(3, 1), data, weights)
```

### High-Dimensional Data

```julia
# For high-dimensional data
high_dim_data = randn(100, 50)

# Use low-rank methods
gmm_pca = fit(PCAEM(3, 10), high_dim_data)
gmm_factor = fit(FactorEM(3, 10), high_dim_data)
```

## Related Documentation

- **[LRDMvNormal](@ref)**: Learn about the low-rank plus diagonal distribution used by PCAEM and FactorEM methods 