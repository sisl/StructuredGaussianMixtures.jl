# StructuredGaussianMixtures.jl

A Julia package for fitting and conditional prediction of Gaussian Mixture Models (GMMs) with structured covariance matrices.

## Key Features

- **Efficient high-dimensional fitting**: Different fitting methods, including those that efficiently fit low-rank covariance structures for `m ≫ n` settings
- **Conditional prediction**: Compute posterior distributions over unobserved variables
- **Weighted fitting**: Support for weighted data points in model fitting

## Installation

```julia
using Pkg
Pkg.add("StructuredGaussianMixtures")
```

## Methods

Structure and method are separate: full/diagonal covariances use exact updates,
low-rank covariances use inner covariance EM, and PCAEM fits projected mixtures.
All support weighted data. Single-Gaussian fitting reuses the covariance solvers.

## Quick Start

```julia
using StructuredGaussianMixtures

# Fit a GMM using EM
data = randn(100, 1000)  # 100D data with 1000 samples
w = rand(1000) # weights on data 
gmm = fit(MixtureSpec(LowRankDiagonal(5), 3), EM(covariance_method=CovarianceEM()), data; weights=w)  # 3-component rank-5 low-rank-plus-diagonal GMM

# Make predictions
query_point = [0.5]
posterior = predict(gmm, query_point)  # Posterior over dimensions 2:100 when x1 = 0.5
```

## Documentation and API Reference sections

- **[Fitting](@ref)**: Learn about the different fitting algorithms and when to use each
    - **[Structured Gaussians](@ref)**: Learn about the structured Gaussians underpinning this project  
- **[Prediction](@ref)**: Understand conditional prediction and posterior computation
- **[Examples](@ref)**: Complete working examples from the test files