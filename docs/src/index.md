# StructuredGaussianMixtures.jl

A Julia package for fitting and conditional prediction of Gaussian Mixture Models (GMMs) with structured covariance matrices.

## Overview

This package implements three main fitting methods for Gaussian Mixture Models:

- **EM**: Standard Expectation Maximization for full covariance matrices
- **PCAEM**: Mixture of Probabilistic Principal Component Analysis models
- **FactorEM**: Mixture of Factor Analyzers with low-rank plus diagonal covariance structure

The package also supports conditional prediction, allowing you to compute posterior distributions over unobserved dimensions given observed values.

## Quick Start

```julia
using StructuredGaussianMixtures

# Fit a GMM using EM
data = randn(2, 1000)  # 2D data with 1000 samples
gmm = fit(EM(3), data)  # 3-component GMM

# Make predictions
query_point = [0.5]
posterior = predict(gmm, query_point)  # Posterior over second dimension
```

## Documentation Sections

- **[Fitting Methods](@ref)**: Learn about the different fitting algorithms and when to use each
- **[Prediction](@ref)**: Understand conditional prediction and posterior computation
- **[Examples](@ref)**: Complete working examples from the test files

## Key Features

- **Efficient high-dimensional fitting**: Low-rank covariance structures for `m ≫ n` settings
- **Conditional prediction**: Compute posterior distributions over unobserved variables
- **Weighted fitting**: Support for weighted data points in model fitting
- **Multiple initialization methods**: K-means, random, and custom initialization
- **Comprehensive evaluation**: Log-likelihood, JSD, and visualization tools

## Installation

```julia
using Pkg
Pkg.add("StructuredGaussianMixtures")
```

## API Reference

The main functions and types are documented in their respective sections:

- **[Fitting Methods](@ref)**: `EM`, `PCAEM`, `FactorEM`, and `fit` functions
- **[LRDMvNormal](@ref)**: Low-rank plus diagonal distribution
- **[Prediction](@ref)**: `predict` functions for conditional inference
