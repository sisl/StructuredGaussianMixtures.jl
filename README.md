# StructuredGaussianMixtures.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://sisl.github.io/StructuredGaussianMixtures.jl/dev)
[![codecov](https://codecov.io/gh/sisl/StructuredGaussianMixtures.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/sisl/StructuredGaussianMixtures.jl)



### Overview

This package implements fitting and conditional prediction for Gaussian Mixture Models (GMM). Given a matrix of data $X$ of shape $(n_{features}, n_{samples})$, the method `fit(::GMMFitMethod, X)` returns a fitted GMM as a `Distributions.MixtureModel`. Given a fitted gmm, the method `predict(gmm, x; input_indices=1:length(x), output_indices=length(x)+1:length(gmm))` returns a posterior GMM over the output dimensions conditioned on the observed vector `x` at the input indices.

This package implements three `GMMFitMethod`s.

- **EM**: Standard Expectation Maximization for fitting Gaussian Mixture Models with full covariance matrices. Uses the GaussianMixtures.jl implementation with configurable initialization methods and covariance structures.

- **PCAEM**: Fit a GMM in PCA-reduced space and transforms back to the original space, effectively learning low-rank covariance structures through dimensionality reduction. 

- **FactorEM**: Fit a mixture of factor analyzers [under construction]. Directly fit GMMs with covariance matrices constrained to the form $Σ = FF' + D$, where F is a low-rank factor matrix and D is diagonal, relying on an internal EM step for factor analysis. This method is particularly effective for high-dimensional data where the number of features exceeds the number of samples.

To perform fitting and prediction efficiently, `PCAEM` and `FactorEM` rely on an `LRDMvNormal <: Distributions.AbstractMvNormal` class that this packages introduces to represent structured covariance matrices $Σ = FF' + D$ using their factor loadings and diagonal vectors.

### Usage

See the `examples` folder for example usage (note to  `] dev ..` in the `examples` environment to add `StructuredGaussianMixtures` in development mode).