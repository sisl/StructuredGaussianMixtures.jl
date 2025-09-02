# StructuredGaussianMixtures.jl
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://sisl.github.io/StructuredGaussianMixtures.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://sisl.github.io/StructuredGaussianMixtures.jl/dev)
[![codecov](https://codecov.io/gh/sisl/StructuredGaussianMixtures.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/sisl/StructuredGaussianMixtures.jl)



### Overview

This package implements fitting and conditional prediction for Gaussian Mixture Models (GMM). Given a matrix of data $X$ of shape $(n_{features}, n_{samples})$, the method `fit(::GMMFitMethod, X)` returns a fitted GMM as a `Distributions.MixtureModel`. Given a fitted GMM, the method `predict(gmm, x; input_indices=1:length(x), output_indices=length(x)+1:length(gmm))` returns a posterior GMM over the output dimensions conditioned on the observed vector `x` at the input indices.

This package supports efficient fitting of low-rank-plus-diagonal GMMs, i.e. those with covariance structure $Σ = FF' + D$ with low-rank factors $F$ and diagonals $D$. To do so, this package implements a `LRDMvNormal <: Distributions.AbstractMvNormal` class with efficient marginal, conditional, and `logpdf` calculations. 

This package currently implements three `GMMFitMethod`s.

- **FactorEM** fits a mixture of factor analyzers using Expectation Maximization (EM), fitting GMMs with covariance matrices constrained to the form $Σ = FF' + D$, where F is a low-rank factor matrix and D is diagonal. During Maximization, $F$ and $D$ are updated with an inner EM routine -- the full covariance is never formed or inverted. This method is effective for high-dimensional data, or data where the number of features exceeds the number of samples. This method also currently supports fitting weighted data.

- **EM** fits a GMM with full-rank covariance using a standard Expectation Maximization procedure from  `GaussianMixtures.jl`. 

- **PCAEM** fit a low-rank-plus-diagonal GMM by compressing the data using PCA, fitting a GMM in the reduced space, transforming back, and adding residuals to the diagonals. 

### Installation

This package can be installed using the Julia package manager

```
using Pkg
Pkg.add('StructuredGaussianMixtures')
```

### Usage

Given a data matrix `X`, fit a 4-component, rank-2 mixture of factor analyzers with:
```
using StructuredGaussianMixtures
fit_method = FactorEM(4, 2)
gmm = fit(fit_method, X)
```

Alternatively, fit a GMM on weighted data with weights `w`:
```
gmm = fit(fit_method, X, w)
```

Condition the distributions with:
```
x_obs = rand(5)
posterior_a = predict(gmm, x_obs) # p(x_{6:n_features} | x_{1:5} = x_obs)
posterior_b = predict(gmm, x_obs, 6:10, 1:3) # p(x_{1:3} | x_{6:10} = x_obs)
```

See the `examples` folder for example usage (note to  `] dev ..` in the `examples` environment to add `StructuredGaussianMixtures` in development mode).