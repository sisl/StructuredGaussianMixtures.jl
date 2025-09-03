# Prediction

This page describes how to perform conditional prediction with fitted Gaussian Mixture Models using StructuredGaussianMixtures.jl.

## Overview

The `predict` function computes posterior distributions over unobserved dimensions given observed values. This is useful for:

- **Missing data imputation**: Fill in missing values based on observed data
- **Conditional sampling**: Generate samples from the posterior distribution
- **Uncertainty quantification**: Assess uncertainty in predictions

## Function Signature

```julia
predict(gmm, x; input_indices=1:length(x), output_indices=length(x)+1:length(gmm))
```

### Parameters

- `gmm`: A fitted MixtureModel
- `x`: Observed values
- `input_indices`: Indices of observed dimensions (default: `1:length(x)`)
- `output_indices`: Indices of dimensions to predict (default: `length(x)+1:length(gmm)`)

## Basic Usage

```julia
using StructuredGaussianMixtures

# Fit a GMM
data = randn(2, 1000)
gmm = fit(EM(3), data)

# Make predictions
query_point = [0.5]  # Observed value for first dimension
posterior = predict(gmm, query_point)  # Posterior over second dimension
```

## Predict Functions

### MvNormal Prediction

```@docs
StructuredGaussianMixtures.predict(::MvNormal, ::AbstractVector, ::Union{Vector{Int},AbstractRange}, ::Union{Vector{Int},AbstractRange})
```

### LRDMvNormal Prediction

```@docs
StructuredGaussianMixtures.predict(::LRDMvNormal, ::AbstractVector, ::Union{Vector{Int},AbstractRange}, ::Union{Vector{Int},AbstractRange})
```

### MixtureModel Prediction

```@docs
StructuredGaussianMixtures.predict(::MultivariateMixture, ::AbstractVector, ::Union{Vector{Int},AbstractRange}, ::Union{Vector{Int},AbstractRange})
```

### Convenience Function

```@docs
StructuredGaussianMixtures.predict(::Union{MvNormal,LRDMvNormal,MultivariateMixture}, ::AbstractVector)
```

## Marginal Functions

### Marginal Distribution

```@docs
StructuredGaussianMixtures.marginal(::MvNormal, ::Union{AbstractRange,Vector{Int}})
StructuredGaussianMixtures.marginal(::LRDMvNormal, ::Union{AbstractRange,Vector{Int}})
```

## Simple Examples

### 2D GMM Prediction

```julia
# Fit a GMM
data = randn(2, 1000)
gmm = fit(EM(3), data)

# Make prediction
x_query = [0.5]
posterior = predict(gmm, x_query)

# Generate samples from posterior
samples = rand(posterior, 100)
println("Posterior mean: ", mean(samples))
```

### High-Dimensional Prediction

```julia
# Fit a high-dimensional GMM
data = randn(10, 1000)
gmm = fit(PCAEM(3, 3), data)

# Predict multiple dimensions
observed_dims = [1, 3, 5]
query_values = [0.5, -0.2, 1.1]
unobserved_dims = [2, 4, 6, 7, 8, 9, 10]

posterior = predict(gmm, query_values; 
    input_indices=observed_dims, 
    output_indices=unobserved_dims)

# Generate samples
samples = rand(posterior, 100)
println("Predicted dimensions shape: ", size(samples))
```

### Partial Prediction

```julia
# Fit a 5D GMM
data = randn(5, 1000)
gmm = fit(EM(3), data)

# Observe dimensions 1 and 3, predict dimensions 2, 4, and 5
observed_values = [0.5, -0.2]
observed_dims = [1, 3]
output_dims = [2, 4, 5]

posterior = predict(gmm, observed_values; 
    input_indices=observed_dims, 
    output_indices=output_dims)

samples = rand(posterior, 100)
println("Predicted dimensions shape: ", size(samples))
```

## Mathematical Background

### Conditional Gaussian Distribution

For a Gaussian Mixture Model with components $k = 1, \ldots, K$, the posterior distribution given observed values $x_{obs}$ is:

$$p(x_{unobs} \mid x_{obs}) = \sum_{k=1}^K w^k \mathcal{N}(x_{unobs} \mid \mu^k_{unobs}, \Sigma^k_{unobs})$$

where $w^k$ are the posterior component weights, $\mu^k_{unobs}$ is the conditional mean for component $k$, and $\Sigma^k_{unobs}$ is the conditional covariance for component $k$

### Component Weight Updates

The posterior component weights are computed as:

$$w^k = \frac{\pi^k \mathcal{N}(x_{obs} \mid \mu^k_{obs}, \Sigma^k_{obs})}{\sum_{j=1}^K \pi^j \mathcal{N}(x_{obs} \mid \mu^j_{obs}, \Sigma^j_{obs})}$$

### Conditional Parameters

For each component $k$, the conditional parameters are:

$$\mu^k_{unobs} = \mu^k_{unobs} + \Sigma^k_{unobs,obs} (\Sigma^k_{obs})^{-1} (x_{obs} - \mu^k_{obs})$$

$$\Sigma^k_{unobs} = \Sigma^k_{unobs,unobs} - \Sigma^k_{unobs,obs} (\Sigma^k_{obs})^{-1} \Sigma^k_{obs,unobs}$$

## Advanced Usage

### Multiple Query Points

```julia
# Predict for multiple query points
query_points = [[0.5], [-0.2], [1.1]]
posteriors = [predict(gmm, q) for q in query_points]

# Generate samples for each
samples = [rand(p, 50) for p in posteriors]
```

### Uncertainty Quantification

```julia
# Compute posterior statistics
posterior = predict(gmm, query_point)
samples = rand(posterior, 1000)

# Mean and variance
posterior_mean = mean(samples, dims=2)
posterior_var = var(samples, dims=2)

# Confidence intervals
posterior_quantiles = quantile(samples, [0.025, 0.975], dims=2)
```