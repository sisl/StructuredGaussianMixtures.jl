# Examples

This page provides simple working examples demonstrating typical workflows for fitting, prediction, and weighted fitting.

## Example 1: Basic Fitting

This example shows how to fit GMMs using the three different methods.

```julia
using StructuredGaussianMixtures

# Generate some data
data = randn(2, 1000)

# Fit with different methods
gmm_em = fit(EM(3), data)
gmm_pca = fit(PCAEM(3, 1), data)
gmm_factor = fit(FactorEM(3, 1), data)

# Evaluate performance
println("EM log-likelihood: ", mean(logpdf(gmm_em, data)))
println("PCAEM log-likelihood: ", mean(logpdf(gmm_pca, data)))
println("FactorEM log-likelihood: ", mean(logpdf(gmm_factor, data)))
```

## Example 2: High-Dimensional Data

This example demonstrates the effectiveness of low-rank methods for high-dimensional data.

```julia
# High-dimensional data
n_features = 100
n_samples = 50
data = randn(n_features, n_samples)

# Compare methods
println("Fitting EM model...")
@time gmm_em = fit(EM(3), data)

println("Fitting PCAEM model...")
@time gmm_pca = fit(PCAEM(3, 10), data)

println("Fitting FactorEM model...")
@time gmm_factor = fit(FactorEM(3, 10), data)

# Compare performance
println("EM log-likelihood: ", mean(logpdf(gmm_em, data)))
println("PCAEM log-likelihood: ", mean(logpdf(gmm_pca, data)))
println("FactorEM log-likelihood: ", mean(logpdf(gmm_factor, data)))
```

## Example 3: Conditional Prediction

This example demonstrates how to perform conditional prediction.

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
println("Posterior variance: ", var(samples))
```

## Example 4: Weighted Fitting

This example shows how to fit a GMM with weighted data points.

```julia
# Generate data
data = randn(2, 1000)

# Create weights based on data values
weights = [data[1, i] > 0 ? 1.0 : 0.5 for i in 1:size(data, 2)]

# Fit with weights (only FactorEM supports this)
gmm_weighted = fit(FactorEM(3, 1), data, weights)

# Print results
println("Number of samples with weight 1: ", sum(weights .== 1.0))
println("Number of samples with weight 0.5: ", sum(weights .== 0.5))
println("Weighted log-likelihood: ", mean(logpdf(gmm_weighted, data)))
```

## Example 5: LRDMvNormal Usage

This example shows how to work directly with LRDMvNormal distributions.

```julia
using StructuredGaussianMixtures

# Create a low-rank distribution
n_features = 10
rank = 3
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

## Example 6: Partial Prediction

This example shows how to predict specific dimensions.

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

# Generate samples
samples = rand(posterior, 100)
println("Predicted dimensions shape: ", size(samples))
```

## Example 7: Multiple Query Points

This example shows how to make predictions for multiple query points.

```julia
# Fit a GMM
data = randn(2, 1000)
gmm = fit(EM(3), data)

# Predict for multiple query points
query_points = [[0.5], [-0.2], [1.1]]
posteriors = [predict(gmm, q) for q in query_points]

# Generate samples for each
samples = [rand(p, 50) for p in posteriors]

# Print results
for (i, (q, s)) in enumerate(zip(query_points, samples))
    println("Query $i: mean = ", mean(s), ", variance = ", var(s))
end
```

## Running the Examples

To run these examples:

1. **Install dependencies**:
   ```julia
   using Pkg
   Pkg.add(["StructuredGaussianMixtures", "Distributions", "LinearAlgebra", "Random", "Statistics"])
   ```

2. **Load the package**:
   ```julia
   using StructuredGaussianMixtures
   ```

3. **Run examples**: Copy and paste any example into a Julia session.
