using GaussianMixtures
using Distributions
using LinearAlgebra
using MultivariateStats
using Random
using Statistics
using StructuredGaussianMixtures
using Plots

# Three-component GMM with specified means and weighted fitting
Random.seed!(42)
n_features = 2
n_components = 3
n_samples = 1000

# Create means at (-1, -1), (0, 0), and (1, 1)
true_means = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]

# Create small covariance matrices (identity scaled by 0.1)
true_covs = [0.1^2 * Matrix{Float64}(I, n_features, n_features) for _ in 1:n_components]

# Equal mixing weights
true_probs = ones(n_components) / n_components

# Create the GMM
true_gmm = MixtureModel(MvNormal.(true_means, true_covs), true_probs)

# Take 1000 samples
data = rand(true_gmm, n_samples)

# Assign weights: 1 if x1 <= 0, 0 otherwise
weights = [data[1, i] <= 0 ? 1.0 : 0.0 for i in 1:n_samples]

# Fit a 2-component rank-1 GMM using FactorEM with weights
@info "Fitting weighted FactorEM model"
fitmethod = FactorEM(2, 1; initialization_method=:kmeans, nInit=10, nIter=20)
@time gmm_weighted = StructuredGaussianMixtures.fit(fitmethod, data, weights)

# Print results
println("Number of samples with weight 1: ", sum(weights))
println("Number of samples with weight 0: ", sum(weights .== 0))
println("Weighted FactorEM Avg. Training LL: ", mean(logpdf(gmm_weighted, data)))
println(
    "Weighted FactorEM Avg. Training LL (weighted): ",
    sum(weights .* logpdf(gmm_weighted, data)) / sum(weights),
)

# Plot samples with different colors based on weights
p1 = scatter(
    data[1, weights .== 1],
    data[2, weights .== 1];
    alpha=0.8,
    title="Weighted Data (x1 ≤ 0)",
    xlabel="x1",
    ylabel="x2",
    label="Weight = 1",
    markersize=4,
    color=:red,
    margin=5Plots.mm,
)
scatter!(
    p1,
    data[1, weights .== 0],
    data[2, weights .== 0];
    alpha=0.3,
    label="Weight = 0",
    markersize=2,
    color=:gray,
)

x_lims = xlims(p1)
y_lims = ylims(p1)

# Generate samples from the fitted model
gmm_samples = rand(gmm_weighted, n_samples)
p2 = scatter(
    gmm_samples[1, :],
    gmm_samples[2, :];
    alpha=0.6,
    title="Fitted Model Samples",
    xlabel="x1",
    ylabel="x2",
    label="Model samples",
    markersize=3,
    margin=5Plots.mm,
    xlims=x_lims,
    ylims=y_lims,
)

# Combine plots
weighted_plot = plot(p1, p2; layout=(1, 2), size=(800, 400))
display(weighted_plot)
