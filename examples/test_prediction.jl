using GaussianMixtures
using Distributions
using LinearAlgebra
using MultivariateStats
using Random
using Statistics
using StructuredGaussianMixtures
using Plots

# Full-rank 2D GMM with non-overlapping components
n_features = 2
n_components = 5
true_probs = rand(n_components) 
true_probs = true_probs / sum(true_probs)
true_means = [rand([-3.0, 3.0], n_features) for _ in 1:n_components]
true_Ls = [randn(n_features, n_features) for _ in 1:n_components]
true_covs = [true_Ls[i] * true_Ls[i]' for i in 1:n_components]
true_gmm = MixtureModel(MvNormal.(true_means, true_covs), true_probs)
data = rand(true_gmm, 1000)
gmm_full = StructuredGaussianMixtures.fit(EM(n_components), data)
gmm_full_samples = rand(gmm_full, 1000)
gmm_pca = StructuredGaussianMixtures.fit(PCAEM(n_components, 1), data)
gmm_pca_samples = rand(gmm_pca, 1000)
x_query = data[1:1,1]
posterior_full = StructuredGaussianMixtures.predict(gmm_full, x_query)
posterior_pca = StructuredGaussianMixtures.predict(gmm_pca, x_query)
n_posterior_samples = 100
posterior_full_samples = rand(posterior_full, n_posterior_samples)
posterior_pca_samples = rand(posterior_pca, n_posterior_samples)

# Plot the results
p1 = scatter(data[1,:], data[2,:], label="True Data", alpha=0.5)
p2 = scatter(gmm_full_samples[1,:], gmm_full_samples[2,:], label="Full GMM", alpha=0.5)
posterior_xs = [x_query[1] for _ in 1:n_posterior_samples]
scatter!(p2, posterior_xs, posterior_full_samples[1,:], label="Posterior Full", alpha=0.5, color=:red)
vline!(p2, [x_query[1]], label="Query Point", color=:black, linestyle=:dash)

p3 = scatter(gmm_pca_samples[1,:], gmm_pca_samples[2,:], label="PCA GMM", alpha=0.5)
scatter!(p3, posterior_xs, posterior_pca_samples[1,:], label="Posterior PCA", alpha=0.5, color=:red)
vline!(p3, [x_query[1]], label="Query Point", color=:black, linestyle=:dash)
plot(p1, p2, p3, layout=(1,3), size=(1500, 500))

