using GaussianMixtures
using Distributions
using LinearAlgebra
using MultivariateStats
using Random
using Statistics
using StructuredGaussianMixtures
using Plots

function compare_methods_pca(true_gmm, n_components, n_rank; run_pca=true, n_samples=1000)
    data = rand(true_gmm, n_samples)

    # Fit using EM
    fitmethod = EM(n_components)
    gmm_full = StructuredGaussianMixtures.fit(fitmethod, data)
    gmm_full_samples = rand(gmm_full, n_samples)

    # Fit using PCAEM
    fitmethod = PCAEM(n_components, n_rank)
    gmm_pca = StructuredGaussianMixtures.fit(fitmethod, data)
    gmm_pca_samples = rand(gmm_pca, n_samples)


    # plot samples from the three models
    if run_pca
        pca = MultivariateStats.fit(PCA, data; maxoutdim=2);
        data = transform(pca, data)
        gmm_full_samples = transform(pca, gmm_full_samples)   
        gmm_pca_samples = transform(pca, gmm_pca_samples)   
    end

    # Create scatter plots of the three models
    xlabel = run_pca ? "PC1" : "Feature 1"
    ylabel = run_pca ? "PC2" : "Feature 2"
    p1 = scatter(data[1, :], data[2, :], 
                alpha=0.6, title="True Model", xlabel=xlabel, ylabel=ylabel, 
                label="True samples", markersize=2, margin=5Plots.mm)
    p2 = scatter(gmm_full_samples[1, :], gmm_full_samples[2, :], 
                alpha=0.6, title="EM Model", xlabel=xlabel, ylabel=ylabel, 
                label="EM samples", markersize=2, margin=5Plots.mm)
    p3 = scatter(gmm_pca_samples[1, :], gmm_pca_samples[2, :], 
                alpha=0.6, title="PCAEM Model", xlabel=xlabel, ylabel=ylabel, 
                label="PCAEM samples", markersize=2, margin=5Plots.mm)

    # Combine all plots
    return plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
end

# Full-rank 2D GMM with potentially overlapping components
n_features = 2
n_components = 5
n_rank = 1
true_probs = rand(n_components) 
true_probs = true_probs / sum(true_probs)
true_means = [rand([-3.0, 3.0], n_features) for _ in 1:n_components]
true_Ls = [randn(n_features, n_features) for _ in 1:n_components]
true_covs = [true_Ls[i] * true_Ls[i]' for i in 1:n_components]
true_gmm = MixtureModel(MvNormal.(true_means, true_covs), true_probs)
compare_methods_pca(true_gmm, n_components, n_rank; run_pca=false)

# Low-Rank GMM with tied component ranks
n_rank = 2
n_features = 400
n_components = 3
sigma = 0.1
true_probs = rand(n_components) 
true_probs = true_probs / sum(true_probs)
F = randn(n_features, n_rank)
true_Ls = [F * randn(n_rank, n_rank) for _ in 1:n_components]
true_covs = [true_Ls[i] * true_Ls[i]' + sigma^2*I for i in 1:n_components]
true_means = [rand([-1.0, 1.0], n_features) for _ in 1:n_components]
true_gmm = MixtureModel(MvNormal.(true_means, true_covs), true_probs)
compare_methods_pca(true_gmm, n_components, n_rank; run_pca=true)

# Low-Rank GMM with independent component ranks
n_rank = 2
n_features = 400
n_components = 3
sigma = 0.1
true_probs = rand(n_components) 
true_probs = true_probs / sum(true_probs)
true_means = [rand([-1.0, 1.0], n_features) for _ in 1:n_components]
true_Fs = [randn(n_features, n_rank) for _ in 1:n_components]
true_covs = [true_Fs[i] * true_Fs[i]' + sigma^2 * I for i in 1:n_components]
true_gmm = MixtureModel(MvNormal.(true_means, true_covs), true_probs)
compare_methods_pca(true_gmm, n_components, n_rank; run_pca=true)
