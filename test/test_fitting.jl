using Test
using LinearAlgebra
using Random
using Distributions
using StructuredGaussianMixtures
using GaussianMixtures
using Statistics
using MultivariateStats: PCA, fit as pca_fit, projection, mean as pca_mean, predict as pca_predict, reconstruct

@testset "Fitting Methods" begin
    # Set random seed for reproducibility
    Random.seed!(42)

    # Create test data
    n_features = 20
    n_samples = 1000
    X = randn(n_features, n_samples)

    @testset "EM" begin
        # Test EM constructor
        em = EM(3)
        @test em.n_components == 3
        @test em.method == :kmeans
        @test em.kind == :full
        @test em.nInit == 50
        @test em.nIter == 10
        @test em.nFinal == 10

        # Test EM with custom parameters
        em_custom = EM(3; method=:kmeans, kind=:diag, nInit=10, nIter=5, nFinal=3)
        @test em_custom.n_components == 3
        @test em_custom.method == :kmeans
        @test em_custom.kind == :diag
        @test em_custom.nInit == 10
        @test em_custom.nIter == 5
        @test em_custom.nFinal == 3

        # Test EM fitting
        gmm = StructuredGaussianMixtures.fit(em_custom, X)
        @test length(gmm.components) == 3
        @test length(gmm.prior.p) == 3
        @test sum(gmm.prior.p) ≈ 1.0 atol = 1e-10

        # Test that components are MvNormal
        for comp in gmm.components
            @test comp isa MvNormal
        end

        # Test logpdf on fitted model
        logps = logpdf(gmm, X)
        @test size(logps) == (n_samples,)
        @test all(isfinite, logps)
    end

    @testset "PCAEM" begin
        # Test PCAEM constructor
        pcaem = PCAEM(3, 5)
        @test pcaem.n_components == 3
        @test pcaem.rank == 5
        @test pcaem.gmm_method == :kmeans
        @test pcaem.gmm_kind == :full
        @test pcaem.gmm_nInit == 50
        @test pcaem.gmm_nIter == 10
        @test pcaem.gmm_nFinal == 10

        # Test PCAEM with custom parameters
        pcaem_custom = PCAEM(
            4, 6; gmm_method=:kmeans, gmm_kind=:diag, gmm_nInit=5, gmm_nIter=3, gmm_nFinal=2
        )
        @test pcaem_custom.n_components == 4
        @test pcaem_custom.rank == 6
        @test pcaem_custom.gmm_method == :kmeans
        @test pcaem_custom.gmm_kind == :diag
        @test pcaem_custom.gmm_nInit == 5
        @test pcaem_custom.gmm_nIter == 3
        @test pcaem_custom.gmm_nFinal == 2

        # Test PCAEM fitting
        gmm = StructuredGaussianMixtures.fit(pcaem, X)
        @test length(gmm.components) == 3
        @test length(gmm.prior.p) == 3
        @test sum(gmm.prior.p) ≈ 1.0 atol = 1e-10

        # Test that components are LRDMvNormal
        for comp in gmm.components
            @test comp isa LRDMvNormal
            @test StructuredGaussianMixtures.rank(comp) == 5
        end

        # Test logpdf on fitted model
        logps = logpdf(gmm, X)
        @test size(logps) == (n_samples,)
        @test all(isfinite, logps)

        # Test with rank larger than n_features
        pcaem_large_rank = PCAEM(2, n_features + 5)
        @test_throws ArgumentError StructuredGaussianMixtures.fit(pcaem_large_rank, X)
    end

    @testset "Truncated PCA equivalence" begin
        # The truncated PCA helper must reproduce what PCAEM actually uses from a
        # full PCA: the top-rank reconstruction subspace and the reconstruction-error
        # diagonal. Loadings are only defined up to per-column sign / rotation within
        # tied eigenvalues, so we assert sign/rotation-invariant quantities only.

        # d <= n branch (eigendecompose the d x d covariance)
        for (d, n, rank) in [(20, 1000, 5), (20, 1000, 6)]
            data = randn(d, n)

            P_trunc, μ_trunc = StructuredGaussianMixtures._truncated_pca(data, rank)
            @test size(P_trunc) == (d, rank)

            pca = pca_fit(PCA, data; maxoutdim=rank)
            P_full = projection(pca)
            μ_full = pca_mean(pca)

            # Means match exactly
            @test μ_trunc ≈ μ_full atol = 1e-10

            # Reconstruction subspace (projector onto top-rank subspace) matches
            @test P_trunc * P_trunc' ≈ P_full * P_full' atol = 1e-8

            # Reconstruction-error variance diagonal D matches
            reduced_trunc = P_trunc' * (data .- μ_trunc)
            recon_trunc = P_trunc * reduced_trunc .+ μ_trunc
            D_trunc = vec(var(data .- recon_trunc; dims=2))

            reduced_full = pca_predict(pca, data)
            recon_full = reconstruct(pca, reduced_full)
            D_full = vec(var(data .- recon_full; dims=2))

            @test D_trunc ≈ D_full atol = 1e-8
        end

        # d <= n branch, large enough to exercise Arpack's iterative eigs path
        # (smaller dimension > 100 so it does not hit the dense fallback)
        let d = 150, n = 400, rank = 5
            data = randn(d, n)
            P_trunc, μ_trunc = StructuredGaussianMixtures._truncated_pca(data, rank)
            @test size(P_trunc) == (d, rank)

            pca = pca_fit(PCA, data; maxoutdim=rank)
            P_full = projection(pca)
            μ_full = pca_mean(pca)

            @test μ_trunc ≈ μ_full atol = 1e-10
            @test P_trunc * P_trunc' ≈ P_full * P_full' atol = 1e-8

            reduced_trunc = P_trunc' * (data .- μ_trunc)
            recon_trunc = P_trunc * reduced_trunc .+ μ_trunc
            D_trunc = vec(var(data .- recon_trunc; dims=2))

            reduced_full = pca_predict(pca, data)
            recon_full = reconstruct(pca, reduced_full)
            D_full = vec(var(data .- recon_full; dims=2))

            @test D_trunc ≈ D_full atol = 1e-8
        end

        # d > n branch, large enough to exercise Arpack's iterative eigs path on the
        # smaller n x n Gram matrix (smaller dimension > 100)
        let d = 400, n = 150, rank = 5
            data = randn(d, n)
            P_trunc, μ_trunc = StructuredGaussianMixtures._truncated_pca(data, rank)
            @test size(P_trunc) == (d, rank)

            pca = pca_fit(PCA, data; maxoutdim=rank)
            P_full = projection(pca)
            μ_full = pca_mean(pca)

            @test μ_trunc ≈ μ_full atol = 1e-10
            @test P_trunc * P_trunc' ≈ P_full * P_full' atol = 1e-8

            reduced_trunc = P_trunc' * (data .- μ_trunc)
            recon_trunc = P_trunc * reduced_trunc .+ μ_trunc
            D_trunc = vec(var(data .- recon_trunc; dims=2))

            reduced_full = pca_predict(pca, data)
            recon_full = reconstruct(pca, reduced_full)
            D_full = vec(var(data .- recon_full; dims=2))

            @test D_trunc ≈ D_full atol = 1e-8
        end

        # d > n branch (dense fallback path, smaller n x n Gram matrix)
        let d = 200, n = 30, rank = 5
            data = randn(d, n)
            P_trunc, μ_trunc = StructuredGaussianMixtures._truncated_pca(data, rank)
            @test size(P_trunc) == (d, rank)

            pca = pca_fit(PCA, data; maxoutdim=rank)
            P_full = projection(pca)
            μ_full = pca_mean(pca)

            @test μ_trunc ≈ μ_full atol = 1e-10
            @test P_trunc * P_trunc' ≈ P_full * P_full' atol = 1e-8

            reduced_trunc = P_trunc' * (data .- μ_trunc)
            recon_trunc = P_trunc * reduced_trunc .+ μ_trunc
            D_trunc = vec(var(data .- recon_trunc; dims=2))

            reduced_full = pca_predict(pca, data)
            recon_full = reconstruct(pca, reduced_full)
            D_full = vec(var(data .- recon_full; dims=2))

            @test D_trunc ≈ D_full atol = 1e-8
        end
    end

    @testset "FactorEM" begin
        # Test FactorEM constructor
        factorem = FactorEM(3, 5)
        @test factorem.n_components == 3
        @test factorem.rank == 5
        @test factorem.initialization_method == :kmeans
        @test factorem.nInit == 1
        @test factorem.nIter == 10
        @test factorem.nInternalIter == 10

        # Test FactorEM with custom parameters
        factorem_custom = FactorEM(
            4, 6; initialization_method=:rand, nInit=3, nIter=5, nInternalIter=7
        )
        @test factorem_custom.n_components == 4
        @test factorem_custom.rank == 6
        @test factorem_custom.initialization_method == :rand
        @test factorem_custom.nInit == 3
        @test factorem_custom.nIter == 5
        @test factorem_custom.nInternalIter == 7

        # Test FactorEM fitting (unweighted)
        gmm = StructuredGaussianMixtures.fit(factorem, X)
        @test length(gmm.components) == 3
        @test length(gmm.prior.p) == 3
        @test sum(gmm.prior.p) ≈ 1.0 atol = 1e-10

        # Test that components are LRDMvNormal
        for comp in gmm.components
            @test comp isa LRDMvNormal
            @test StructuredGaussianMixtures.rank(comp) == 5
        end

        # Test logpdf on fitted model
        logps = logpdf(gmm, X)
        @test size(logps) == (n_samples,)
        @test all(isfinite, logps)

        # Test weighted fitting
        weights = rand(n_samples)
        weights ./= sum(weights)  # normalize
        gmm_weighted = StructuredGaussianMixtures.fit(factorem, X, weights)
        @test length(gmm_weighted.components) == 3

        # Test with invalid initialization method
        factorem_invalid = FactorEM(2, 3; initialization_method=:invalid)
        @test_throws ArgumentError StructuredGaussianMixtures.fit(factorem_invalid, X)
    end

    @testset "Edge Cases and Error Handling" begin
        # Test with single feature
        single_feature = randn(1, n_samples)
        gmm_single_feature = StructuredGaussianMixtures.fit(EM(2), single_feature)
        @test length(gmm_single_feature.components) == 2

        # Test with very small data
        tiny_data = randn(5, 10)
        gmm_tiny = StructuredGaussianMixtures.fit(EM(2), tiny_data)
        @test length(gmm_tiny.components) == 2
    end

    @testset "Mixture Model Properties" begin
        # Test that fitted models have reasonable properties
        gmm = StructuredGaussianMixtures.fit(EM(3; kind=:diag, nInit=2, nIter=3), X)

        # Test component means are finite
        for comp in gmm.components
            @test all(isfinite, mean(comp))
        end

        # Test component covariances are positive definite
        for comp in gmm.components
            Σ = cov(comp)
            @test all(eigvals(Σ) .> -1e-10)  # allow small numerical errors
        end

        # Test mixture weights are valid
        @test all(gmm.prior.p .> 0)
        @test sum(gmm.prior.p) ≈ 1.0 atol = 1e-10

        # Test that model can generate samples
        samples = rand(gmm, 10)
        @test size(samples) == (n_features, 10)

        # Test that logpdf works on samples
        logps = logpdf(gmm, samples)
        @test size(logps) == (10,)
        @test all(isfinite, logps)
    end
end
