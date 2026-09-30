using MultivariateStats:
    PCA, fit as pca_fit, projection, mean as pca_mean, predict as pca_predict, reconstruct
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
