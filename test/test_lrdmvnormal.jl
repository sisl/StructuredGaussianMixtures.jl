using Test
using LinearAlgebra
using Random
using Distributions
using StructuredGaussianMixtures

@testset "LRDMvNormal" begin
    # Set random seed for reproducibility
    Random.seed!(42)

    # Test dimensions
    n = 100  # dimension
    r = 5    # rank

    # Create test parameters
    μ = randn(n)
    F = randn(n, r)
    D = abs.(randn(n)) .+ 1e-6  # ensure positive diagonal

    # Create distribution
    dist = LRDMvNormal(μ, F, D)

    # Test basic properties
    @test length(dist) == n
    @test size(dist) == (n,)
    @test eltype(dist) == Float64

    # Test mean and covariance
    @test mean(dist) ≈ μ
    @test cov(dist) ≈ F * F' + Diagonal(D)

    # Test logpdf with various inputs
    @testset "logpdf" begin
        # Test single point
        x = randn(n)
        logp = logpdf(dist, x)
        @test isfinite(logp)
        @test logp isa Float64

        # Test multiple points
        X = randn(n, 10)
        logps = logpdf(dist, X)
        @test size(logps) == (10,)
        @test all(isfinite, logps)

        # Test against MvNormal for small dimensions
        small_n = 5
        small_r = 2
        small_μ = randn(small_n)
        small_F = randn(small_n, small_r)
        small_D = abs.(randn(small_n)) .+ 1e-6

        small_dist = LRDMvNormal(small_μ, small_F, small_D)
        small_mvn = MvNormal(small_μ, small_F * small_F' + Diagonal(small_D))
        
        small_x = randn(small_n)
        @test logpdf(small_dist, small_x) ≈ logpdf(small_mvn, small_x) atol=1e-10

        # Test edge cases
        @test logpdf(dist, μ) > logpdf(dist, μ .+ 10)  # closer to mean should have higher probability
        @test logpdf(dist, μ) > logpdf(dist, μ .- 10)

        # Test with zero F (should reduce to diagonal normal)
        zero_F_dist = LRDMvNormal(μ, zeros(n, r), D)
        diag_mvn = MvNormal(μ, Diagonal(D))
        @test cov(zero_F_dist) ≈ cov(diag_mvn)
        @test logpdf(zero_F_dist, x) ≈ logpdf(diag_mvn, x) atol=1e-10

        # Test against MvNormal for large dimensions
        large_n = 1000
        large_r = 10
        large_μ = randn(large_n)
        large_F = randn(large_n, large_r)
        large_D = abs.(randn(large_n)) .+ 1e-6

        large_dist = LRDMvNormal(large_μ, large_F, large_D)
        large_mvn = MvNormal(large_μ, large_F * large_F' + Diagonal(large_D))
        @test cov(large_dist) ≈ cov(large_mvn)
        @test mean(large_dist) ≈ mean(large_mvn)
        n_samples = 20
        large_x = randn(large_n, n_samples)
        full_pdfs = logpdf(large_mvn, large_x)
        lrd_pdfs = logpdf(large_dist, large_x)
        @test lrd_pdfs ≈ full_pdfs
    end

    # Test numerical stability
    @testset "numerical stability" begin
        # Test with very small D
        small_D = fill(1e-10, n)
        small_D_dist = LRDMvNormal(μ, F, small_D)
        @test isfinite(logpdf(small_D_dist, μ))

        # Test with very large D
        large_D = fill(1e10, n)
        large_D_dist = LRDMvNormal(μ, F, large_D)
        @test isfinite(logpdf(large_D_dist, μ))

        # Test with ill-conditioned F
        ill_F = F .* 1e10
        ill_D = D .* 1e-10
        ill_dist = LRDMvNormal(μ, ill_F, ill_D)
        @test isfinite(logpdf(ill_dist, μ))
    end

    # Test predict functionality
    @testset "predict" begin
        # Split dimensions
        n1 = 50
        n2 = n - n1
        x1 = randn(n1)
        
        # Get conditional distribution
        cond_dist = predict(dist, x1)
        
        # Test properties of conditional distribution
        @test length(cond_dist) == n2
        @test isfinite(logpdf(cond_dist, randn(n2)))
        
        # Test that conditional mean is correct
        Σ11 = cov(dist)[1:n1, 1:n1]
        Σ12 = cov(dist)[1:n1, n1+1:end]
        Σ22 = cov(dist)[n1+1:end, n1+1:end]
        expected_mean = μ[n1+1:end] + Σ12' * (Σ11 \ (x1 - μ[1:n1]))
        @test mean(cond_dist) ≈ expected_mean atol=1e-10
    end

    # Test marginal functionality
    @testset "marginal" begin
        # Get marginal distribution
        indices = 1:10
        marg_dist = marginal(dist, indices)
        
        # Test properties of marginal distribution
        @test length(marg_dist) == length(indices)
        @test isfinite(logpdf(marg_dist, randn(length(indices))))
        
        # Test that marginal mean and covariance are correct
        @test mean(marg_dist) ≈ μ[indices]
        @test cov(marg_dist) ≈ cov(dist)[indices, indices]
    end
end 