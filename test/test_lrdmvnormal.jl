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

    # Test additional methods
    @test StructuredGaussianMixtures.rank(dist) == r
    @test low_rank_factor(dist) == F
    @test diagonal(dist) == D

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

    # Test rand functionality
    @testset "rand" begin
        # Test single sample
        x1 = rand(dist)
        @test length(x1) == n
        @test eltype(x1) == Float64
        
        # Test multiple samples
        X1 = rand(dist, 5)
        @test size(X1) == (n, 5)
        @test eltype(X1) == Float64
        
        # Test with custom RNG
        rng = MersenneTwister(123)
        x2 = rand(rng, dist)
        @test length(x2) == n
        
        X2 = rand(rng, dist, 3)
        @test size(X2) == (n, 3)
        
        # Test in-place rand!
        x3 = similar(μ)
        rand!(rng, dist, x3)
        @test length(x3) == n
        
        X3 = similar(X1)
        rand!(rng, dist, X3)
        @test size(X3) == size(X1)
        
        # Test that samples have reasonable properties
        n_samples = 1000
        samples = rand(dist, n_samples)
        sample_mean = mean(samples, dims=2)[:]
        sample_cov = cov(samples, dims=2)
        
        @test norm(sample_mean - μ) < 1.0  # mean should be close (relaxed tolerance)
        @test norm(sample_cov - cov(dist)) < 20.0  # covariance should be reasonable (relaxed tolerance)
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

    # Test error handling and edge cases
    @testset "error handling" begin
        # Test dimension mismatch
        @test_throws DimensionMismatch LRDMvNormal(μ[1:end-1], F, D)
        @test_throws DimensionMismatch LRDMvNormal(μ, F[1:end-1, :], D)
        @test_throws DimensionMismatch LRDMvNormal(μ, F, D[1:end-1])
        
        # Test rank too large
        @test_throws ArgumentError LRDMvNormal(μ, randn(n, n), D)
        
        # Test with negative diagonal elements
        bad_D = copy(D)
        bad_D[1] = -1.0
        @test_throws ArgumentError LRDMvNormal(μ, F, bad_D)
        
        # Test with zero diagonal elements
        zero_D = copy(D)
        zero_D[1] = 0.0
        @test_throws ArgumentError LRDMvNormal(μ, F, zero_D)
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
        
        # Test with custom indices
        input_idx = [1, 3, 5, 7, 9]
        output_idx = [2, 4, 6, 8, 10]
        x_input = randn(length(input_idx))
        cond_dist_custom = predict(dist, x_input, input_idx, output_idx)
        @test length(cond_dist_custom) == length(output_idx)
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
        
        # Test with custom indices
        custom_indices = [5, 10, 15, 20, 25]
        marg_dist_custom = marginal(dist, custom_indices)
        @test length(marg_dist_custom) == length(custom_indices)
        @test mean(marg_dist_custom) ≈ μ[custom_indices]
    end
    
    # Test edge cases for small dimensions
    @testset "small dimensions" begin
        # Test with rank 1
        tiny_n = 3
        tiny_r = 1
        tiny_μ = randn(tiny_n)
        tiny_F = randn(tiny_n, tiny_r)
        tiny_D = abs.(randn(tiny_n)) .+ 1e-6
        
        tiny_dist = LRDMvNormal(tiny_μ, tiny_F, tiny_D)
        @test StructuredGaussianMixtures.rank(tiny_dist) == 1
        @test length(tiny_dist) == 3
        
        # Test with rank 0 (diagonal only)
        diag_only_dist = LRDMvNormal(tiny_μ, zeros(tiny_n, 0), tiny_D)
        @test StructuredGaussianMixtures.rank(diag_only_dist) == 0
        @test cov(diag_only_dist) ≈ Diagonal(tiny_D)
    end
end 