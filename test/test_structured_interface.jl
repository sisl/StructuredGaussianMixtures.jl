using Test, Random, LinearAlgebra, Statistics, Distributions
using StructuredGaussianMixtures

@testset "Shared structured Gaussian interface" begin
    rng = MersenneTwister(882)
    for r in (0, 2), kind in (:lrd, :latent)
        μ, F, D = randn(rng, 7), randn(rng, 7, r), 0.5 .+ rand(rng, 7)
        A_factor = randn(rng, r, r)
        g = kind == :lrd ? LRDMvNormal(μ, F, D) : LatentMvNormal(μ, F, D, A_factor)
        A = kind == :lrd ? Matrix{Float64}(I, r, r) : A_factor * A_factor'
        @test loading(g) == F
        @test latent_covariance(g) ≈ A
        @test size(latent_covariance(g)) == (r, r)
        @test latent_covariance_factor(g) * latent_covariance_factor(g)' ≈ A
        @test low_rank_factor(g) ≈ loading(g) * latent_covariance_factor(g)
        @test cov(g) ≈ F * A * F' + Diagonal(D)
        @test var(g) ≈ diag(cov(g))
        @test size(g) == (7,)
        @test StructuredGaussianMixtures.rank(g) == r
        @test hasproperty(g, :F)
        @test !hasproperty(g, :L)

        dense = MvNormal(mean(g), Symmetric(cov(g)))
        X = randn(rng, 7, 11)
        out = zeros(11)
        @test logpdf!(out, g, X) === out
        @test out ≈ logpdf(dense, X)
        @test logpdf(g, X[:, 1]) ≈ logpdf(dense, X[:, 1])
        @test_throws DimensionMismatch logpdf!(zeros(10), g, X)
        @test rand(MersenneTwister(9), g, 4) == rand(MersenneTwister(9), g, 4)
        sample = zeros(7)
        @test rand!(rng, g, sample) === sample
        @test all(isfinite, sample)

        # Constructors, accessors, marginals and conditionals never alias parameters.
        μ_before, Σ_before = mean(g), cov(g)
        μ .= 99
        F .= 99
        D .= 99
        A_factor .= 99
        for accessor in (
            mean,
            loading,
            diagonal,
            low_rank_factor,
            latent_covariance_factor,
            latent_covariance,
            var,
        )
            value = accessor(g)
            # Only mutate stored entries of a Diagonal identity.
            if value isa Diagonal
                value.diag .= 0
            else
                value .= 0
            end
            @test mean(g) == μ_before
            @test cov(g) ≈ Σ_before
        end
        m = marginal(g, [7, 2, 4, 1])
        c = predict(g, [0.2], [3], [7, 2, 4, 1])
        @test mean(m) ≈ μ_before[[7, 2, 4, 1]]
        @test cov(m) ≈ Σ_before[[7, 2, 4, 1], [7, 2, 4, 1]]
        @test cov(predict(g, Float64[], Int[], [7, 2])) ≈ Σ_before[[7, 2], [7, 2]]
        m.F .= 0
        m.D .= 99
        c.F .= 0
        c.D .= 99
        @test cov(g) ≈ Σ_before
        @test_throws ArgumentError marginal(g, [1, 1])
        @test_throws ArgumentError marginal(g, [0])
        @test_throws ArgumentError marginal(g, Int[])
        @test_throws ArgumentError predict(g, [0.0], [1], [1])
        @test_throws ArgumentError predict(g, [0.0, 0.0], [1, 1], [2])
        @test_throws ArgumentError predict(g, [0.0], [1], [2, 2])
        @test_throws ArgumentError predict(g, [0.0], [8], [2])
        @test_throws ArgumentError predict(g, [0.0], [1], Int[])
        @test_throws DimensionMismatch predict(g, Float64[], [1], [2])
    end
    for bad in (NaN, Inf)
        @test_throws ArgumentError LRDMvNormal([bad, 0.0], ones(2, 1), ones(2))
        @test_throws ArgumentError LRDMvNormal(zeros(2), fill(bad, 2, 1), ones(2))
        @test_throws ArgumentError LRDMvNormal(zeros(2), ones(2, 1), fill(bad, 2))
    end
    # Both constructors accept abstract array inputs and own their Float64 storage.
    μ, F, D = zeros(Float32, 4), ones(Float32, 4, 1), ones(Float32, 4)
    for g in (
        LRDMvNormal(view(μ, :), view(F, :, :), view(D, :)),
        LatentMvNormal(view(μ, :), view(F, :, :), view(D, :), ones(Float32, 1, 1)),
    )
        @test eltype(g) == Float64
        @test mean(g) == zeros(4)
        @test loading(g) == ones(4, 1)
    end
end
