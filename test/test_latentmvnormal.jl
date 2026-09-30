using Test, Random, LinearAlgebra, Distributions
using StructuredGaussianMixtures

@testset "LatentMvNormal" begin
    rng = MersenneTwister(912)
    for (p, r) in ((12, 3), (3, 5), (6, 0))
        μ, F, D = randn(rng, p), randn(rng, p, r), 0.3 .+ rand(rng, p)
        A_factor = randn(rng, r, r) # deliberately nonsymmetric, not triangular
        g = LatentMvNormal(μ, F, D, A_factor)
        Σ = F * A_factor * A_factor' * F' + Diagonal(D)
        dense = MvNormal(copy(μ), Symmetric(Σ))
        X = randn(rng, p, 19)
        @test size(g) == (p,)
        @test length(g) == p
        @test StructuredGaussianMixtures.rank(g) == r
        @test mean(g) == μ
        @test cov(g) ≈ Σ
        @test low_rank_factor(g) ≈ F * A_factor
        @test logpdf(g, X) ≈ logpdf(dense, X) atol=1e-9
        @test logpdf(g, X[:, 1]) ≈ logpdf(dense, X[:, 1]) atol=1e-9
        @test logpdf(g, X) ≈ [logpdf(g, x) for x in eachcol(X)]
        @test isempty(logpdf(g, zeros(p, 0)))
        @test_throws DimensionMismatch logpdf(g, zeros(p + 1))
        @test_throws DimensionMismatch logpdf(g, zeros(p + 1, 2))

        obs, target = [p, 1], [2] # omit other unobserved coordinates
        x = randn(rng, 2)
        c = predict(g, x, obs, target)
        expected_mean = μ[target] + Σ[target, obs] * (Σ[obs, obs] \ (x - μ[obs]))
        expected_cov = Σ[target, target] - Σ[target, obs] * (Σ[obs, obs] \ Σ[obs, target])
        @test c isa LatentMvNormal
        @test mean(c) ≈ expected_mean atol=1e-9
        @test cov(c) ≈ expected_cov atol=1e-9
        @test loading(c) == F[target, :]
        @test cov(predict(g, Float64[], Int[], target)) ≈ Σ[target, target]
        @test cov(marginal(g, [p, 2, 1])) ≈ Σ[[p, 2, 1], [p, 2, 1]]
        @test length(predict(g, [0.0])) == p - 1
        @test_throws ArgumentError predict(g, [0.0], [1], [1])
        @test_throws ArgumentError predict(g, [0.0, 1.0], [1, 1], [2])
        @test_throws ArgumentError predict(g, [0.0], [0], [2])
        @test_throws ArgumentError predict(g, [0.0], [1], Int[])
        @test_throws DimensionMismatch predict(g, [0.0], obs, target)
        @test_throws ArgumentError marginal(g, [1, 1])
        @test_throws ArgumentError marginal(g, Int[])

        # Constructor and derived distributions own their arrays.
        F[1, :] .= 99
        A_factor .= 99
        D .= 99
        μ .= 99
        @test cov(g) ≈ Σ
        lc = loading(g)
        lc .= 0
        bc = latent_covariance_factor(g)
        bc .= 0
        dc = diagonal(g)
        dc .= 0
        mc = mean(g)
        mc .= 0
        @test cov(g) ≈ Σ
        @test mean(g) == mean(dense)
        @test size(rand(rng, g, 7)) == (p, 7)
        @test length(rand(rng, g)) == p
    end

    @testset "identity, singular factors, and LRD equivalence" begin
        μ, F, D = randn(rng, 8), randn(rng, 8, 3), ones(8)
        for A_factor in (Matrix{Float64}(I, 3, 3), zeros(3, 3), [1.0 2 0; 0 0 0; 0 0 0])
            g = LatentMvNormal(μ, F, D, A_factor)
            lrd = LRDMvNormal(μ, F * A_factor, D)
            X = randn(rng, 8, 10)
            @test logpdf(g, X) ≈ logpdf(lrd, X)
            @test cov(predict(g, [0.2, -0.5], [1, 3], [2, 4, 5, 6])) ≈
                cov(predict(lrd, [0.2, -0.5], [1, 3], [2, 4, 5, 6]))
        end
        @test_throws DimensionMismatch LatentMvNormal(μ, F, D, ones(2, 2))
        @test_throws DimensionMismatch LatentMvNormal(μ[1:2], F, D, ones(3, 3))
        @test_throws ArgumentError LatentMvNormal(μ, F, zeros(8), ones(3, 3))
        @test_throws ArgumentError LatentMvNormal(μ, F, fill(Inf, 8), ones(3, 3))
        @test_throws ArgumentError LatentMvNormal(μ, F, D, fill(NaN, 3, 3))
    end

    @testset "sampling moments" begin
        g = LatentMvNormal(
            [1.0, -2, 3], [1.0 0; 0 1; 1 1], [0.3, 0.5, 0.8], [1.0 0.2; -0.3 0.7]
        )
        samples = rand(rng, g, 60000)
        @test vec(mean(samples; dims=2)) ≈ mean(g) atol=0.035
        @test cov(samples; dims=2) ≈ cov(g) atol=0.06
    end

    @testset "mixture posterior" begin
        gs = [
            LatentMvNormal(randn(rng, 7), randn(rng, 7, 2), ones(7), randn(rng, 2, 2)) for
            _ in 1:3
        ]
        mix = MixtureModel(gs, [0.2, 0.3, 0.5])
        obs, target, x = [5, 1], [2, 7], [0.1, -0.7]
        post = predict(mix, x, obs, target)
        refs = map(gs) do g
            μ, Σ = mean(g), cov(g)
            m = μ[target] + Σ[target, obs] * (Σ[obs, obs] \ (x - μ[obs]))
            S = Σ[target, target] - Σ[target, obs] * (Σ[obs, obs] \ Σ[obs, target])
            MvNormal(m, Symmetric(S))
        end
        logw =
            log.(probs(mix)) +
            [logpdf(MvNormal(mean(g)[obs], Symmetric(cov(g)[obs, obs])), x) for g in gs]
        w = exp.(logw .- maximum(logw))
        reference = MixtureModel(refs, w / sum(w))
        @test probs(post) ≈ probs(reference)
        @test mean(post) ≈ mean(reference)
        @test cov(post) ≈ cov(reference)
        Y = randn(rng, 2, 5)
        @test logpdf(post, Y) ≈ logpdf(reference, Y)
        @test size(rand(rng, post, 7)) == (2, 7)
    end

    @testset "PCAEM decomposition and ownership" begin
        X = randn(rng, 9, 150)
        P, μ = StructuredGaussianMixtures._truncated_pca(X, 3)
        D = vec(var(X - (P * (P' * (X .- μ)) .+ μ); dims=2))
        # Re-run the reduced-space fit with the same seed to verify the mapping.
        Random.seed!(97)
        reduced = StructuredGaussianMixtures.fit(EM(2; nInit=2, nIter=3), P' * (X .- μ))
        Random.seed!(97)
        model = StructuredGaussianMixtures.fit(PCAEM(2, 3; gmm_nInit=2, gmm_nIter=3), X)
        for (g, z) in zip(components(model), components(reduced))
            @test g isa LatentMvNormal
            @test loading(g) ≈ P
            @test diagonal(g) ≈ D
            @test mean(g) ≈ μ + P * mean(z)
            @test cov(g) ≈ P * cov(z) * P' + Diagonal(D)
        end
        @test probs(model) ≈ probs(reduced)
        a, b = components(model)
        before = cov(b)
        a.F .= 0 # Even direct field mutation cannot affect another component.
        a.D .= 7
        @test cov(b) == before
    end
end
