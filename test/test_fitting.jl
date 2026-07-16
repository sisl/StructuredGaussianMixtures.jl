using Test
using LinearAlgebra
using Random
using Distributions
using StructuredGaussianMixtures
using GaussianMixtures

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

    @testset "FactorEM data-driven initialization" begin
        # initialize_gmm must produce finite, correctly shaped F and strictly
        # positive D for both initialization methods and a range of ranks.
        for method in (:kmeans, :rand), r in (0, 2, 5)
            init = StructuredGaussianMixtures.initialize_gmm(method, 3, r, X)
            @test length(init.components) == 3
            @test sum(init.prior.p) ≈ 1.0 atol = 1e-10
            for comp in init.components
                @test comp isa LRDMvNormal
                @test size(comp.F) == (n_features, r)
                @test all(isfinite, comp.F)
                @test all(comp.D .> 0)
                @test all(isfinite, comp.D)
            end
        end

        # The data-driven factor must be non-zero on data with real structure
        # (so the m_step! zero-fallback never fires on normal inputs).
        init_structured = StructuredGaussianMixtures.initialize_gmm(:kmeans, 3, 3, X)
        @test any(comp -> !all(iszero, comp.F), init_structured.components)
    end

    @testset "FactorEM non-regression on low-rank data" begin
        # Generate data with genuine low-rank-plus-diagonal cluster structure:
        # X = F_true * z + cluster mean + diagonal noise.
        Random.seed!(20240709)
        d = 15
        rank_true = 3
        n_clusters = 3
        n_per = 400
        cluster_cols = Matrix{Float64}[]
        for _ in 1:n_clusters
            μ_k = 5.0 .* randn(d)
            F_true = randn(d, rank_true)
            noise_var = 0.3 .* abs.(randn(d)) .+ 0.05
            z = randn(rank_true, n_per)
            cluster_cols = push!(
                cluster_cols, F_true * z .+ μ_k .+ (sqrt.(noise_var) .* randn(d, n_per))
            )
        end
        Xstruct = hcat(cluster_cols...)
        n = size(Xstruct, 2)
        w = ones(n) / n

        # Data-driven init reaches a good absolute mean log-likelihood on data
        # whose structure it is designed to capture.
        factorem = FactorEM(n_clusters, rank_true; nIter=15, nInternalIter=15)
        gmm_new = StructuredGaussianMixtures.fit(factorem, Xstruct, w)
        ll_new = logpdf(gmm_new, Xstruct)' * w
        @test isfinite(ll_new)
        @test ll_new >= -20.0  # comfortably above the poorly-fit regime

        # Compare against the historic 0.1*I initialization run through the same
        # EM loop: the data-driven init must not regress (>= old, within noise).
        function old_identity_init(K, r, data)
            dim, m = size(data)
            km = StructuredGaussianMixtures.kmeans(data, K)
            a = StructuredGaussianMixtures.assignments(km)
            centers = km.centers
            comps = Vector{LRDMvNormal}(undef, K)
            weights = zeros(K)
            for k in 1:K
                mask = a .== k
                cd = data[:, mask]
                μ = isempty(cd) ? centers[:, k] : vec(mean(cd; dims=2))
                D = isempty(cd) ? vec(var(data; dims=2)) : vec(var(cd; dims=2))
                F = 0.1 * Matrix{Float64}(I, dim, r)
                comps[k] = LRDMvNormal(μ, F, D)
                weights[k] = count(mask) / m
            end
            return MixtureModel(comps, weights)
        end

        Random.seed!(555)
        gmm_old = old_identity_init(n_clusters, rank_true, Xstruct)
        for _ in 1:15
            lr = StructuredGaussianMixtures.e_step(gmm_old, Xstruct)
            StructuredGaussianMixtures.m_step!(gmm_old, Xstruct, lr, w; nInternalIter=15)
        end
        ll_old = logpdf(gmm_old, Xstruct)' * w

        @test ll_new >= ll_old - 1e-2  # no meaningful regression at convergence

        # With only a couple of EM iterations, the data-driven init should be
        # strictly better than 0.1*I because it starts near the true structure.
        Random.seed!(777)
        gmm_new_fast = StructuredGaussianMixtures.initialize_gmm(
            :kmeans, n_clusters, rank_true, Xstruct
        )
        Random.seed!(888)
        gmm_old_fast = old_identity_init(n_clusters, rank_true, Xstruct)
        for _ in 1:2
            for g in (gmm_new_fast, gmm_old_fast)
                lr = StructuredGaussianMixtures.e_step(g, Xstruct)
                StructuredGaussianMixtures.m_step!(g, Xstruct, lr, w; nInternalIter=3)
            end
        end
        ll_new_fast = logpdf(gmm_new_fast, Xstruct)' * w
        ll_old_fast = logpdf(gmm_old_fast, Xstruct)' * w
        @test ll_new_fast > ll_old_fast
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
