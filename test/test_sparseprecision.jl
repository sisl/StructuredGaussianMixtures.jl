using Test, Random, LinearAlgebra, SparseArrays, Distributions
using StructuredGaussianMixtures
@testset "Sparse precision Gaussian" begin
    rng=MersenneTwister(661)
    p=7
    Q=spdiagm(-1=>fill(-0.3, p-1), 0=>fill(2.0, p), 1=>fill(-0.3, p-1))
    μ=randn(rng, p)
    g=SparsePrecisionMvNormal(μ, Q)
    dense=MvNormal(μ, Symmetric(inv(Matrix(Q))))
    X=randn(rng, p, 11)
    @test Distributions.invcov(g) ≈ Q
    @test cov(g) ≈ cov(dense)
    @test var(g) ≈ var(dense)
    @test Distributions.logdetcov(g) ≈ logdet(cov(dense))
    @test Distributions.sqmahal(g, X[:, 1]) ≈ Distributions.sqmahal(dense, X[:, 1])
    @test Distributions.partype(g)==Float64
    @test Distributions.sqmahal(g, X) ≈ Distributions.sqmahal(dense, X)
    quadratic=zeros(size(X, 2))
    @test Distributions.sqmahal!(quadratic, g, X) === quadratic
    @test quadratic ≈ Distributions.sqmahal(dense, X)
    @test_throws DimensionMismatch Distributions.sqmahal(g, zeros(p-1, 3))
    @test_throws DimensionMismatch Distributions.sqmahal!(zeros(10), g, X)
    @test logpdf(g, X) ≈ logpdf(dense, X)
    buffer=zeros(11)
    logpdf!(buffer, g, X)
    @test buffer ≈ logpdf(dense, X)
    draws=rand(rng, g, 60000)
    @test mean(draws; dims=2)[:] ≈ μ atol=0.02
    @test cov(draws; dims=2) ≈ cov(dense) atol=0.02
    for (obs, target) in (([1, 3], [7, 2, 4, 5, 6]), ([3, 1], [7, 2]), (Int[], [7, 3, 2]))
        values=X[obs, 1]
        post=predict(g, values, obs, target)
        reference=if isempty(obs)
            marginal(dense, target)
        else
            predict(dense, values, obs, target)
        end
        @test mean(post) ≈ mean(reference)
        @test cov(post) ≈ cov(reference)
        @test (post isa SparsePrecisionMvNormal)==(length(obs)+length(target)==p)
    end
    @test cov(predict(g, X[1:2, 1])) ≈ cov(predict(dense, X[1:2, 1]))
    @test cov(marginal(g, [7, 2])) ≈ cov(marginal(dense, [7, 2]))
    mix=MixtureModel([g, SparsePrecisionMvNormal(μ .+ 1, Q)])
    @test sum(responsibilities(mix, X); dims=2) ≈ ones(11, 1)
    @test all(isfinite, logpdf(predict(mix, [0.0], [1], [3, 5]), zeros(2, 3)))
    Q[1, 1]=10
    μ[1]=10
    @test Distributions.invcov(g)[1, 1]==2
    @test mean(g) != μ
    owned=Distributions.invcov(g)
    owned[1, 1]=99
    @test Distributions.invcov(g)[1, 1]==2
    @test_throws ArgumentError SparsePrecisionMvNormal([0.0, 0.0], [1.0 1.0; 0.0 1.0])
    @test_throws PosDefException SparsePrecisionMvNormal([0.0, 0.0], [1.0 2.0; 2.0 1.0])
    @test_throws DimensionMismatch SparsePrecisionMvNormal(zeros(3), ones(2, 2))
    @test_throws ArgumentError marginal(g, [1, 1])
    @test_throws ArgumentError predict(g, [0.0], [1], [1])
    @test_throws DimensionMismatch logpdf(g, ones(2))
    @test_throws ArgumentError GraphicalLasso(penalty=-1)
    @test_throws ArgumentError GraphicalLasso(kkt_tol=0)
    if Base.get_extension(
        StructuredGaussianMixtures, :StructuredGaussianMixturesConvexExt
    )===nothing
        err=try
            fit(SparsePrecision(), GraphicalLasso(), X)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("using Convex", sprint(showerror, err))
    end
end
