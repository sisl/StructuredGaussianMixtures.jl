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
end

@testset "Native graphical lasso fitting" begin
    rng=MersenneTwister(91)
    X=[1.0 0.6 0.0; 0.0 1.0 0.2; 0.0 0.0 1.0]*randn(rng,3,500)
    w=rand(rng,500); w/=sum(w)
    mu=X*w; R=X .- mu; S=(R .* w')*R'
    method=GraphicalLasso(penalty=0.1, kkt_tol=1e-8,zero_tol=1e-9)
    result=fit(SparsePrecision(),method,X;weights=w,report=true)
    @test result.report.status==:converged
    @test mean(result.model) ≈ mu
    Q=Matrix(Distributions.invcov(result.model))
    @test StructuredGaussianMixtures._glasso_kkt(Q,S+method.regularization*I,method.penalty)<3e-8
    @test result.report.observed_objective ≈ dot(w,logpdf(result.model,X))
    @test isposdef(Symmetric(Q))
    exact=fit(SparsePrecision(),GraphicalLasso(penalty=0, regularization=0),X;weights=w)
    @test cov(exact) ≈ S rtol=1e-6
    diagonal=fit(SparsePrecision(),GraphicalLasso(penalty=10),X;weights=w)
    @test isdiag(Distributions.invcov(diagonal))
    @test cov(diagonal) ≈ Diagonal(diag(S).+1e-6)
    state=workspace(SparsePrecision(),method,result.model)
    fit!(state,method,X;weights=w)
    @test cov(state.model) ≈ cov(result.model) rtol=1e-6
    shifted=X .+ 0.05randn(rng,size(X))
    fit!(state,method,shifted;weights=w)
    cold=fit(SparsePrecision(),method,shifted;weights=w)
    @test cov(state.model) ≈ cov(cold) rtol=1e-6
    singular=vcat(X,X[1:1,:])
    robust=fit(SparsePrecision(),method,singular;report=true)
    @test robust.report.status==:converged
    @test isposdef(Symmetric(cov(robust.model)))
    limited=fit(SparsePrecision(),GraphicalLasso(penalty=0.03,maxiter=1,inner_maxiter=1,kkt_tol=1e-12),randn(rng,20,50);report=true)
    @test limited.report.status==:iteration_limit
    @test isposdef(Symmetric(cov(limited.model)))
    @test cov(fit(SparsePrecision(),method,hcat(X,fill(1e9,3));weights=vcat(w,0))) ≈ cov(result.model)
    @test_throws ArgumentError GraphicalLasso(maxiter=0)
    scalar=fit(SparsePrecision(),method,X[1:1,:];weights=w)
    @test cov(scalar)[1,1] ≈ S[1,1]+method.regularization
    mixture=initialize(MixtureSpec(SparsePrecision(),2),EM(covariance_method=method),X;rng=MersenneTwister(12))
    fit!(mixture,EM(covariance_method=method,maxiter=2,tol=0),X;weights=w)
    @test mixture.report.status==:iteration_limit
    @test mixture.report.objective ≈ dot(w,logpdf(mixture.model,X))
    @test all(r->r.status==:converged,mixture.inner_reports)
    # Analytic 2x2 dual: soft-threshold the empirical off diagonal.
    X2=X[1:2,:]
    mu2=X2*w; R2=X2.-mu2; S2=(R2.*w')*R2'
    expected=copy(S2)+method.regularization*I
    expected[1,2]=expected[2,1]=sign(S2[1,2])*max(abs(S2[1,2])-method.penalty,0)
    @test cov(fit(SparsePrecision(),method,X2;weights=w)) ≈ expected rtol=1e-6
    # Thresholding away a genuine edge remains SPD, but violates KKT. Retain
    # the converged unthresholded solution instead of exhausting the budget.
    aggressive=fit(SparsePrecision(),GraphicalLasso(penalty=0.1,zero_tol=100,kkt_tol=1e-8),X2;weights=w,report=true)
    @test aggressive.report.status==:converged
    @test cov(aggressive.model) ≈ expected rtol=1e-6
    @test !isdiag(Distributions.invcov(aggressive.model))
end
