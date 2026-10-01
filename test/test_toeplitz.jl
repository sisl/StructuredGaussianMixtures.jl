using Test, Random, Distributions, LinearAlgebra, Statistics
@testset "Toeplitz covariance" begin
    rng=MersenneTwister(712)
    for p in (1, 2, 7, 20), rho in (-0.8, 0.0, 0.7, 0.99)
        c=2 .* rho .^ (0:(p - 1))
        μ=randn(rng, p)
        g=ToeplitzMvNormal(μ, c)
        dense=MvNormal(μ, Symmetric([c[abs(i - j) + 1] for i in 1:p, j in 1:p]))
        X=randn(rng, p, 6)
        @test cov(g) ≈ cov(dense)
        @test mean(g)==μ
        @test var(g)==var(dense)
        @test Distributions.logdetcov(g) ≈ logdet(cov(dense)) atol=1e-10
        @test logpdf(g, X) ≈ logpdf(dense, X)
        @test logpdf(g, X[:, 1]) ≈ logpdf(dense, X[:, 1])
        out=zeros(6)
        logpdf!(out, g, X)
        @test out ≈ logpdf(dense, X)
        @test g.W*cov(g)*g.W' ≈ Matrix(I, p, p) atol=1e-10
        @test size(rand(rng, g, 5))==(p, 5)
        @test length(rand(rng, g))==p
    end
    μ=zeros(5)
    c=0.7 .^ (0:4)
    g=ToeplitzMvNormal(μ, c)
    μ[1]=100
    c[1]=100
    @test mean(g)==zeros(5)
    @test var(g)==ones(5)
    sampled=rand(rng, g, 30000)
    @test cov(sampled; dims=2) ≈ cov(g) atol=0.03
    dense=MvNormal(mean(g), Symmetric(cov(g)))
    for idx in ([1, 3, 5], [4, 1])
        @test cov(marginal(g, idx)) ≈ cov(marginal(dense, idx))
    end
    conditional=predict(g, [0.4, -0.1], [1, 4], [2, 5])
    reference=predict(dense, [0.4, -0.1], [1, 4], [2, 5])
    @test mean(conditional) ≈ mean(reference)
    @test cov(conditional) ≈ cov(reference)
    @test predict(g, Float64[], Int[], [1, 3]) isa MvNormal
    @test predict(g, [0.1]) isa MvNormal
    @test_throws ArgumentError predict(g, [0.1], [1], [1, 2])
    @test_throws ArgumentError predict(g, [NaN], [1], [2])
    @test_throws ArgumentError marginal(g, [1, 1])
    @test_throws ArgumentError marginal(g, Int[])
    @test_throws DimensionMismatch ToeplitzMvNormal(zeros(2), [1.0])
    @test_throws PosDefException ToeplitzMvNormal(zeros(2), [1.0, 2.0])
    @test_throws ArgumentError ToeplitzMvNormal(zeros(2), [NaN, 0.0])
    # Independently finite-difference the lag gradient for a non-Toeplitz scatter.
    A=randn(rng, 4, 4)
    S=A*A'+I
    c=[2.0, 0.4, 0.2, -0.1]
    value, gradient=StructuredGaussianMixtures._toeplitz_objective_gradient(c, S)
    independent(t) = begin
        T=[t[abs(i - j) + 1] for i in 1:4, j in 1:4]
        logdet(Symmetric(T))+tr(T\S)
    end
    @test value ≈ independent(c)
    for j in 1:4
        delta=zeros(4)
        delta[j]=1e-6
        @test gradient[j] ≈ (independent(c+delta)-independent(c-delta))/2e-6 rtol=1e-6
    end
    # Dimension two has a closed-form constrained optimum in fixed eigenvectors.
    X=[2.0 1.0 -2.0 3.0 0.0; -1.0 2.0 0.0 1.0 3.0]
    counts=[1, 3, 2, 4, 1]
    w=counts/sum(counts)
    method=ToeplitzMLE(regularization=0.03, maxiter=3000, tol=1e-7)
    result=fit(ToeplitzCovariance(), method, X; weights=counts, report=true)
    μ=X*w
    R=X .- μ
    S=(R .* w')*R'
    expected=[tr(S)/2+0.03 S[1, 2]; S[1, 2] tr(S)/2+0.03]
    @test cov(result.model) ≈ expected rtol=1e-5
    @test mean(result.model) ≈ μ
    @test result.report.status==:converged
    @test minimum(diff(result.report.history)) >= -1e-12
    @test result.report.objective_kind==:penalized_covariance_loglikelihood
    @test result.report.observed_objective ≈ dot(w, logpdf(result.model, X))
    @test result.report.objective ≈ dot(w, logpdf(result.model, X))-0.03/2*tr(inv(expected)) rtol=1e-6
    duplicate=X[:, vcat([fill(i, counts[i]) for i in eachindex(counts)]...)]
    @test cov(fit(ToeplitzCovariance(), method, duplicate)) ≈ cov(result.model)
    @test cov(fit(ToeplitzCovariance(), method, X; weights=counts .* 1e100)) ≈
        cov(result.model)
    state=workspace(ToeplitzCovariance(), method, result.model)
    @test cov(fit!(state, method, X; weights=counts)) ≈ cov(result.model)
    @test state.model !== result.model
    zero=fit(ToeplitzCovariance(), ToeplitzMLE(maxiter=0), X; report=true)
    @test zero.report.status==:iteration_limit
    failed=fit(
        ToeplitzCovariance(),
        ToeplitzMLE(max_backtracks=1, maxiter=1, tol=0),
        [0.0 0.0 10.0; 0.0 1.0 20.0];
        report=true,
    )
    @test failed.report.status==:failed
    failing_method=EM(
        covariance_method=ToeplitzMLE(max_backtracks=1, maxiter=1, tol=0), maxiter=1
    )
    failing_data=[0.0 0.0 10.0; 0.0 1.0 20.0]
    failing_state=initialize(
        MixtureSpec(ToeplitzCovariance(), 1), failing_method, failing_data; rng
    )
    before=failing_state.model
    fit!(failing_state, failing_method, failing_data)
    @test failing_state.report.status==:failed
    @test failing_state.model === before
    @test occursin("line search", failing_state.report.message)
    @test_throws ArgumentError fit(ToeplitzCovariance(), Exact(), X)
    @test_throws ArgumentError workspace(
        ToeplitzCovariance(), method, MvNormal(zeros(2), 1.0)
    )
    @test_throws ArgumentError ToeplitzMLE(regularization=-1)
    @test_throws ArgumentError ToeplitzMLE(tol=NaN)
    @test_throws ArgumentError ToeplitzMLE(max_backtracks=0)
    # In higher dimensions diagonal averaging is generally not stationary.
    A=randn(rng, 4, 40)
    A[1, :] .*= 3
    S=A*A'/40
    averaged=[mean(diag(S, k)) for k in 0:3]
    initial=ToeplitzMvNormal(zeros(4), averaged)
    optimized, report=StructuredGaussianMixtures._covariance(
        ToeplitzCovariance(),
        ToeplitzMLE(regularization=0, maxiter=1000),
        initial,
        A,
        fill(1/40, 40),
    )
    @test logdet(cov(optimized))+tr(cov(optimized)\S) <
        logdet(cov(initial))+tr(cov(initial)\S)-1e-5
    @test isposdef(cov(optimized))
    X=rand(rng, g, 160)
    spec=MixtureSpec(ToeplitzCovariance(), 2)
    outer=EM(covariance_method=ToeplitzMLE(maxiter=20), maxiter=3)
    state=initialize(spec, outer, X; rng)
    mixture=fit!(state, outer, X)
    @test state.report.iterations==3
    @test all(c->c isa ToeplitzMvNormal, components(mixture))
    @test sum(responsibilities(mixture, X); dims=2) ≈ ones(160)
    @test isfinite(logpdf(predict(mixture, [0.1], [1], [2, 3]), zeros(2)))
    @test workspace(spec, outer, mixture) isa MixtureWorkspace
end
