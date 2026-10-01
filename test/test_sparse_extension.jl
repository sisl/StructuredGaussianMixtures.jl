using Test, Random, LinearAlgebra, SparseArrays, Distributions
using StructuredGaussianMixtures, Convex, SCS
@testset "Convex graphical lasso extension" begin
    @test Base.get_extension(
        StructuredGaussianMixtures, :StructuredGaussianMixturesConvexExt
    )!==nothing
    rng=MersenneTwister(19)
    X=[1.0 0.5 0.0; 0.2 1.0 0.3; 0.0 0.2 1.0]*randn(rng, 3, 300)
    w=rand(rng, 300)
    w/=sum(w)
    optimizer=Convex.MOI.OptimizerWithAttributes(
        SCS.Optimizer, "eps_abs"=>1e-7, "eps_rel"=>1e-7, "max_iters"=>100000
    )
    m=GraphicalLasso(penalty=0.15, optimizer=optimizer, zero_tol=1e-5, kkt_tol=1e-4)
    result=fit(SparsePrecision(), m, X; weights=w, report=true)
    g=result.model
    μ=X*w
    R=X .- μ
    S=(R .* w')*R'
    @test mean(g) ≈ μ
    @test isposdef(Symmetric(Matrix(Distributions.invcov(g))))
    @test result.report.status==:converged
    @test result.report.objective_kind==:penalized_covariance_loglikelihood
    @test result.report.observed_objective ≈ dot(w, logpdf(g, X))
    Q=Matrix(Distributions.invcov(g))
    gradient=S+m.regularization*I-inv(Q)
    for j in 1:3, i in 1:3
        if i==j
            @test abs(gradient[i, j])<1e-3
        elseif Q[i, j]==0
            @test abs(gradient[i, j])<=m.penalty+1e-3
        else
            @test abs(gradient[i, j]+m.penalty*sign(Q[i, j]))<1e-3
        end
    end
    @test nnz(Distributions.invcov(g))<9
    exact=fit(
        SparsePrecision(),
        GraphicalLasso(penalty=0, regularization=0, optimizer=optimizer, kkt_tol=1e-4),
        X;
        weights=w,
    )
    @test cov(exact) ≈ S rtol=1e-3
    state=workspace(SparsePrecision(), m, g)
    fit!(state, m, X; weights=w)
    @test cov(state.model) ≈ cov(g) rtol=1e-3
    strong=fit(
        SparsePrecision(),
        GraphicalLasso(penalty=10, optimizer=optimizer, zero_tol=1e-5),
        X;
        weights=w,
    )
    @test isdiag(Distributions.invcov(strong))
    @test cov(strong) ≈ Diagonal(diag(S) .+ m.regularization) rtol=1e-3
    @test cov(fit(SparsePrecision(), m, hcat(X, fill(1e9, 3)); weights=vcat(w, 0))) ≈ cov(g) rtol=1e-3
    initial=initialize(
        MixtureSpec(SparsePrecision(), 2),
        EM(covariance_method=m),
        X;
        rng=MersenneTwister(9),
    )
    fit!(initial, EM(covariance_method=m, maxiter=2, tol=0), X)
    @test initial.report.status==:iteration_limit
    @test initial.report.objective ≈ mean(logpdf(initial.model, X))
    @test occursin("need not increase", initial.report.message)
    @test all(
        r->r.objective_kind==:penalized_covariance_loglikelihood, initial.inner_reports
    )
    broken=GraphicalLasso(
        optimizer=Convex.MOI.OptimizerWithAttributes(SCS.Optimizer, "max_iters"=>1)
    )
    old=initial.model
    fit!(initial, EM(covariance_method=broken), X)
    @test initial.report.status==:failed
    @test initial.model===old
    @test occursin("solver", initial.report.message)
    @test_throws ArgumentError fit(SparsePrecision(), GraphicalLasso(), X)
end
