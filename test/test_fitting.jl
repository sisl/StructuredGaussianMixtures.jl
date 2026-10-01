using Test, Random, Distributions, LinearAlgebra, Statistics
using StructuredGaussianMixtures
const sgmfit=StructuredGaussianMixtures.fit

@testset "Fitting architecture" begin
    rng=MersenneTwister(202)
    X=randn(rng, 5, 80)
    w=rand(rng, 80)
    w ./= sum(w)
    @testset "Weighted exact Gaussian fits" begin
        μ=X*w
        R=X .- μ
        S=(R .* w')*R'
        for s in (FullCovariance(), DiagonalCovariance())
            result=sgmfit(s, Exact(regularization=0), X; weights=w, report=true)
            @test mean(result.model) ≈ μ
            @test cov(result.model) ≈ (s isa FullCovariance ? S : Diagonal(diag(S)))
            @test result.report.status==:converged
            @test result.report.iterations==1
            @test result.report.objective ≈ dot(w, logpdf(result.model, X))
            @test cov(sgmfit(s, Exact(), X; weights=1e200*w)) ≈
                cov(sgmfit(s, Exact(), X; weights=w))
            @test cov(sgmfit(s, Exact(), X; weights=ones(80))) ≈ cov(sgmfit(s, Exact(), X))
            @test cov(sgmfit(s, Exact(), hcat(X, fill(1e5, 5)); weights=vcat(w, 0))) ≈
                cov(sgmfit(s, Exact(), X; weights=w))
        end
        counts=repeat([1, 2, 3, 4], 20)
        duplicated=X[:, vcat([fill(i, counts[i]) for i in 1:80]...)]
        @test cov(sgmfit(FullCovariance(), Exact(), X; weights=counts)) ≈
            cov(sgmfit(FullCovariance(), Exact(), duplicated))
        @test cov(sgmfit(DiagonalCovariance(), Exact(), zeros(3, 2))) ≈ 1e-6*Matrix(I, 3, 3)
        @test_throws PosDefException sgmfit(
            FullCovariance(), Exact(regularization=0), zeros(3, 2)
        )
    end
    @testset "One native EM step against independent reference" begin
        spec=MixtureSpec(FullCovariance(), 2)
        initial=MixtureModel(
            [
                MvNormal(fill(-0.5, 5), Matrix{Float64}(I, 5, 5)),
                MvNormal(fill(0.8, 5), 2Matrix{Float64}(I, 5, 5)),
            ],
            [0.4, 0.6],
        )
        probabilities=[
            probs(initial)[j]*pdf(components(initial)[j], X[:, i]) for i in 1:80, j in 1:2
        ]
        probabilities ./= sum(probabilities; dims=2)
        @test responsibilities(initial, X) ≈ probabilities
        @test vec(sum(responsibilities(initial, X); dims=2)) ≈ ones(80)
        method=EM(covariance_method=Exact(regularization=0), maxiter=1, tol=0)
        state=workspace(spec, method, initial)
        result=fit!(state, method, X; weights=w)
        masses=probabilities'*w
        @test probs(result) ≈ masses
        for j in 1:2
            q=w .* probabilities[:, j]
            q ./= sum(q)
            μ=X*q
            centered=X .- μ
            @test mean(components(result)[j]) ≈ μ
            @test cov(components(result)[j]) ≈ (centered .* q')*centered'
        end
        @test state.report.objective ≈ dot(w, logpdf(result, X))
        @test state.report.objective >= first(state.report.history)-1e-10
        @test mean(components(initial)[1])==fill(-0.5, 5) # copied state
    end
    @testset "LRD covariance EM reference and continuation" begin
        s=LowRankDiagonal(2)
        m=CovarianceEM(maxiter=1, tol=0, variance_floor=0)
        F=randn(rng, 5, 2)*0.2
        D=ones(5)
        initial=LRDMvNormal(zeros(5), F, D)
        μ=X*w
        R=X .- μ
        G=inv(I+F'*Diagonal(1 ./ D)*F)
        Ez=G*F'*Diagonal(1 ./ D)*R
        Crz=(R .* w')*Ez'
        Czz=G+(Ez .* w')*Ez'
        expected_F=Crz*inv(Czz)
        expected_D=diag(
            (R .* w')*R' - expected_F*Crz' - Crz*expected_F' + expected_F*Czz*expected_F'
        )
        state=workspace(s, m, initial)
        g=fit!(state, m, X; weights=w)
        @test loading(g) ≈ expected_F
        @test diagonal(g) ≈ expected_D
        @test state.report.history[end] >= state.report.history[1]-1e-10
        fit!(state, m, X; weights=w)
        longer=workspace(s, m, initial)
        fit!(longer, CovarianceEM(maxiter=2, tol=0, variance_floor=0), X; weights=w)
        @test cov(state.model) ≈ cov(longer.model)
        @test all(>(0), diagonal(state.model))
        # Zero rank is the diagonal MLE, and n < p needs no dense scatter.
        z=sgmfit(LowRankDiagonal(0), CovarianceEM(), X; weights=w)
        @test cov(z) ≈
            cov(sgmfit(DiagonalCovariance(), Exact(regularization=0), X; weights=w))
        @test all(
            isfinite,
            logpdf(
                sgmfit(LowRankDiagonal(3), CovarianceEM(maxiter=3), randn(rng, 200, 20)),
                randn(rng, 200, 2),
            ),
        )
    end
    @testset "Mixture continuation, restart selection, and weighting" begin
        for (s, c) in (
            (FullCovariance(), Exact()),
            (DiagonalCovariance(), Exact()),
            (LowRankDiagonal(2), CovarianceEM(maxiter=2, tol=0)),
        )
            spec=MixtureSpec(s, 2)
            short=EM(covariance_method=c, maxiter=2, tol=0)
            state=initialize(spec, short, X; rng=MersenneTwister(41))
            full=deepcopy(state)
            fit!(state, short, X; weights=w)
            fit!(state, short, X; weights=w)
            fit!(
                full,
                EM(covariance_method=c, maxiter=4, tol=0, n_init=9, init=RandomInit()),
                X;
                weights=w,
            )
            @test logpdf(state.model, X) ≈ logpdf(full.model, X)
            # fit! ignores initializers and restart count, including covariance init.
            @test state.report.iterations==2
            @test full.report.iterations==4
            @test state.report.objective ≈ dot(w, logpdf(state.model, X))
            a=initialize(spec, short, X; weights=w, rng=MersenneTwister(71))
            b=initialize(
                spec,
                short,
                hcat(X, fill(1e5, 5));
                weights=vcat(w, 0),
                rng=MersenneTwister(71),
            )
            fit!(a, short, X; weights=w)
            fit!(b, short, X; weights=w*100)
            @test logpdf(a.model, X) ≈ logpdf(b.model, X)
            counts=repeat([1, 2, 3, 4], 20)
            duplicated=X[:, vcat([fill(i, counts[i]) for i in 1:80]...)]
            a=deepcopy(full)
            b=deepcopy(full)
            fit!(a, short, X; weights=counts)
            fit!(b, short, duplicated)
            @test logpdf(a.model, X) ≈ logpdf(b.model, X)
        end
        method=EM(n_init=3, maxiter=4, tol=0)
        result=sgmfit(
            MixtureSpec(FullCovariance(), 2), method, X; rng=MersenneTwister(4), report=true
        )
        @test length(result.report.runs)==3
        @test result.report.objective==maximum(
            r.objective for r in result.report.runs if r.status!=:failed
        )
        @test all(diff(result.report.history) .>= -1e-8)
        @test logpdf(result.model, X) ≈ logpdf(
            sgmfit(MixtureSpec(FullCovariance(), 2), method, X; rng=MersenneTwister(4)), X
        )
        @test all(
            isfinite,
            logpdf(
                sgmfit(
                    MixtureSpec(FullCovariance(), 2),
                    EM(init=RandomInit()),
                    randn(rng, 1, 50),
                ),
                randn(rng, 1, 5),
            ),
        )
        state=initialize(MixtureSpec(FullCovariance(), 2), EM(), X; rng)
        fit!(state, EM(maxiter=0), X)
        @test state.report.iterations==0
        @test state.report.status==:iteration_limit
        @test state.report.objective ≈ mean(logpdf(state.model, X))
        fit!(state, EM(tol=1e10), X)
        @test state.report.status==:converged
    end
    @testset "Single Gaussian versus one component" begin
        for (s, m) in (
            (FullCovariance(), Exact()),
            (LowRankDiagonal(2), CovarianceEM(maxiter=3, tol=0)),
        )
            state=initialize(s, m, X; rng)
            mix=workspace(
                MixtureSpec(s, 1),
                EM(covariance_method=m),
                MixtureModel([deepcopy(state.model)]),
            )
            fit!(state, m, X; weights=w)
            fit!(mix, EM(covariance_method=m, maxiter=1, tol=0), X; weights=w)
            @test mean(state.model) ≈ mean(first(components(mix.model)))
            @test cov(state.model) ≈ cov(first(components(mix.model)))
        end
    end
    @testset "PCA pipeline and retained state" begin
        for latent in (FullCovariance(), DiagonalCovariance())
            spec=MixtureSpec(LatentCovariance(2; latent), 2; tied=Tied(:D, :F))
            method=PCAEM(latent_method=EM(maxiter=2, tol=0, n_init=2))
            state=initialize(spec, method, X; weights=w, rng)
            F=copy(state.F)
            D=copy(state.D)
            offset=copy(state.offset)
            full=deepcopy(state)
            fit!(state, method, X; weights=w)
            fit!(state, method, X; weights=w)
            fit!(full, PCAEM(latent_method=EM(maxiter=4, tol=0)), X; weights=w)
            @test logpdf(state.model, X) ≈ logpdf(full.model, X)
            @test state.F==F && state.D==D && state.offset==offset
            @test state.report.objective_kind==:projected_loglikelihood
            @test state.report.observed_objective ≈ dot(w, logpdf(state.model, X))
            @test state.report.objective ≈
                dot(w, logpdf(state.latent.model, F'*(X .- offset)))
            for (g, z) in zip(components(state.model), components(state.latent.model))
                @test loading(g)==F && diagonal(g)==D
                @test mean(g) ≈ offset+F*mean(z)
                @test cov(g) ≈ F*cov(z)*F'+Diagonal(D)
            end
            result=sgmfit(spec, method, X; weights=w, rng=MersenneTwister(10), report=true)
            @test length(result.report.runs)==2
            @test all(isfinite, logpdf(result.model, X))
            # Projected fitting honors zero weights during PCA and initialization.
            b=sgmfit(
                spec,
                method,
                hcat(X, fill(1e5, 5));
                weights=vcat(w, 0),
                rng=MersenneTwister(10),
            )
            @test logpdf(result.model, X) ≈ logpdf(b, X)
        end
    end
    @testset "Fitted model operations and separated-cluster recovery" begin
        data=hcat(randn(rng, 4, 150) .- 4, randn(rng, 4, 150) .+ 4)
        for (spec, method) in (
            (MixtureSpec(FullCovariance(), 2), EM()),
            (MixtureSpec(DiagonalCovariance(), 2), EM()),
            (
                MixtureSpec(LowRankDiagonal(1), 2),
                EM(covariance_method=CovarianceEM(maxiter=5)),
            ),
            (MixtureSpec(LatentCovariance(1), 2; tied=Tied(:F, :D)), PCAEM()),
        )
            g=sgmfit(spec, method, data; rng=MersenneTwister(1))
            means=sort([mean(c)[1] for c in components(g)])
            @test means ≈ [-4, 4] atol=0.4
            @test all(isfinite, logpdf(g, data))
            @test size(rand(rng, g, 10))==(4, 10)
            posterior=predict(g, [0.2], [1], [2, 3])
            @test length(mean(posterior))==2
            @test isposdef(cov(posterior))
            @test sum(probs(posterior)) ≈ 1
            @test length(mean(marginal(g, [3, 1])))==2
            @test mean(marginal(g, [3, 1])) ≈ mean(g)[[3, 1]]
            @test cov(marginal(g, [3, 1])) ≈ cov(g)[[3, 1], [3, 1]]
        end
    end
    @testset "Validation and failure reporting" begin
        @test_throws ErrorException sgmfit(
            MixtureSpec(DiagonalCovariance(), 2),
            EM(covariance_method=Exact(regularization=0)),
            zeros(5, 10),
        )
        @test_throws ArgumentError sgmfit(MixtureSpec(FullCovariance(), 3), EM(), X[:, 1:2])
        pca = sgmfit(
            MixtureSpec(LatentCovariance(5), 1; tied=Tied(:F, :D)),
            PCAEM(),
            X;
            rng=MersenneTwister(2),
        )
        @test all(isfinite, logpdf(pca, X))

        @test_throws ArgumentError LowRankDiagonal(-1)
        @test_throws ArgumentError LatentCovariance(0)
        @test_throws ArgumentError MixtureSpec(FullCovariance(), 0)
        @test_throws ArgumentError Tied(:F, :F)
        @test_throws ArgumentError Exact(regularization=-1)
        @test_throws ArgumentError CovarianceEM(maxiter=-1)
        @test_throws ArgumentError EM(n_init=0)
        @test_throws ArgumentError EM(tol=NaN)
        for weights in (zeros(80), fill(-1.0, 80), fill(NaN, 80))
            @test_throws ArgumentError sgmfit(FullCovariance(), Exact(), X; weights)
        end
        @test_throws DimensionMismatch sgmfit(FullCovariance(), Exact(), X; weights=ones(2))
        @test_throws ArgumentError sgmfit(FullCovariance(), Exact(), fill(NaN, 2, 3))
        @test_throws ArgumentError sgmfit(FullCovariance(), Exact(), zeros(2, 0))
        @test_throws ArgumentError sgmfit(LowRankDiagonal(5), CovarianceEM(), X)
        @test_throws ArgumentError sgmfit(LowRankDiagonal(2), Exact(), X)
        @test_throws ArgumentError sgmfit(LatentCovariance(2), Exact(), X)
        @test_throws ArgumentError sgmfit(
            MixtureSpec(FullCovariance(), 2; tied=Tied(:F)), EM(), X
        )
        @test_throws ArgumentError sgmfit(MixtureSpec(LatentCovariance(2), 2), PCAEM(), X)
        @test_throws ArgumentError sgmfit(
            MixtureSpec(LatentCovariance(5), 2; tied=Tied(:F, :D)), PCAEM(), ones(5, 80)
        )
        @test_throws ArgumentError initialize(
            MixtureSpec(FullCovariance(), 3), EM(), X[:, 1:2]
        )
        spec=MixtureSpec(FullCovariance(), 2)
        g=MixtureModel([MvNormal(zeros(5), ones(5)), MvNormal(fill(1e6, 5), ones(5))])
        state=workspace(spec, EM(), g)
        fit!(state, EM(), X)
        @test state.report.status==:failed
        @test occursin("mass", state.report.message)
        @test state.report.iterations==0
        @test logpdf(state.model, X)==logpdf(g, X)
        @test_throws ErrorException sgmfit(
            spec, EM(covariance_method=Exact(regularization=0)), zeros(5, 10)
        )
        # Numerical failure before the first E-step must update diagnostics and
        # preserve the initialized model, just like failure during an M-step.
        state=workspace(spec, EM(), g)
        original=state.model
        @test fit!(state, EM(), fill(1e200, 5, 3)) === original
        @test state.report.status==:failed
        @test occursin("nonfinite", state.report.message)
        @test state.report.iterations==0
        @test state.report.objective == -Inf
        @test isempty(state.report.history)
        @test isempty(state.inner_reports)
        @test_throws ArgumentError fit!(state, EM(), fill(NaN, 5, 3))

        # A singular candidate fails without partially committing the M-step.
        single=MixtureSpec(FullCovariance(), 1)
        state=workspace(single, EM(), MixtureModel([MvNormal(zeros(5), ones(5))]))
        original=state.model
        @test fit!(state, EM(covariance_method=Exact(regularization=0)), zeros(5, 8)) ===
            original
        @test state.report.status==:failed
        @test state.report.iterations==0
        @test length(state.report.history)==1
        @test isfinite(state.report.objective)

        # Validation must still reject incompatible models without allocating a
        # discarded Gaussian workspace; successful copies must be independent.
        diagonal=MixtureSpec(DiagonalCovariance(), 1)
        correlated=MvNormal(zeros(2), [2.0 0.5; 0.5 1.0])
        @test_throws ArgumentError workspace(diagonal, EM(), MixtureModel([correlated]))
        source=MixtureModel([MvNormal(zeros(2), ones(2))])
        copied=workspace(diagonal, EM(), source)
        mean(first(components(copied.model)))[1]=7
        @test mean(first(components(source)))[1]==0
        @test_throws ArgumentError workspace(
            LowRankDiagonal(2), CovarianceEM(), MvNormal(zeros(5), ones(5))
        )
    end
end
include("test_pca_reference.jl")

# A solver can return its last valid iterate while reporting failure. The outer
# driver must not accept that candidate as a successful mixture update.
struct FailedCovarianceForTest <: StructuredGaussianMixtures.CovarianceMethod end
StructuredGaussianMixtures._check(::FullCovariance, ::FailedCovarianceForTest, p) = nothing
function StructuredGaussianMixtures._covariance(
    ::FullCovariance, ::FailedCovarianceForTest, current, R, w
)
    report=FitReport()
    report.status=:failed
    report.message="deliberate inner solver failure"
    return current, report
end
@testset "Reported inner solver failure" begin
    X=randn(MersenneTwister(61), 3, 12)
    spec=MixtureSpec(FullCovariance(), 1)
    initial=MixtureModel([MvNormal(zeros(3), Matrix{Float64}(I, 3, 3))])
    state=workspace(spec, EM(), initial)
    original=state.model
    fit!(state, EM(covariance_method=FailedCovarianceForTest(), maxiter=1), X)
    @test state.report.status==:failed
    @test occursin("deliberate inner solver failure", state.report.message)
    @test state.report.iterations==0
    @test state.model === original
end
