using Test, Random, Distributions, LinearAlgebra, Statistics
@testset "Isotropic and tied exact fitting" begin
    rng=MersenneTwister(841)
    X=randn(rng, 4, 120)
    X[:, 1:50] .+= 4
    counts=repeat([1, 2, 3], 40)
    w=counts/sum(counts)
    replicated=X[:, vcat([fill(i, counts[i]) for i in eachindex(counts)]...)]
    ridge=0.03
    exact=Exact(regularization=ridge)
    μ=X*w
    centered=X .- μ
    variance=sum(abs2.(centered)*w)/4+ridge
    single=fit(IsotropicCovariance(), exact, X; weights=counts)
    @test mean(single) ≈ μ
    @test cov(single) ≈ variance*I
    @test cov(fit(IsotropicCovariance(), exact, replicated)) ≈ cov(single)
    @test cov(fit(IsotropicCovariance(), exact, X; weights=1e200*w)) ≈ cov(single)
    @test workspace(IsotropicCovariance(), exact, single).model !== single
    @test_throws ArgumentError workspace(
        IsotropicCovariance(), exact, MvNormal(zeros(4), Diagonal([1.0, 2, 1, 1]))
    )
    for structure in (FullCovariance(), DiagonalCovariance(), IsotropicCovariance())
        spec=MixtureSpec(structure, 2; tied=Tied(:covariance))
        method=EM(covariance_method=exact, maxiter=1, tol=0)
        initial=MixtureModel(
            [MvNormal(zeros(4), 2.0), MvNormal(fill(3.0, 4), 2.0)], [0.3, 0.7]
        )
        resp=responsibilities(initial, X)
        weights=resp .* w
        masses=vec(sum(weights; dims=1))
        means=X*weights ./ masses'
        scatter=zeros(4, 4)
        for k in 1:2
            residual=X .- means[:, k]
            scatter .+= (residual .* weights[:, k]')*residual'
        end
        expected=if structure isa FullCovariance
            scatter
        elseif structure isa DiagonalCovariance
            Diagonal(diag(scatter))
        else
            tr(scatter)/4*Matrix(I, 4, 4)
        end
        expected=expected+ridge*I
        state=workspace(spec, method, initial)
        model=fit!(state, method, X; weights=counts)
        @test state.report.iterations==1
        @test length(state.inner_reports)==1
        @test probs(model) ≈ masses
        for k in 1:2
            @test mean(components(model)[k]) ≈ means[:, k]
            @test cov(components(model)[k]) ≈ expected
        end
        @test components(model)[1].Σ === components(model)[2].Σ
        @test mean(components(model)[1]) != mean(components(model)[2])
        @test state.inner_reports[1].objective ≈
            sum(weights[:, k]'*logpdf(components(model)[k], X) for k in 1:2)
        repeated=fit!(workspace(spec, method, initial), method, replicated)
        @test cov(components(repeated)[1]) ≈ expected
        @test mean(components(repeated)[2]) ≈ means[:, 2]
        one=fit(
            MixtureSpec(structure, 1; tied=Tied(:covariance)),
            method,
            X;
            weights=counts,
            rng=MersenneTwister(1),
        )
        reference=fit(structure, exact, X; weights=counts)
        @test cov(only(components(one))) ≈ cov(reference)
        @test mean(only(components(one))) ≈ mean(reference)
        continued=fit!(workspace(spec, method, model), method, X; weights=counts)
        @test cov(components(continued)[1]) == cov(components(continued)[2])
        predicted=predict(continued, [0.2], [1], [2, 3])
        @test isfinite(logpdf(predicted, zeros(2)))
        @test vec(sum(responsibilities(continued, X); dims=2)) ≈ ones(120)
        initialized=initialize(spec, method, X; rng=MersenneTwister(42))
        @test components(initialized.model)[1].Σ === components(initialized.model)[2].Σ
        bad=MixtureModel([MvNormal(zeros(4), 1.0), MvNormal(ones(4), 2.0)])
        @test_throws ArgumentError workspace(spec, method, bad)
        @test_throws ArgumentError fit(MixtureSpec(structure, 2; tied=Tied(:F)), method, X)
        unregularized=EM(covariance_method=Exact(regularization=0), maxiter=8, tol=0)
        state=workspace(spec, unregularized, initial)
        fit!(state, unregularized, X; weights=counts)
        @test minimum(diff(state.report.history)) >= -1e-10
    end
    independent=fit(
        MixtureSpec(IsotropicCovariance(), 2), EM(maxiter=2), X; rng=MersenneTwister(2)
    )
    @test all(c->all(==(first(var(c))), var(c)), components(independent))
    @test_throws ArgumentError fit(IsotropicCovariance(), CovarianceEM(), X)
end
