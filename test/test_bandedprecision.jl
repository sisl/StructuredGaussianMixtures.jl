using Random, LinearAlgebra, Distributions, Statistics
@testset "Banded precision representation" begin
    rng=MersenneTwister(31)
    for p in (1, 5, 12), b in unique([0, min(2, p-1), p-1])
        L=zeros(p, p)
        for i in 1:p, j in max(1, i - b):i
            L[i, j]=i==j ? 1.5 : 0.15randn(rng)
        end
        Q=L*L'
        mu=randn(rng, p)
        g=BandedPrecisionMvNormal(mu, Q; bandwidth=b)
        dense=MvNormal(copy(mu), Symmetric(inv(Q)))
        @test cov(g) ≈ cov(dense)
        @test invcov(g) ≈ Q
        @test logdetcov(g) ≈ logdetcov(dense)
        X=randn(rng, p, 17)
        @test logpdf(g, X) ≈ logpdf(dense, X)
        @test sqmahal(g, X[:, 1]) ≈ sqmahal(dense, X[:, 1])
        @test size(rand(rng, g, 4))==(p, 4)
        @test length(rand(rng, g))==p
        mu[1]+=3
        @test mean(g) ≈ mean(dense)
        if p>1
            for (inputs, outputs) in
                (([1], collect(2:p)), ([2], [1]), ([p, 1], collect(2:(p - 1))))
                isempty(outputs) && continue
                x=randn(rng, length(inputs))
                conditional=predict(g, x, inputs, outputs)
                reference=predict(dense, x, inputs, outputs)
                @test mean(conditional) ≈ mean(reference)
                @test cov(conditional) ≈ cov(reference)
                @test logpdf(conditional, zeros(length(outputs))) ≈
                    logpdf(reference, zeros(length(outputs)))
            end
            @test cov(marginal(g, [p, 1])) ≈ cov(dense)[[p, 1], [p, 1]]
            @test cov(predict(g, Float64[], Int[], [p, 1])) ≈ cov(dense)[[p, 1], [p, 1]]
        end
    end
    g=BandedPrecisionMvNormal(zeros(2), Matrix{Float64}(I, 2, 2); bandwidth=0)
    @test_throws ArgumentError BandedPrecision(-1)
    @test_throws ArgumentError BandedPrecisionMvNormal(zeros(2), ones(2, 2); bandwidth=0)
    @test_throws PosDefException BandedPrecisionMvNormal(zeros(2), zeros(2, 2); bandwidth=1)
    @test_throws ArgumentError marginal(g, [1, 1])
    @test_throws ArgumentError predict(g, [0.0], [1], [1])
    @test_throws DimensionMismatch logpdf(g, zeros(3))
    @test_throws ArgumentError initialize(BandedPrecision(2), Exact(), zeros(2, 4))
    samples=rand(rng, g, 20000)
    @test maximum(abs.(mean(samples; dims=2)))<0.04
    @test cov(samples; dims=2) ≈ Matrix{Float64}(I, 2, 2) atol=0.04
end
@testset "Banded weighted MLE" begin
    rng=MersenneTwister(72)
    X=randn(rng, 8, 120)
    weights=rand(rng, 1:4, 120)
    w=weights/sum(weights)
    mu=X*w
    R=X .- mu
    S=(R .* w')*R'
    for b in (0, 1, 3, 7), ridge in (0.0, 0.02, 1e6)
        g=fit(BandedPrecision(b), Exact(; regularization=ridge), X; weights)
        @test mean(g) ≈ mu
        C=cov(g)
        Q=invcov(g)
        # Convex precision MLE stationarity on every free band entry.
        for i in 1:8, j in 1:8
            if abs(i-j)<=b
                @test C[i, j] ≈ S[i, j]+(i==j ? ridge : 0) atol=1e-10 rtol=1e-12
            else
                @test Q[i, j]==0
            end
        end
        ids=vcat([fill(j, weights[j]) for j in 1:120]...)
        @test cov(fit(BandedPrecision(b), Exact(; regularization=ridge), X[:, ids])) ≈ C
        state=workspace(BandedPrecision(b), Exact(), g)
        @test cov(fit!(state, Exact(; regularization=ridge), X; weights)) ≈ C
        @test state.report.status==:converged
        # The report uses compact sufficient statistics. Check it against
        # independent dense scoring, including a dominating regularizer.
        reference=MvNormal(mu, Symmetric(C))
        @test state.report.objective ≈ dot(logpdf(reference, X), w) rtol=1e-12
    end
    mixture=fit(
        MixtureSpec(BandedPrecision(2), 2), EM(; maxiter=3), X; rng=MersenneTwister(2)
    )
    @test all(c->c isa BandedPrecisionMvNormal, components(mixture))
    @test vec(sum(responsibilities(mixture, X); dims=2)) ≈ ones(120)
    @test isfinite(
        logpdf(predict(mixture, [0.0]; input_indices=[1], output_indices=[2, 3]), zeros(2))
    )
    state=workspace(MixtureSpec(BandedPrecision(2), 2), EM(), mixture)
    @test fit!(state, EM(; maxiter=1), X) isa MixtureModel
end

@testset "Banded edge cases and continuation" begin
    rng=MersenneTwister(3)
    Q=[3.0 0.4 0 0; 0.4 2.0 -0.2 0; 0 -0.2 2.0 0.5; 0 0 0.5 2.0]
    g=BandedPrecisionMvNormal([1.0, -1.0, 0.0, 2.0], Q; bandwidth=1)
    dense=MvNormal(mean(g), Symmetric(inv(Q)))
    out=[4, 1, 3]
    inp=[2]
    x=[0.2]
    c=predict(g, x, inp, out)
    ref=predict(dense, x, inp, out)
    @test cov(c) ≈ cov(ref)
    @test mean(c) ≈ mean(ref)
    @test cov(rand(rng, g, 30000); dims=2) ≈ cov(g) atol=0.03
    data=rand(rng, g, 300)
    state=initialize(MixtureSpec(BandedPrecision(1), 2), EM(), data; rng)
    fit!(state, EM(covariance_method=Exact(regularization=0), maxiter=10, tol=0), data)
    @test state.report.status==:iteration_limit
    @test all(diff(state.report.history) .>= -1e-10)
    singular=repeat(reshape(1.0:10.0, 1, :), 4, 1)
    @test_throws PosDefException fit(BandedPrecision(1), Exact(regularization=0), singular)
    model=MixtureModel([g])
    state=workspace(MixtureSpec(BandedPrecision(1), 1), EM(), model)
    initial=state.model
    fit!(state, EM(covariance_method=Exact(regularization=0), maxiter=1), singular)
    @test state.report.status==:failed
    @test state.model === initial
    regularized=fit(BandedPrecision(1), Exact(regularization=0.01), singular)
    @test isposdef(Symmetric(invcov(regularized)))
    @test all(isfinite, logpdf(regularized, singular))
    # A zero-weight distant sample must be removed before moment arithmetic.
    weighted=fit(
        BandedPrecision(1),
        Exact(),
        hcat(data, fill(1e200, 4));
        weights=[ones(size(data, 2)); 0.0],
    )
    reference=fit(BandedPrecision(1), Exact(), data)
    @test mean(weighted) ≈ mean(reference)
    @test cov(weighted) ≈ cov(reference)
    # For almost collinear data, direct tr(Q*S) summation can cancel large
    # positive and negative terms. The report must retain scoring accuracy.
    near_singular=repeat(randn(rng, 1, 200), 6, 1)+1e-5randn(rng, 6, 200)
    result=fit(BandedPrecision(3), Exact(regularization=1e-10), near_singular; report=true)
    @test result.report.objective ≈ mean(logpdf(result.model, near_singular)) rtol=1e-12
end

@testset "Banded mixture conditioning without observations" begin
    Q=[3.0 0.4 0 0; 0.4 2.0 -0.2 0; 0 -0.2 2.0 0.5; 0 0 0.5 2.0]
    cs=[
        BandedPrecisionMvNormal(zeros(4), Q; bandwidth=1),
        BandedPrecisionMvNormal([1.0, -2.0, 3.0, 0.0], 2Q; bandwidth=1),
    ]
    mixture=MixtureModel(cs, [0.3, 0.7])
    for outputs in ([4, 1], [4, 1, 3, 2])
        conditional=predict(mixture, Float64[]; input_indices=Int[], output_indices=outputs)
        reference=MixtureModel(
            [MvNormal(mean(c)[outputs], Symmetric(cov(c)[outputs, outputs])) for c in cs],
            [0.3, 0.7],
        )
        @test probs(conditional) ≈ probs(mixture)
        @test mean(conditional) ≈ mean(reference)
        @test cov(conditional) ≈ cov(reference)
        @test logpdf(conditional, ones(length(outputs))) ≈
            logpdf(reference, ones(length(outputs)))
        for (actual, expected) in zip(components(conditional), components(reference))
            @test mean(actual) ≈ mean(expected)
            @test cov(actual) ≈ cov(expected)
        end
    end
end

@testset "Banded batched quadratic forms" begin
    g=BandedPrecisionMvNormal([1.0, 2.0], [2.0 0.3; 0.3 1.0]; bandwidth=1)
    reference=MvNormal(mean(g), Symmetric(cov(g)))
    X=[1.0 2 3; 2.0 -1 0]
    @test sqmahal(g, X) ≈ sqmahal(reference, X)
    buffer=zeros(3)
    @test sqmahal!(buffer, g, X) === buffer
    @test buffer ≈ sqmahal(reference, X)
    @test_throws DimensionMismatch sqmahal!(zeros(2), g, X)
    @test_throws DimensionMismatch sqmahal!(zeros(3), g, zeros(3, 3))
end
