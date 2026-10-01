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
    for b in (0, 1, 3, 7), ridge in (0.0, 0.02)
        g=fit(BandedPrecision(b), Exact(; regularization=ridge), X; weights)
        @test mean(g) ≈ mu
        C=cov(g)
        Q=invcov(g)
        # Convex precision MLE stationarity on every free band entry.
        for i in 1:8, j in 1:8
            if abs(i-j)<=b
                @test C[i, j] ≈ S[i, j]+(i==j ? ridge : 0) atol=1e-10
            else
                @test Q[i, j]==0
            end
        end
        ids=vcat([fill(j, weights[j]) for j in 1:120]...)
        @test cov(fit(BandedPrecision(b), Exact(; regularization=ridge), X[:, ids])) ≈ C
        state=workspace(BandedPrecision(b), Exact(), g)
        @test cov(fit!(state, Exact(; regularization=ridge), X; weights)) ≈ C
        @test state.report.status==:converged
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
end
