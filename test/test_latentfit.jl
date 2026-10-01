using Random, LinearAlgebra, Distributions
const SGMlatent = StructuredGaussianMixtures
@testset "Joint latent covariance EM" begin
    rng=MersenneTwister(728)
    p, r, n=6, 2, 400
    F=randn(rng, p, r)
    D=0.5 .+ rand(rng, p)
    g=LatentMvNormal(zeros(p), F, D, [1.2 0.3; 0.0 0.7])
    X=rand(rng, g, n)
    w=rand(rng, n)
    w/=sum(w)
    C, H=SGMlatent._latent_moments(g, X, w)
    A=latent_covariance(g)
    B=A*F'/cov(g)
    V=A-B*F*A
    Z=B*X
    @test C ≈ (X .* w')*Z'
    @test H ≈ V+(Z .* w')*Z'
    # Independently derive one joint M-step with dense posterior conditioning.
    g2=LatentMvNormal(zeros(p), F, D, [0.6 0.1; 0.0 1.5])
    other=rand(rng, g2, n)
    C2B=latent_covariance(g2)*F'/cov(g2)
    Z2=C2B*other
    C2=(other .* w')*Z2'
    H2=latent_covariance(g2)-C2B*F*latent_covariance(g2)+(Z2 .* w')*Z2'
    pooledC=0.3C+0.7C2
    pooledH=0.3H+0.7H2
    expectedF=pooledC/pooledH
    expectedD=0.3vec(abs2.(X)*w) +
              0.7vec(abs2.(other)*w) +
              vec(sum((expectedF*pooledH) .* expectedF; dims=2))-2vec(
        sum(pooledC .* expectedF; dims=2)
    )
    updated, rep=SGMlatent._joint_latent_covariance(
        LatentCovariance(r),
        CovarianceEM(maxiter=1, variance_floor=0),
        [g, g2],
        k -> k==1 ? X : other,
        [w, w],
        [0.3, 0.7],
    )
    @test updated[1].F ≈ expectedF
    @test updated[1].D ≈ expectedD
    @test latent_covariance(updated[1]) ≈ H
    @test latent_covariance(updated[2]) ≈ H2
    @test rep.history[end] >= rep.history[1]-1e-9
    method=CovarianceEM(maxiter=8, tol=0, variance_floor=0)
    for latent in (FullCovariance(), DiagonalCovariance())
        spec=LatentCovariance(r; latent)
        state=initialize(spec, method, X; weights=w, rng=MersenneTwister(21))
        fit!(state, method, X; weights=w)
        @test all(diff(state.report.history) .>= -1e-9)
        @test mean(state.model) ≈ X*w
        @test logpdf(state.model, X) ≈
            logpdf(MvNormal(mean(state.model), Symmetric(cov(state.model))), X)
        latent isa DiagonalCovariance && @test isdiag(latent_covariance(state.model))
        mixture=MixtureSpec(spec, 1; tied=Tied(:F, :D))
        start=initialize(spec, method, X; weights=w, rng=MersenneTwister(21)).model
        single=workspace(spec, method, start)
        joint=workspace(
            mixture, EM(covariance_method=method, maxiter=1), MixtureModel([start])
        )
        fit!(single, method, X; weights=w)
        fit!(joint, EM(covariance_method=method, maxiter=1), X; weights=w)
        @test cov(single.model) ≈ cov(only(components(joint.model)))
        @test length(joint.inner_reports)==1
        @test joint.inner_reports[1].history ≈ single.report.history
    end
    spec=MixtureSpec(LatentCovariance(r), 2; tied=Tied(:F, :D))
    em=EM(covariance_method=method, maxiter=4, tol=0)
    state=initialize(spec, em, X; rng)
    fit!(state, em, X; weights=w)
    @test state.report.status==:iteration_limit
    @test all(diff(state.report.history) .>= -1e-9)
    cs=components(state.model)
    @test cs[1].F==cs[2].F
    @test cs[1].D==cs[2].D
    @test cs[1].F !== cs[2].F # Constructors retain independently owned arrays.
    copied=workspace(spec, em, state.model)
    fit!(copied, em, X; weights=w)
    @test copied.report.objective >= state.report.objective-1e-9
    @test cov(components(state.model)[1])==cov(cs[1])
    @test_throws ArgumentError initialize(MixtureSpec(LatentCovariance(r), 2), em, X)
    @test_throws ArgumentError initialize(
        MixtureSpec(LatentCovariance(r), 2; tied=Tied(:F)), em, X
    )
    @test_throws ArgumentError initialize(LatentCovariance(r), Exact(), X)
    @test_throws ArgumentError workspace(
        spec,
        em,
        MixtureModel([cs[1], LatentMvNormal(cs[2].μ, 2cs[2].F, cs[2].D, cs[2].A_factor)]),
    )
    @test_throws PosDefException workspace(
        LatentCovariance(r), method, LatentMvNormal(zeros(p), F, D, zeros(r, r))
    )
    # Integer weighting agrees with explicitly repeated samples from the same start.
    counts=rand(rng, 1:3, n)
    ids=reduce(vcat, [fill(j, counts[j]) for j in 1:n])
    a=workspace(LatentCovariance(r), method, g)
    b=workspace(LatentCovariance(r), method, g)
    fit!(a, method, X; weights=counts)
    fit!(b, method, X[:, ids])
    @test cov(a.model) ≈ cov(b.model)
    zero_method=CovarianceEM(maxiter=0)
    zero_state=workspace(LatentCovariance(r), zero_method, g)
    fit!(zero_state, zero_method, X)
    @test cov(zero_state.model) ≈ cov(g)
    @test zero_state.report.iterations==0
    @test zero_state.report.status==:iteration_limit
    converged=workspace(LatentCovariance(r), CovarianceEM(tol=1e10), g)
    fit!(converged, CovarianceEM(tol=1e10), X)
    @test converged.report.status==:converged
    @test converged.report.iterations==1
    floored=fit(LatentCovariance(r), CovarianceEM(variance_floor=2.0), X; rng)
    @test all(diagonal(floored) .>= 2.0)
    @test_throws ArgumentError workspace(
        LatentCovariance(r; latent=DiagonalCovariance()), method, g
    )
    # Orthogonal reparameterization of an A factor leaves posterior moments invariant.
    Q=Matrix(qr(randn(rng, r, r)).Q)
    rotated=LatentMvNormal(mean(g), g.F, g.D, g.A_factor*Q)
    Cr, Hr=SGMlatent._latent_moments(rotated, X, w)
    @test Cr ≈ C
    @test Hr ≈ H
    # Underflowed responsibilities exclude extreme observations before centering
    # and scoring the inner objective, rather than evaluating 0*Inf.
    far=fill(1e160, 2)
    extremes=hcat(zeros(2), far, zeros(2), far)
    separated=MixtureModel([
        LatentMvNormal(μ, zeros(2, 1), ones(2), ones(1, 1)) for μ in (zeros(2), far)
    ])
    separated_spec=MixtureSpec(LatentCovariance(1), 2; tied=Tied(:F, :D))
    separated_method=EM(covariance_method=CovarianceEM(maxiter=1), maxiter=1)
    separated_state=workspace(separated_spec, separated_method, separated)
    fit!(separated_state, separated_method, extremes)
    @test separated_state.report.status==:iteration_limit
    @test isfinite(separated_state.report.objective)
end
