# Shared-loading latent covariance EM. Posterior moments are computed without
# forming an observation-space scatter matrix. The block statistics also support
# future sharing patterns, but only shared F,D is exposed here.
function _check(s::LatentCovariance, m::CovarianceMethod, p)
    m isa CovarianceEM || throw(ArgumentError("LatentCovariance requires CovarianceEM"))
    return s.r <= p || throw(ArgumentError("latent rank exceeds observation dimension"))
end
function _check(s::MixtureSpec{<:LatentCovariance}, m::EM, p)
    Set(s.tied.parameters)==Set((:F, :D)) ||
        throw(ArgumentError("latent EM requires exactly Tied(:F,:D)"))
    return _check(s.covariance, m.covariance_method, p)
end
_remean(g::LatentMvNormal, μ) = LatentMvNormal(μ, g.F, g.D, g.A_factor)

function _validate_latent(s, g)
    g isa LatentMvNormal && rank(g)==s.r ||
        throw(ArgumentError("model does not match latent structure"))
    A=latent_covariance(g)
    cholesky(Symmetric(A)) # Fitting requires a nonsingular latent prior.
    if s.latent isa DiagonalCovariance
        isapprox(A, Diagonal(diag(A)); atol=0, rtol=1e-12) ||
            throw(ArgumentError("latent covariance must be diagonal"))
    end
    return nothing
end
function workspace(
    s::LatentCovariance, m::CovarianceMethod, g::Distributions.AbstractMvNormal
)
    _check(s, m, length(g))
    _validate_latent(s, g)
    return GaussianWorkspace(s, deepcopy(g), FitReport())
end
function _validate_latent_collection(s, current)
    length(current)==s.k || throw(DimensionMismatch("component count mismatch"))
    for g in current
        _validate_latent(s.covariance, g)
        g.F==first(current).F && g.D==first(current).D ||
            throw(ArgumentError("latent components must share equal F and D"))
    end
end
function workspace(s::MixtureSpec{<:LatentCovariance}, m::EM, g::MixtureModel)
    _check(s, m, length(first(components(g))))
    _validate_latent_collection(s, components(g))
    return MixtureWorkspace(s, deepcopy(g), FitReport(), FitReport[])
end
function initialize(
    s::LatentCovariance,
    m::CovarianceMethod,
    X::AbstractMatrix;
    weights=nothing,
    rng=Random.default_rng(),
)
    X, w=_data(X, weights)
    _check(s, m, size(X, 1))
    μ=_mean(X, w)
    D=max.(_variance(X .- μ, w), max(m.variance_floor, eps(Float64)))
    F=0.1 .* sqrt.(D) .* randn(rng, size(X, 1), s.r) ./ sqrt(s.r)
    return GaussianWorkspace(
        s, LatentMvNormal(μ, F, D, Matrix{Float64}(I, s.r, s.r)), FitReport()
    )
end

# Work in whitened prior coordinates to avoid explicitly inverting A or D.
function _latent_moments(g::LatentMvNormal, R, w)
    Q=g.F*g.A_factor
    chol=cholesky(Symmetric(I+Q'*(Q ./ g.D)))
    posterior=g.A_factor*(chol \ (Q'*(R ./ g.D)))
    V=g.A_factor*(chol \ g.A_factor')
    C=(R .* w')*posterior'
    H=Symmetric((posterior .* w')*posterior'+V)
    return C, Matrix(H)
end
function _joint_latent_covariance(s, m, current, residual, weights, mass)
    F=copy(first(current).F)
    D=copy(first(current).D)
    gs=[LatentMvNormal(zeros(size(F, 1)), F, D, g.A_factor) for g in current]
    variances=[_variance(residual(k), weights[k]) for k in eachindex(current)]
    objective(gs) =
        sum(mass[k]*_objective(gs[k], residual(k), weights[k]) for k in eachindex(gs))
    report=FitReport(; kind=:covariance_loglikelihood)
    previous=objective(gs)
    isfinite(previous) || throw(ArgumentError("nonfinite joint covariance objective"))
    push!(report.history, previous)
    for iteration in 1:m.maxiter
        moments=[_latent_moments(gs[k], residual(k), weights[k]) for k in eachindex(gs)]
        cross=sum(mass[k]*moments[k][1] for k in eachindex(gs))
        second=sum(mass[k]*moments[k][2] for k in eachindex(gs))
        F=cross/cholesky(Symmetric(second))
        D=zeros(size(F, 1))
        factors=Matrix{Float64}[]
        for k in eachindex(gs)
            C, H=moments[k]
            D .+=
                mass[k] .*
                (variances[k]+vec(sum((F*H) .* F; dims=2))-2vec(sum(C .* F; dims=2)))
            A=s.latent isa DiagonalCovariance ? Diagonal(diag(H)) : Symmetric(H)
            push!(factors, Matrix(cholesky(A).L))
        end
        D=max.(D, m.variance_floor)
        gs=[LatentMvNormal(zeros(size(F, 1)), F, D, factor) for factor in factors]
        value=objective(gs)
        isfinite(value) || throw(ArgumentError("nonfinite joint covariance objective"))
        push!(report.history, value)
        report.iterations=iteration
        if m.tol>0 && abs(value-previous)<=m.tol*(1+abs(previous))
            report.status=:converged
            break
        end
        previous=value
    end
    report.status==:initialized && (report.status=:iteration_limit)
    report.objective=last(report.history)
    return gs, report
end
function _covariance(s::LatentCovariance, m::CovarianceEM, current, R, w)
    _validate_latent(s, current)
    gs, report=_joint_latent_covariance(s, m, [current], _ -> R, [w], [1.0])
    return only(gs), report
end
function _fit_components(
    s::MixtureSpec{<:LatentCovariance}, m::CovarianceEM, current, X, component_weights
)
    _validate_latent_collection(s, current)
    mass=vec(sum(component_weights; dims=1))
    all(>(0), mass) || throw(ArgumentError("components must have positive mass"))
    mass ./= sum(mass)
    weights=[component_weights[:, k]/sum(component_weights[:, k]) for k in 1:s.k]
    means=[_mean(X, w) for w in weights]
    residual=k -> X .- means[k]
    gs, report=_joint_latent_covariance(s.covariance, m, current, residual, weights, mass)
    return [_remean(gs[k], means[k]) for k in 1:s.k], [report]
end
