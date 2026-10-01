# Covariance fitting consumes weighted centered observations, never requiring a
# dense p × p statistic for low-rank structures.
function _exact_report(g, quad)
    report=FitReport(; kind=:covariance_loglikelihood)
    report.status=:converged
    report.iterations=1
    report.objective=-0.5*(length(g)*log(2π)+logdet(g.Σ)+quad)
    push!(report.history, report.objective)
    return report
end
function _covariance(s::FullCovariance, m::Exact, current, R, w)
    scatter=(R .* w')*R'
    S=Symmetric(scatter+m.regularization*I)
    g=MvNormal(zeros(size(R, 1)), S)
    quad=m.regularization==0 ? length(g) : tr(g.Σ \ scatter)
    return g, _exact_report(g, quad)
end
function _covariance(s::DiagonalCovariance, m::Exact, current, R, w)
    v=_variance(R, w)
    variances=v .+ m.regularization
    g=MvNormal(zeros(size(R, 1)), Diagonal(variances))
    return g, _exact_report(g, sum(v ./ variances))
end
function _covariance(s::LowRankDiagonal, m::CovarianceEM, current, R, w)
    F=copy(current.F)
    D=copy(current.D)
    v=_variance(R, w)
    report=FitReport(; kind=:covariance_loglikelihood)
    g=LRDMvNormal(zeros(size(R, 1)), F, D)
    previous=_objective(g, R, w)
    push!(report.history, previous)
    for iteration in 1:m.maxiter
        chol=cholesky(Symmetric(I+F'*(F ./ D)))
        latent_mean=chol \ (F'*(R ./ D))
        cross=(R .* w')*latent_mean'
        second=(latent_mean .* w')*latent_mean' + (chol \ Matrix{Float64}(I, s.r, s.r))
        F=cross / cholesky(Symmetric(second))
        D=max.(
            v + vec(sum((F*second) .* F; dims=2)) - 2vec(sum(cross .* F; dims=2)),
            m.variance_floor,
        )
        g=LRDMvNormal(zeros(size(R, 1)), F, D)
        value=_objective(g, R, w)
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
    return g, report
end

"""Continue fitting initialized state without invoking initialization strategies.
Returns the fitted distribution; inspect `state.report` for diagnostics.
"""
function fit!(
    state::GaussianWorkspace, m::CovarianceMethod, X::AbstractMatrix; weights=nothing
)
    X, w=_data(X, weights)
    _check(state.spec, m, size(X, 1))
    length(state.model)==size(X, 1) ||
        throw(DimensionMismatch("workspace dimension mismatch"))
    μ=_mean(X, w)
    g, inner=_covariance(state.spec, m, state.model, X .- μ, w)
    state.model=_remean(g, μ)
    state.report=inner
    return state.model
end

function _covariance(s::IsotropicCovariance, m::Exact, current, R, w)
    v=sum(_variance(R, w))/size(R, 1)
    variance=v+m.regularization
    g=MvNormal(zeros(size(R, 1)), sqrt(variance))
    return g, _exact_report(g, length(g)*v/variance)
end

# Component weights retain their masses: tied updates must pool within-component
# statistics before normalization, rather than average component covariances.
function _fit_components(s::MixtureSpec, m::CovarianceMethod, current, X, weights)
    fitted=Distributions.AbstractMvNormal[]
    reports=FitReport[]
    for k in 1:s.k
        state=GaussianWorkspace(s.covariance, current[k], FitReport())
        push!(fitted, fit!(state, m, X; weights=view(weights, :, k)))
        push!(reports, state.report)
    end
    return fitted, reports
end
function _fit_components(
    s::MixtureSpec{S}, m::Exact, current, X, weights
) where {S<:Union{FullCovariance,DiagonalCovariance,IsotropicCovariance}}
    isempty(s.tied.parameters) && return invoke(
        _fit_components,
        Tuple{MixtureSpec,CovarianceMethod,Any,Any,Any},
        s,
        m,
        current,
        X,
        weights,
    )
    masses=vec(sum(weights; dims=1))
    total=sum(masses)
    means=X*weights ./ masses'
    p=size(X, 1)
    scatter=s.covariance isa FullCovariance ? zeros(p, p) : zeros(p)
    for k in 1:s.k
        R=X .- view(means, :, k)
        w=view(weights, :, k)
        if s.covariance isa FullCovariance
            scatter .+= (R .* w')*R'
        else
            scatter .+= _variance(R, w)
        end
    end
    scatter ./= total
    g=if s.covariance isa FullCovariance
        MvNormal(zeros(p), Symmetric(scatter+m.regularization*I))
    elseif s.covariance isa DiagonalCovariance
        MvNormal(zeros(p), Diagonal(scatter .+ m.regularization))
    else
        MvNormal(zeros(p), sqrt(sum(scatter)/p+m.regularization))
    end
    quad=if s.covariance isa FullCovariance
        tr(g.Σ \ scatter)
    else
        sum(scatter ./ var(g))
    end
    # One report describes the joint covariance solve, not k independent solves.
    return [_remean(g, means[:, k]) for k in 1:s.k], [_exact_report(g, quad)]
end
