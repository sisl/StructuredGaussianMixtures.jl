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
