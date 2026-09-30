"""Fit from fresh initialization. Returns a distribution, or `(model, report)`
with `report=true`. Methods may specialize initialization and fitting orchestration.
"""
function fit(
    s::GaussianStructure,
    m::CovarianceMethod,
    X::AbstractMatrix;
    weights=nothing,
    rng=Random.default_rng(),
    report=false,
)
    state=initialize(s, m, X; weights, rng)
    model=fit!(state, m, X; weights)
    return report ? (model=model, report=state.report) : model
end
function fit(
    s::MixtureSpec,
    m::EM,
    X::AbstractMatrix;
    weights=nothing,
    rng=Random.default_rng(),
    report=false,
)
    X, w=_data(X, weights)
    _check(s, m, size(X, 1))
    s.k<=size(X, 2) ||
        throw(ArgumentError("components exceed positive-weight observation count"))
    best=nothing
    runs=FitReport[]
    for run in 1:m.n_init
        try
            state=initialize(s, m, X; weights=w, rng)
            fit!(state, m, X; weights=w)
            push!(runs, state.report)
            if state.report.status!=:failed &&
                (best===nothing || state.report.objective>best.report.objective)
                best=state
            end
        catch err
            err isa Union{PosDefException,SingularException,ArgumentError,DomainError} ||
                rethrow()
            failure=FitReport()
            failure.status=:failed
            failure.message=sprint(showerror, err)
            push!(runs, failure)
        end
    end
    best===nothing && error("all EM runs failed: "*join([r.message for r in runs], "; "))
    best.report=deepcopy(best.report)
    best.report.runs=runs
    return report ? (model=best.model, report=best.report) : best.model
end

"""Copy a fitted model into initialized state. No initialization strategy is run.
Use `workspace(spec, method, model)` followed by `fit!`. PCAEM requires its retained
PCAWorkspace because an observed distribution does not uniquely identify its offset.
"""
function workspace(
    s::GaussianStructure, m::CovarianceMethod, g::Distributions.AbstractMvNormal
)
    _check(s, m, length(g))
    valid=if s isa FullCovariance
        g isa MvNormal
    elseif s isa DiagonalCovariance
        g isa MvNormal && isdiag(cov(g))
    else
        g isa LRDMvNormal && rank(g)==s.r
    end
    valid || throw(ArgumentError("model does not match structure"))
    return GaussianWorkspace(s, deepcopy(g), FitReport())
end
function workspace(s::MixtureSpec, m::EM, g::MixtureModel)
    _check(s, m, length(first(components(g))))
    length(components(g))==s.k || throw(DimensionMismatch("component count mismatch"))
    for c in components(g)
        workspace(s.covariance, m.covariance_method, c)
    end
    return MixtureWorkspace(s, deepcopy(g), FitReport(), FitReport[])
end
