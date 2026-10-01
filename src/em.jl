function _logjoints(g::MixtureModel, X)
    return hcat([_scores(c, X) .+ log(p) for (c, p) in zip(components(g), probs(g))]...)
end
function _lognormalizers(scores)
    rowmax=maximum(scores; dims=2)
    all(isfinite, rowmax) || throw(ArgumentError("nonfinite mixture density"))
    return rowmax .+ log.(sum(exp.(scores .- rowmax); dims=2))
end
"""Posterior component probabilities as samples × components (separate from conditional `predict`)."""
function responsibilities(g::MixtureModel, X::AbstractMatrix)
    scores=_logjoints(g, X)
    return exp.(scores .- _lognormalizers(scores))
end
_objective(g::MixtureModel, X, w) = dot(vec(_lognormalizers(_logjoints(g, X))), w)
function fit!(state::MixtureWorkspace, m::EM, X::AbstractMatrix; weights=nothing)
    X, w=_data(X, weights)
    _check(state.spec, m, size(X, 1))
    length(first(components(state.model)))==size(X, 1) ||
        throw(DimensionMismatch("workspace dimension mismatch"))
    report=FitReport()
    state.report=report
    empty!(state.inner_reports)
    joints, normalizers, previous=try
        scores=_logjoints(state.model, X)
        norms=_lognormalizers(scores)
        value=dot(vec(norms), w)
        isfinite(value) || throw(ArgumentError("nonfinite objective"))
        (scores, norms, value)
    catch err
        err isa Union{PosDefException,SingularException,ArgumentError,DomainError} ||
            rethrow()
        report.status=:failed
        report.message=sprint(showerror, err)
        return state.model
    end
    push!(report.history, previous)
    for iteration in 1:m.maxiter
        resp=exp.(joints .- normalizers)
        mass=vec(resp'*w)
        if any(t -> t<=m.min_mass, mass)
            report.status=:failed
            report.message="component mass at or below min_mass"
            break
        end
        newcomponents=Distributions.AbstractMvNormal[]
        inner_reports=FitReport[]
        try
            for j in 1:state.spec.k
                localstate=GaussianWorkspace(
                    state.spec.covariance, components(state.model)[j], FitReport()
                )
                component_weights=w .* view(resp, :, j)
                # Underflowed responsibilities have zero mass; exclude those
                # observations before moments (so 0 * overflow cannot become NaN).
                component_data, component_weights=if any(iszero, component_weights)
                    _data(X, component_weights)
                else
                    component_weights ./= sum(component_weights)
                    (X, component_weights)
                end
                push!(
                    newcomponents,
                    _fit_gaussian!(
                        localstate, m.covariance_method, component_data, component_weights
                    ),
                )
                push!(inner_reports, localstate.report)
            end
            any(r -> r.status==:failed, inner_reports) && throw(
                ArgumentError(
                    "covariance fitting failed: "*join(
                        [r.message for r in inner_reports if r.status==:failed], "; "
                    ),
                ),
            )
            typed_components=Vector{typeof(first(newcomponents))}(newcomponents)
            candidate=MixtureModel(typed_components, mass ./ sum(mass))
            candidate_joints=_logjoints(candidate, X)
            candidate_normalizers=_lognormalizers(candidate_joints)
            value=dot(vec(candidate_normalizers), w)
            isfinite(value) || throw(ArgumentError("nonfinite objective"))
            state.model=candidate
            joints=candidate_joints
            normalizers=candidate_normalizers
            state.inner_reports=inner_reports
            report.iterations=iteration
            push!(report.history, value)
            if m.tol>0 && abs(value-previous)<=m.tol*(1+abs(previous))
                report.status=:converged
                break
            end
            previous=value
        catch err
            err isa Union{PosDefException,SingularException,ArgumentError,DomainError} ||
                rethrow()
            report.status=:failed
            report.message=sprint(showerror, err)
            break
        end
    end
    report.status==:initialized && (report.status=:iteration_limit)
    report.objective=last(report.history)
    return state.model
end
