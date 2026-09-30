function _truncated_pca(X::AbstractMatrix, r::Int)
    return _weighted_pca(X, fill(1/size(X, 2), size(X, 2)), r)
end

# Weighted PCA uses normalized observation weights. The eigenvectors do not
# depend on an unbiased-vs-MLE scaling convention.
function _weighted_pca(X, w, r)
    μ=_mean(X, w)
    Z=(X .- μ) .* sqrt.(w')
    p, n=size(Z)
    r<=min(p, n-1) || throw(
        ArgumentError(
            "PCA rank must not exceed min(features, positive-weight samples - 1)"
        ),
    )
    gram=Symmetric(p<=n ? Z*Z' : Z'*Z)
    if min(p, n)<=100 || r>=min(p, n)-1
        eig=eigen(gram)
        indices=sortperm(eig.values; rev=true)[1:r]
        values=eig.values[indices]
        vectors=eig.vectors[:, indices]
    else
        values, vectors=eigs(
            gram; nev=r, which=:LR, v0=randn(MersenneTwister(0), size(gram, 1))
        )
        indices=sortperm(values; rev=true)
        values=values[indices]
        vectors=vectors[:, indices]
    end
    minimum(values)>eps(Float64)*maximum(values)*max(p, n) ||
        throw(ArgumentError("requested PCA rank exceeds numerical data rank"))
    F=p<=n ? vectors : (Z*vectors) ./ sqrt.(values')
    return Matrix(F), μ
end
function _pca_model(F, offset, D, latent)
    comps=[
        LatentMvNormal(offset+F*mean(c), F, D, Matrix(cholesky(Symmetric(cov(c))).L)) for
        c in components(latent)
    ]
    return MixtureModel(comps, copy(probs(latent)))
end
function _pca_setup(s, m, X, weights)
    X, w=_data(X, weights)
    _check(s, m, size(X, 1))
    F, offset=_weighted_pca(X, w, s.covariance.r)
    scores=F'*(X .- offset)
    residual=X .- offset .- F*scores
    D=max.(_variance(residual, w), m.residual_floor)
    all(>(0), D) ||
        throw(ArgumentError("zero PCA residual variance; use residual_floor > 0"))
    latent_spec=MixtureSpec(s.covariance.latent, s.k)
    return X, w, F, offset, D, scores, latent_spec
end
function initialize(
    s::MixtureSpec, m::PCAEM, X::AbstractMatrix; weights=nothing, rng=Random.default_rng()
)
    X, w, F, offset, D, scores, latent_spec=_pca_setup(s, m, X, weights)
    latent=initialize(latent_spec, m.latent_method, scores; weights=w, rng)
    model=_pca_model(F, offset, D, latent.model)
    return PCAWorkspace(
        s, F, offset, D, latent, model, FitReport(; kind=:projected_loglikelihood)
    )
end
function fit!(state::PCAWorkspace, m::PCAEM, X::AbstractMatrix; weights=nothing)
    X, w=_data(X, weights)
    _check(state.spec, m, size(X, 1))
    size(X, 1)==size(state.F, 1) || throw(DimensionMismatch("workspace dimension mismatch"))
    scores=state.F'*(X .- state.offset)
    fit!(state.latent, m.latent_method, scores; weights=w)
    state.model=_pca_model(state.F, state.offset, state.D, state.latent.model)
    state.report=deepcopy(state.latent.report)
    state.report.objective_kind=:projected_loglikelihood
    state.report.observed_objective=_objective(state.model, X, w)
    return state.model
end
function fit(
    s::MixtureSpec,
    m::PCAEM,
    X::AbstractMatrix;
    weights=nothing,
    rng=Random.default_rng(),
    report=false,
)
    X, w, F, offset, D, scores, latent_spec=_pca_setup(s, m, X, weights)
    # The nested method owns its restart policy; fit! never runs restarts.
    result=fit(latent_spec, m.latent_method, scores; weights=w, rng, report=true)
    model=_pca_model(F, offset, D, result.model)
    diagnostics=deepcopy(result.report)
    diagnostics.objective_kind=:projected_loglikelihood
    for run in diagnostics.runs
        run.objective_kind=:projected_loglikelihood
    end
    diagnostics.observed_objective=_objective(model, X, w)
    return report ? (model=model, report=diagnostics) : model
end
