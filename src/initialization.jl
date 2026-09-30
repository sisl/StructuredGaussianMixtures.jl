"""Initialize fitting state using the method's strategy and explicit RNG.
Data columns are observations. Nonnegative weights are normalized internally.
"""
function initialize(
    s::GaussianStructure,
    m::CovarianceMethod,
    X::AbstractMatrix;
    weights=nothing,
    rng=Random.default_rng(),
)
    X, w=_data(X, weights)
    _check(s, m, size(X, 1))
    μ=_mean(X, w)
    R=X .- μ
    if s isa LowRankDiagonal
        D=max.(_variance(R, w), max(m.variance_floor, eps(Float64)))
        F=0.1 .* sqrt.(D) .* randn(rng, size(X, 1), s.r) ./ sqrt(max(s.r, 1))
        g=LRDMvNormal(μ, F, D)
    else
        g, _=_covariance(s, m, nothing, R, w)
        g=_remean(g, μ)
    end
    return GaussianWorkspace(s, g, FitReport())
end
function _centers(init, k, X, w, rng)
    n=size(X, 2)
    k<=n || throw(ArgumentError("components exceed positive-weight observation count"))
    ids=Int[]
    available=copy(w)
    for j in 1:k
        i=rand(rng, Categorical(available ./ sum(available)))
        push!(ids, i)
        available[i]=0
    end
    centers=copy(X[:, ids])
    if init isa KMeansInit
        for iteration in 1:init.maxiter
            distances =
                vec(sum(abs2, X; dims=1)) .+ sum(abs2, centers; dims=1) .-
                2 .* (X' * centers)
            labels = [argmin(view(distances, i, :)) for i in 1:n]
            updated=copy(centers)
            for j in 1:k
                indices=findall(==(j), labels)
                isempty(indices) && continue
                updated[:, j]=X[:, indices]*w[indices]/sum(w[indices])
            end
            updated==centers && break
            centers=updated
        end
    end
    return centers
end
function initialize(
    s::MixtureSpec, m::EM, X::AbstractMatrix; weights=nothing, rng=Random.default_rng()
)
    X, w=_data(X, weights)
    _check(s, m, size(X, 1))
    centers=_centers(m.init, s.k, X, w, rng)
    base = initialize(s.covariance, m.covariance_method, X; weights=w, rng).model
    comps = [_remean(deepcopy(base), centers[:, j]) for j in 1:s.k]
    return MixtureWorkspace(s, MixtureModel(comps), FitReport(), FitReport[])
end
