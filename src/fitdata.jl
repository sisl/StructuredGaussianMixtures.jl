function _data(X::AbstractMatrix, weights)
    p, n=size(X)
    p > 0 && n > 0 || throw(ArgumentError("data must be nonempty (features × samples)"))
    all(isfinite, X) || throw(ArgumentError("data must be finite"))
    w = weights === nothing ? ones(n) : Float64.(weights)
    w isa AbstractVector && length(w)==n ||
        throw(DimensionMismatch("one weight per observation required"))
    all(t -> isfinite(t) && t >= 0, w) && maximum(w)>0 ||
        throw(ArgumentError("weights must be finite, nonnegative, and have positive mass"))
    # Scaling first avoids overflow for large finite weights. Remove zero mass
    # observations before initialization and PCA as well as moment calculations.
    w ./= maximum(w)
    if any(iszero, w)
        keep=findall(>(0), w)
        X=X[:, keep]
        w=w[keep]
    end
    w ./= sum(w)
    # Fitting only reads data; reuse an existing dense Float64 matrix.
    return X isa Matrix{Float64} ? X : Matrix{Float64}(X), w
end
_mean(X, w) = vec(X*w)
function _variance(R, w)
    v=zeros(size(R, 1))
    for j in axes(R, 2)
        @inbounds for i in axes(R, 1)
            v[i] += abs2(R[i, j])*w[j]
        end
    end
    return v
end
function _scores(g, X)
    scores=Vector{Float64}(undef, size(X, 2))
    logpdf!(scores, g, X)
    return scores
end
_objective(g, X, w) = dot(_scores(g, X), w)
function _check(s::GaussianStructure, m::CovarianceMethod, p)
    if s isa Union{FullCovariance,DiagonalCovariance}
        m isa Exact || throw(ArgumentError("full and diagonal covariance require Exact"))
    elseif s isa LowRankDiagonal
        s.r < p ||
            throw(ArgumentError("LRD rank must be smaller than observation dimension"))
        m isa CovarianceEM || throw(ArgumentError("LRD covariance requires CovarianceEM"))
    else
        throw(
            ArgumentError(
                "LatentCovariance fitting currently requires PCAEM with shared F,D"
            ),
        )
    end
end
function _check(s::MixtureSpec, m::EM, p)
    isempty(s.tied.parameters) ||
        throw(ArgumentError("joint tied updates are not implemented for EM"))
    return _check(s.covariance, m.covariance_method, p)
end
function _check(s::MixtureSpec, m::PCAEM, p)
    s.covariance isa LatentCovariance ||
        throw(ArgumentError("PCAEM requires LatentCovariance"))
    Set(s.tied.parameters)==Set((:F, :D)) ||
        throw(ArgumentError("PCAEM requires exactly Tied(:F,:D)"))
    return s.covariance.r <= p || throw(ArgumentError("PCA rank exceeds feature dimension"))
end
_remean(g::MvNormal, μ) = MvNormal(μ, g.Σ)
_remean(g::LRDMvNormal, μ) = LRDMvNormal(μ, g.F, g.D)
