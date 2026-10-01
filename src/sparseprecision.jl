"""
    SparsePrecisionMvNormal(μ, Q)

Gaussian represented by sparse positive-definite precision `Q = inv(Σ)`.
Inputs and accessors are copied. A cached sparse Cholesky factor supports scoring,
selected covariance solves and sampling. `cov` explicitly materializes dense Σ.
Internal fields must not be mutated after construction.
"""
struct SparsePrecisionMvNormal <: Distributions.AbstractMvNormal
    μ::Vector{Float64}
    Q::SparseMatrixCSC{Float64,Int}
    factor::SparseArrays.CHOLMOD.Factor{Float64,Int}
    function SparsePrecisionMvNormal(μ::AbstractVector, Q::AbstractMatrix)
        p=length(μ)
        p>0 || throw(ArgumentError("dimension must be positive"))
        size(Q)==(p, p) || throw(DimensionMismatch("precision dimension mismatch"))
        μc=Vector{Float64}(μ)
        Qc=copy(SparseMatrixCSC{Float64,Int}(Q))
        all(isfinite, μc) && all(isfinite, nonzeros(Qc)) ||
            throw(ArgumentError("parameters must be finite"))
        issymmetric(Qc) || throw(ArgumentError("precision must be symmetric"))
        dropzeros!(Qc)
        return new(μc, Qc, cholesky(Symmetric(Qc)))
    end
end
Distributions.length(g::SparsePrecisionMvNormal) = length(g.μ)
Distributions.size(g::SparsePrecisionMvNormal) = (length(g),)
Distributions.mean(g::SparsePrecisionMvNormal) = copy(g.μ)
"""Return an owned copy of a sparse Gaussian's precision matrix."""
Distributions.invcov(g::SparsePrecisionMvNormal) = copy(g.Q)
function Distributions.cov(g::SparsePrecisionMvNormal)
    return Matrix(Symmetric(g.factor \ Matrix{Float64}(I, length(g), length(g))))
end
Distributions.var(g::SparsePrecisionMvNormal) = diag(cov(g))
Distributions.logdetcov(g::SparsePrecisionMvNormal) = -logdet(g.factor)
function Distributions.sqmahal(g::SparsePrecisionMvNormal, x::AbstractVector)
    length(x)==length(g) || throw(DimensionMismatch("sample dimension mismatch"))
    y=x-g.μ
    return dot(y, g.Q*y)
end
function Distributions.logpdf(g::SparsePrecisionMvNormal, x::AbstractVector)
    return -0.5*(length(g)*log(2π)+Distributions.logdetcov(g)+Distributions.sqmahal(g, x))
end
function Distributions._logpdf!(
    out::AbstractArray{<:Real}, g::SparsePrecisionMvNormal, X::AbstractMatrix{<:Real}
)
    size(X, 1)==length(g) && length(out)==size(X, 2) ||
        throw(DimensionMismatch("score dimensions mismatch"))
    y=Vector{Float64}(undef, length(g))
    q=similar(y)
    constant=length(g)*log(2π)+Distributions.logdetcov(g)
    for j in axes(X, 2)
        y .= view(X, :, j) .- g.μ
        mul!(q, g.Q, y)
        out[j]=-0.5*(constant+dot(y, q))
    end
    return out
end
function Distributions.logpdf(g::SparsePrecisionMvNormal, X::AbstractMatrix)
    return Distributions._logpdf!(Vector{Float64}(undef, size(X, 2)), g, X)
end
function Distributions._rand!(
    rng::AbstractRNG, g::SparsePrecisionMvNormal, x::AbstractVecOrMat
)
    size(x, 1)==length(g) || throw(DimensionMismatch("sample dimension mismatch"))
    x .= g.factor.UP \ randn(rng, size(x))
    x .+= g.μ
    return x
end
function _selected_sparse_covariance(g, idx)
    rhs=zeros(length(g), length(idx))
    for (j, i) in enumerate(idx)
        rhs[i, j]=1
    end
    return Symmetric((g.factor \ rhs)[idx, :])
end
"""Marginalize using selected precision solves; the resulting covariance may be dense."""
function marginal(g::SparsePrecisionMvNormal, indices::Union{Vector{Int},AbstractRange})
    idx=_structured_indices(g, indices)
    isempty(idx) && throw(ArgumentError("marginal indices must be nonempty"))
    return MvNormal(g.μ[idx], _selected_sparse_covariance(g, idx))
end
"""
Condition using the principal precision block for unobserved coordinates. If all
unobserved coordinates are requested, return a sparse precision Gaussian. If some
are marginalized, use selected solves and return a dense `MvNormal`.
"""
function predict(
    g::SparsePrecisionMvNormal,
    x::AbstractVector,
    input_indices::Union{Vector{Int},AbstractRange},
    output_indices::Union{Vector{Int},AbstractRange},
)
    obs=_structured_indices(g, input_indices)
    target=_structured_indices(g, output_indices)
    length(x)==length(obs) ||
        throw(DimensionMismatch("observed values and indices must match"))
    all(isfinite, x) || throw(ArgumentError("observations must be finite"))
    isempty(target) && throw(ArgumentError("output indices must be nonempty"))
    isempty(intersect(obs, target)) ||
        throw(ArgumentError("input and output indices must be disjoint"))
    isempty(obs) && return marginal(g, target)
    remaining=vcat(target, setdiff(1:length(g), vcat(obs, target)))
    Q=g.Q[remaining, remaining]
    factor=cholesky(Symmetric(Q))
    μ=g.μ[remaining]-factor \ (g.Q[remaining, obs]*(x-g.μ[obs]))
    conditional=SparsePrecisionMvNormal(μ, Q)
    return if length(remaining)==length(target)
        conditional
    else
        marginal(conditional, 1:length(target))
    end
end
function predict(
    g::SparsePrecisionMvNormal,
    x::AbstractVector;
    input_indices=1:length(x),
    output_indices=(length(x) + 1):length(g),
)
    return predict(g, x, collect(input_indices), collect(output_indices))
end
