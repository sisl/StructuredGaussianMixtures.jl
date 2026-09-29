"""
    LatentMvNormal(μ, L, D, B)

Gaussian with covariance `L * B * B' * L' + Diagonal(D)`. `L` is a loading
basis and `B` is a square latent covariance factor (not necessarily triangular
or symmetric). All inputs are copied into owned `Float64` arrays. Accessors
return copies. The latent dimension may exceed the observation dimension;
positive finite `D` ensures a nonsingular observation covariance even if `B`
is singular. Zero latent dimension is supported.

Unlike [`LRDMvNormal`](@ref), this representation retains the basis separately
from latent uncertainty. Equal bases in different components do not impose a
fitting constraint. Fields are implementation details; do not mutate them.
"""
struct LatentMvNormal <: Distributions.AbstractMvNormal
    μ::Vector{Float64}
    L::Matrix{Float64}
    D::Vector{Float64}
    B::Matrix{Float64}

    function LatentMvNormal(
        μ::AbstractVector, L::AbstractMatrix, D::AbstractVector, B::AbstractMatrix
    )
        p, r = size(L)
        length(μ) == p == length(D) ||
            throw(DimensionMismatch("Dimensions of μ, L, and D must match"))
        size(B) == (r, r) || throw(DimensionMismatch("B must be r × r"))
        p > 0 || throw(ArgumentError("Observation dimension must be positive"))
        μc, Lc, Dc, Bc = Vector{Float64}(μ),
        Matrix{Float64}(L), Vector{Float64}(D),
        Matrix{Float64}(B)
        all(isfinite, μc) && all(isfinite, Lc) && all(isfinite, Bc) ||
            throw(ArgumentError("Mean and factors must be finite"))
        all(x -> isfinite(x) && x > 0, Dc) ||
            throw(ArgumentError("Diagonal variances must be positive and finite"))
        return new(μc, Lc, Dc, Bc)
    end
end

Distributions.length(d::LatentMvNormal) = length(d.μ)
Distributions.size(d::LatentMvNormal) = (length(d),)
Distributions.mean(d::LatentMvNormal) = copy(d.μ)
Distributions.cov(d::LatentMvNormal) = (F=low_rank_factor(d); F * F' + Diagonal(d.D))
rank(d::LatentMvNormal) = size(d.L, 2)
diagonal(d::LatentMvNormal) = copy(d.D)
low_rank_factor(d::LatentMvNormal) = d.L * d.B

"""
    loading(d::LatentMvNormal)

Return a copy of the loading basis `L`.
"""
loading(d::LatentMvNormal) = copy(d.L)

"""
    latent_factor(d::LatentMvNormal)

Return a copy of `B`, where the latent covariance is `B * B'`.
"""
latent_factor(d::LatentMvNormal) = copy(d.B)

function Distributions.logpdf(d::LatentMvNormal, x::AbstractVector)
    return _factor_logpdf(d.μ, d.D, _prepare_factor(low_rank_factor(d), d.D), x)
end
function Distributions.logpdf(d::LatentMvNormal, X::AbstractMatrix)
    return _factor_logpdf(d.μ, d.D, _prepare_factor(low_rank_factor(d), d.D), X)
end

function _latent_rand!(rng, d, x)
    size(x, 1) == length(d) || throw(DimensionMismatch("Sample dimension mismatch"))
    z = x isa AbstractVector ? randn(rng, rank(d)) : randn(rng, rank(d), size(x, 2))
    noise = randn(rng, size(x))
    mul!(x, d.L, d.B * z)
    x .+= d.μ
    x .+= sqrt.(d.D) .* noise
    return x
end
function Distributions._rand!(rng::AbstractRNG, d::LatentMvNormal, x::AbstractVector)
    return _latent_rand!(rng, d, x)
end
function Distributions._rand!(rng::AbstractRNG, d::LatentMvNormal, x::AbstractMatrix)
    return _latent_rand!(rng, d, x)
end

# Validate before indexing; exact conditioning on an output coordinate is singular.
function _latent_indices(d, indices)
    idx = collect(indices)
    all(i -> 1 <= i <= length(d), idx) || throw(ArgumentError("Indices out of bounds"))
    length(unique(idx)) == length(idx) || throw(ArgumentError("Indices must be unique"))
    return idx
end

"""
    marginal(d::LatentMvNormal, indices)

Select coordinates, preserving the latent factor. The nonempty indices must be
unique and in bounds. The returned distribution owns independent parameters.
"""
function marginal(d::LatentMvNormal, indices::Union{Vector{Int},AbstractRange})
    idx = _latent_indices(d, indices)
    isempty(idx) && throw(ArgumentError("Marginal indices must be nonempty"))
    return LatentMvNormal(d.μ[idx], d.L[idx, :], d.D[idx], d.B)
end

"""
    predict(d::LatentMvNormal, x, input_indices, output_indices)

Condition on `x` at `input_indices`, returning a `LatentMvNormal` over the
nonempty `output_indices`. Indices must be unique, in bounds and disjoint;
other coordinates are marginalized. Empty input indices are allowed.
The loading rows are retained and the latent factor is updated by a triangular
solve. No dense observation covariance or explicit matrix inverse is formed.
"""
function predict(
    d::LatentMvNormal,
    x::AbstractVector,
    input_indices::Union{Vector{Int},AbstractRange},
    output_indices::Union{Vector{Int},AbstractRange},
)
    obs = _latent_indices(d, input_indices)
    target = _latent_indices(d, output_indices)
    length(x) == length(obs) ||
        throw(DimensionMismatch("Observed values and indices must match"))
    isempty(target) && throw(ArgumentError("Output indices must be nonempty"))
    isempty(intersect(obs, target)) ||
        throw(ArgumentError("Input and output indices must be disjoint"))
    isempty(obs) && return marginal(d, target)
    Fobs = d.L[obs, :] * d.B
    scaled, C, _ = _prepare_factor(Fobs, d.D[obs])
    z = C \ (scaled' * ((x - d.μ[obs]) ./ sqrt.(d.D[obs])))
    μ = d.μ[target] + d.L[target, :] * (d.B * z)
    B = d.B / C.U
    return LatentMvNormal(μ, d.L[target, :], d.D[target], B)
end

# MixtureModel batch scoring calls logpdf! rather than the allocating logpdf.
function Distributions._logpdf!(
    out::AbstractArray{<:Real},
    d::Union{LRDMvNormal,LatentMvNormal},
    X::AbstractMatrix{<:Real},
)
    length(out) == size(X, 2) || throw(DimensionMismatch("Output length mismatch"))
    out .= logpdf(d, X)
    return out
end
