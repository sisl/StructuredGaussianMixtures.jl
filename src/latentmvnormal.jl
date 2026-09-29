"""
    LatentMvNormal(μ, F, D, A_factor)

Gaussian with covariance `F * A_factor * A_factor' * F' + Diagonal(D)`. `F` is a loading
basis and `A_factor` is a square latent covariance factor (not necessarily triangular
or symmetric). All inputs are copied into owned `Float64` arrays. Accessors
return copies. The latent dimension may exceed the observation dimension;
positive finite `D` ensures a nonsingular observation covariance even if `A_factor`
is singular. Zero latent dimension is supported.

Unlike [`LRDMvNormal`](@ref), this representation retains the basis separately
from latent uncertainty. Equal bases in different components do not impose a
fitting constraint. Fields are implementation details; do not mutate them.
"""
struct LatentMvNormal <: Distributions.AbstractMvNormal
    μ::Vector{Float64}
    F::Matrix{Float64}
    D::Vector{Float64}
    A_factor::Matrix{Float64}

    function LatentMvNormal(
        μ::AbstractVector, F::AbstractMatrix, D::AbstractVector, A_factor::AbstractMatrix
    )
        p, r = size(F)
        length(μ) == p == length(D) ||
            throw(DimensionMismatch("Dimensions of μ, F, and D must match"))
        size(A_factor) == (r, r) || throw(DimensionMismatch("A_factor must be r × r"))
        p > 0 || throw(ArgumentError("Observation dimension must be positive"))
        μc = Vector{Float64}(μ)
        F_copy = Matrix{Float64}(F)
        Dc = Vector{Float64}(D)
        A_factor_copy = Matrix{Float64}(A_factor)
        all(isfinite, μc) && all(isfinite, F_copy) && all(isfinite, A_factor_copy) ||
            throw(ArgumentError("Mean and factors must be finite"))
        all(x -> isfinite(x) && x > 0, Dc) ||
            throw(ArgumentError("Diagonal variances must be positive and finite"))
        return new(μc, F_copy, Dc, A_factor_copy)
    end
end

Distributions.length(d::LatentMvNormal) = length(d.μ)
Distributions.size(d::LatentMvNormal) = (length(d),)
Distributions.mean(d::LatentMvNormal) = copy(d.μ)
function Distributions.cov(d::LatentMvNormal)
    return (
        effective_factor=low_rank_factor(d);
        effective_factor * effective_factor' + Diagonal(d.D)
    )
end
rank(d::LatentMvNormal) = size(d.F, 2)
diagonal(d::LatentMvNormal) = copy(d.D)
low_rank_factor(d::LatentMvNormal) = d.F * d.A_factor

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
    mul!(x, d.F, d.A_factor * z)
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

"""
    marginal(d::LatentMvNormal, indices)

Select coordinates, preserving the latent factor. The nonempty indices must be
unique and in bounds. The returned distribution owns independent parameters.
"""
function marginal(d::LatentMvNormal, indices::Union{Vector{Int},AbstractRange})
    idx = _structured_indices(d, indices)
    isempty(idx) && throw(ArgumentError("Marginal indices must be nonempty"))
    return LatentMvNormal(d.μ[idx], d.F[idx, :], d.D[idx], d.A_factor)
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
    obs = _structured_indices(d, input_indices)
    target = _structured_indices(d, output_indices)
    length(x) == length(obs) ||
        throw(DimensionMismatch("Observed values and indices must match"))
    isempty(target) && throw(ArgumentError("Output indices must be nonempty"))
    isempty(intersect(obs, target)) ||
        throw(ArgumentError("Input and output indices must be disjoint"))
    isempty(obs) && return marginal(d, target)
    effective_factor_obs = d.F[obs, :] * d.A_factor
    scaled, latent_chol, _ = _prepare_factor(effective_factor_obs, d.D[obs])
    z = latent_chol \ (scaled' * ((x - d.μ[obs]) ./ sqrt.(d.D[obs])))
    μ = d.μ[target] + d.F[target, :] * (d.A_factor * z)
    A_factor = d.A_factor / latent_chol.U
    return LatentMvNormal(μ, d.F[target, :], d.D[target], A_factor)
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

"""
    loading(d::Union{LRDMvNormal,LatentMvNormal})

Return a copy of the loading matrix `F`. For a latent Gaussian this excludes
`A_factor`; use `low_rank_factor(d)` for the effective factor `F * A_factor`.
"""
loading(d::Union{LRDMvNormal,LatentMvNormal}) = copy(d.F)

"""
    latent_covariance_factor(d::Union{LRDMvNormal,LatentMvNormal})

Return an independent factor of the latent covariance `A`. For `LRDMvNormal`
this is a sized diagonal identity. For `LatentMvNormal` it is a copy of
`A_factor`, with `A = A_factor * A_factor'`; it need not be a Cholesky factor.
"""
latent_covariance_factor(d::LRDMvNormal) = Diagonal(ones(rank(d)))
latent_covariance_factor(d::LatentMvNormal) = copy(d.A_factor)

"""
    latent_covariance(d::Union{LRDMvNormal,LatentMvNormal})

Return the latent covariance `A`: a sized diagonal identity for `LRDMvNormal`,
or `A_factor * A_factor'` for `LatentMvNormal`. The result is independent of the
stored parameters. This is distinct from `cov(d)`, the observation covariance.
"""
latent_covariance(d::LRDMvNormal) = Diagonal(ones(rank(d)))
latent_covariance(d::LatentMvNormal) = d.A_factor * d.A_factor'

# Marginal variances without materializing a dense observation covariance.
Distributions.var(d::LRDMvNormal) = vec(sum(abs2, d.F; dims=2)) + d.D
Distributions.var(d::LatentMvNormal) = vec(sum(abs2, low_rank_factor(d); dims=2)) + d.D
