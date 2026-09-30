"""
    LRDMvNormal

Low-rank plus diagonal multivariate normal distribution.
The covariance matrix is represented as Σ = FF' + D, where F is a low-rank factor matrix
and D is a diagonal matrix.

# Fields
- `μ`: Mean vector
- `F`: Low-rank factor matrix
- `D`: Diagonal vector
- `rank`: Rank of the low-rank component

# Notes
- Inputs are copied into owned `Float64` arrays; accessors return independent values.
- Fields are implementation details; internal fitting routines update owned parameters.
- The covariance matrix is only explicitly formed when requested with `cov`
- All operations use the low-rank plus diagonal structure for efficiency
"""
struct LRDMvNormal <: Distributions.AbstractMvNormal
    μ::Vector{Float64}  # mean vector
    F::Matrix{Float64}  # low-rank factor matrix
    D::Vector{Float64}  # diagonal vector
    rank::Int          # rank of the low-rank component

    function LRDMvNormal(μ::AbstractVector, F::AbstractMatrix, D::AbstractVector)
        length(μ) == size(F, 1) == length(D) ||
            throw(DimensionMismatch("Dimensions of μ, F, and D must match"))
        length(D) > size(F, 2) || throw(
            ArgumentError("Latent dimension must be less than the number of features")
        )
        μ_copy = Vector{Float64}(μ)
        F_copy = Matrix{Float64}(F)
        D_copy = Vector{Float64}(D)
        all(isfinite, μ_copy) && all(isfinite, F_copy) ||
            throw(ArgumentError("Mean and loading must be finite"))
        all(x -> isfinite(x) && x > 0, D_copy) ||
            throw(ArgumentError("Diagonal variances must be positive and finite"))
        return new(μ_copy, F_copy, D_copy, size(F, 2))
    end
end

"""
    length(d::LRDMvNormal)

Return the dimension of the distribution.
"""
Distributions.length(d::LRDMvNormal) = length(d.μ)

"""
    size(d::LRDMvNormal)

Return the size of the distribution as a tuple (dimension,).
"""
Distributions.size(d::LRDMvNormal) = (length(d.μ),)

# Internal function - not documented
function _covariance(d::LRDMvNormal)
    return d.F * d.F' + Diagonal(d.D)
end

"""
    logpdf(d::LRDMvNormal, x::AbstractVector)

Compute the log probability density function at x.
Uses the matrix inversion lemma for efficient computation.

# Arguments
- `d`: The LRDMvNormal distribution
- `x`: The point at which to evaluate the log PDF

# Returns
- The log probability density at x

# Notes
- Uses the matrix inversion lemma: (F*F' + D)^(-1) = D^(-1) - D^(-1)*F*(I + F'*D^(-1)*F)^(-1)*F'*D^(-1)
- Computes the determinant efficiently: det(F*F' + D) = det(D) * det(I + F'*D^(-1)*F)
"""
function Distributions.logpdf(d::LRDMvNormal, x::AbstractVector)
    return _factor_logpdf(d.μ, d.D, _prepare_factor(d.F, d.D), x)
end

"""
    _rand!(rng::AbstractRNG, d::LRDMvNormal, x::VecOrMat)

Generate random samples in-place from the distribution.

# Arguments
- `rng`: Random number generator
- `d`: The LRDMvNormal distribution
- `x`: Vector or matrix to fill with random samples

# Returns
- The filled vector/matrix x

# Notes
- Uses the decomposition: X = μ + F*Z₁ + sqrt(D)*Z₂ where Z₁, Z₂ are standard normal
- For matrices, each column is a sample
"""
function Distributions._rand!(rng::AbstractRNG, d::LRDMvNormal, x::VecOrMat)
    # Generate random vectors from standard normal
    z1 = similar(x)
    z2 = similar(x, size(d.F, 2), size(x, 2))
    randn!(rng, z1)
    randn!(rng, z2)

    # Transform to get samples from our distribution
    if x isa AbstractVector
        # For vectors, z2 is also a vector
        z2_vec = similar(x, size(d.F, 2))
        copyto!(z2_vec, z2)
        mul!(x, d.F, z2_vec)  # x = F * z2
    else
        # For matrices, z2 is a matrix
        mul!(x, d.F, z2)  # x = F * z2
    end
    x .+= d.μ         # x += μ
    x .+= sqrt.(d.D) .* z1  # x += sqrt(D) * z1
    return x
end

"""
    _rand!(rng::AbstractRNG, d::LRDMvNormal, x::AbstractVector)

Generate a random sample in-place from the distribution.

# Arguments
- `rng`: Random number generator
- `d`: The LRDMvNormal distribution
- `x`: Vector to fill with random sample

# Returns
- The filled vector x

# Notes
- Uses the decomposition: X = μ + F*Z₁ + sqrt(D)*Z₂ where Z₁, Z₂ are standard normal
- Handles AbstractVector types that don't support randn!
"""
function Distributions._rand!(rng::AbstractRNG, d::LRDMvNormal, x::AbstractVector)
    # Generate random vectors from standard normal
    z1 = similar(x)
    z2 = similar(x, size(d.F, 2))

    # Fill z1 and z2 with random numbers
    for i in eachindex(z1)
        @inbounds z1[i] = randn(rng, eltype(z1))
    end
    for i in eachindex(z2)
        @inbounds z2[i] = randn(rng, eltype(z2))
    end

    # Transform to get sample from our distribution
    mul!(x, d.F, z2)  # x = F * z2
    x .+= d.μ         # x += μ
    x .+= sqrt.(d.D) .* z1  # x += sqrt(D) * z1
    return x
end

"""
    mean(d::LRDMvNormal)

Return a copy of the mean vector of the distribution.
"""
Distributions.mean(d::LRDMvNormal) = copy(d.μ)

"""
    cov(d::LRDMvNormal)

Return the full covariance matrix FF' + D.
"""
Distributions.cov(d::LRDMvNormal) = _covariance(d)

"""
    rank(d::LRDMvNormal)

Return the stored latent dimension, not the numerical matrix rank.
"""
function rank(d::LRDMvNormal)
    return d.rank
end

"""
    low_rank_factor(d::LRDMvNormal)

Return a copy of the effective low-rank factor `F`.
"""
function low_rank_factor(d::LRDMvNormal)
    return copy(d.F)
end

"""
    diagonal(d::LRDMvNormal)

Return a copy of the residual variance vector `D`.
"""
function diagonal(d::LRDMvNormal)
    return copy(d.D)
end

# Batch preparation shares the same small factorization across observations.
function Distributions.logpdf(d::LRDMvNormal, X::AbstractMatrix)
    return _factor_logpdf(d.μ, d.D, _prepare_factor(d.F, d.D), X)
end
