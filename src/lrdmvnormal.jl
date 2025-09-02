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
- The covariance matrix is never explicitly formed
- All operations use the low-rank plus diagonal structure for efficiency
"""
struct LRDMvNormal <: Distributions.AbstractMvNormal
    μ::Vector{Float64}  # mean vector
    F::Matrix{Float64}  # low-rank factor matrix
    D::Vector{Float64}  # diagonal vector
    rank::Int          # rank of the low-rank component
    
    function LRDMvNormal(μ::Vector{Float64}, F::Matrix{Float64}, D::Vector{Float64})
        length(μ) == size(F, 1) == length(D) || 
            throw(DimensionMismatch("Dimensions of μ, F, and D must match"))
        length(D) > size(F, 2) || throw(ArgumentError("Rank of F must be less than the number of features"))
        all(d -> d > 0, D) || throw(ArgumentError("All diagonal elements must be positive"))
        new(μ, F, D, size(F, 2))
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
    # Center the data
    x_centered = x - d.μ
    
    # Compute the precision matrix efficiently using the matrix inversion lemma
    # (F*F' + D)^(-1) = D^(-1) - D^(-1)*F*(I + F'*D^(-1)*F)^(-1)*F'*D^(-1)
    D_inv = 1 ./ d.D
    F_scaled = d.F .* sqrt.(D_inv)
    I_plus_FF = I + F_scaled' * F_scaled
    
    # Compute the determinant efficiently
    # det(F*F' + D) = det(D) * det(I + F'*D^(-1)*F)
    logdet_cov = sum(log.(d.D)) + logdet(I_plus_FF)
    
    # Compute the quadratic form efficiently with block elimination
    # quad_form = dot(x_centered, precision * x_centered)
    y = (I_plus_FF \ F_scaled') * (sqrt.(D_inv) .* x_centered)
    eta = D_inv .* (x_centered - d.F * y)
    quad_form = dot(x_centered, eta)

    # Return the log PDF
    return -0.5 * (length(d.μ) * log(2π) + logdet_cov + quad_form)
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

Return the mean vector of the distribution.
"""
Distributions.mean(d::LRDMvNormal) = d.μ

"""
    cov(d::LRDMvNormal)

Return the full covariance matrix FF' + D.
"""
Distributions.cov(d::LRDMvNormal) = _covariance(d)

"""
    rank(d::LRDMvNormal)

Return the rank of the low-rank component.
"""
function rank(d::LRDMvNormal)
    return d.rank
end

"""
    low_rank_factor(d::LRDMvNormal)

Return the low-rank factor matrix F.
"""
function low_rank_factor(d::LRDMvNormal)
    return d.F
end

"""
    diagonal(d::LRDMvNormal)

Return the diagonal vector D.
"""
function diagonal(d::LRDMvNormal)
    return d.D
end