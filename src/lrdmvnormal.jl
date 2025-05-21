struct LRDMvNormal <: ContinuousMultivariateDistribution
    μ::Vector{Float64}  # mean vector
    F::Matrix{Float64}  # low-rank factor matrix
    D::Vector{Float64}  # diagonal vector
    rank::Int          # rank of the low-rank component
    
    function LRDMvNormal(μ::Vector{Float64}, F::Matrix{Float64}, D::Vector{Float64})
        length(μ) == size(F, 1) == length(D) || 
            throw(DimensionMismatch("Dimensions of μ, F, and D must match"))
        new(μ, F, D, size(F, 2))
    end
end

# Required Distributions.jl interface methods
Distributions.length(d::LRDMvNormal) = length(d.μ)
Distributions.size(d::LRDMvNormal) = (length(d.μ),)

# Compute the full covariance matrix (used internally)
function _covariance(d::LRDMvNormal)
    return d.F * d.F' + Diagonal(d.D)
end

# Log probability density function
function Distributions.logpdf(d::LRDMvNormal, x::AbstractVector)
    # Center the data
    x_centered = x - d.μ
    
    # Compute the precision matrix efficiently using the matrix inversion lemma
    # (F*F' + D)^(-1) = D^(-1) - D^(-1)*F*(I + F'*D^(-1)*F)^(-1)*F'*D^(-1)
    D_inv = 1 ./ d.D
    F_scaled = d.F .* sqrt.(D_inv)
    I_plus_FF = I + F_scaled' * F_scaled
    precision = Diagonal(D_inv) - 
                (F_scaled * (I_plus_FF \ F_scaled')) .* sqrt.(D_inv * D_inv')
    
    # Compute the determinant efficiently
    # det(F*F' + D) = det(D) * det(I + F'*D^(-1)*F)
    logdet_cov = sum(log.(d.D)) + logdet(I_plus_FF)
    
    # Compute the quadratic form
    quad_form = dot(x_centered, precision * x_centered)
    
    # Return the log PDF
    return -0.5 * (length(d.μ) * log(2π) + logdet_cov + quad_form)
end

# Random number generation
function Distributions.rand(rng::AbstractRNG, d::LRDMvNormal)
    # Generate random vector from standard normal
    z1 = randn(rng, length(d.μ))
    z2 = randn(rng, d.rank)
    
    # Transform to get samples from our distribution
    return d.μ + d.F * z2 + sqrt.(d.D) .* z1
end

# Mean and covariance
Distributions.mean(d::LRDMvNormal) = d.μ
Distributions.cov(d::LRDMvNormal) = _covariance(d)

# Additional utility functions
function rank(d::LRDMvNormal)
    return d.rank
end

function low_rank_factor(d::LRDMvNormal)
    return d.F
end

function diagonal(d::LRDMvNormal)
    return d.D
end