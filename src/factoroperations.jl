# Prepare once per vector or batch, never persist a cache alongside mutable arrays.
function _prepare_factor(F, D)
    scaled = F ./ sqrt.(D)
    C = cholesky(Symmetric(I + scaled' * scaled))
    return scaled, C, sum(log, D) + logdet(C)
end

function _factor_logpdf(μ, D, prepared, x::AbstractVector)
    length(x) == length(μ) || throw(DimensionMismatch("Observation dimension mismatch"))
    scaled, C, ld = prepared
    v = (x - μ) ./ sqrt.(D)
    z = C \ (scaled' * v)
    # Equivalent to Woodbury, but avoids subtracting two large quadratic terms.
    q = sum(abs2, v - scaled * z) + sum(abs2, z)
    return -0.5 * (length(μ) * log(2π) + ld + q)
end

function _factor_logpdf(μ, D, prepared, X::AbstractMatrix)
    size(X, 1) == length(μ) || throw(DimensionMismatch("Observation dimension mismatch"))
    scaled, C, ld = prepared
    V = (X .- μ) ./ sqrt.(D)
    Z = C \ (scaled' * V)
    q = vec(sum(abs2, V - scaled * Z; dims=1) + sum(abs2, Z; dims=1))
    return -0.5 .* (length(μ) * log(2π) + ld .+ q)
end
