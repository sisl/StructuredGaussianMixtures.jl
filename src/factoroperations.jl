# Prepare once per vector or batch, never persist a cache alongside mutable arrays.
function _prepare_factor(effective_factor, D)
    scaled = effective_factor ./ sqrt.(D)
    latent_chol = cholesky(Symmetric(I + scaled' * scaled))
    return scaled, latent_chol, sum(log, D) + logdet(latent_chol)
end

function _factor_logpdf(μ, D, prepared, x::AbstractVector)
    length(x) == length(μ) || throw(DimensionMismatch("Observation dimension mismatch"))
    scaled, latent_chol, ld = prepared
    v = (x - μ) ./ sqrt.(D)
    z = latent_chol \ (scaled' * v)
    # Equivalent to Woodbury, but avoids subtracting two large quadratic terms.
    q = sum(abs2, v - scaled * z) + sum(abs2, z)
    return -0.5 * (length(μ) * log(2π) + ld + q)
end

function _factor_logpdf(μ, D, prepared, X::AbstractMatrix)
    size(X, 1) == length(μ) || throw(DimensionMismatch("Observation dimension mismatch"))
    scaled, latent_chol, ld = prepared
    V = (X .- μ) ./ sqrt.(D)
    Z = latent_chol \ (scaled' * V)
    q = vec(sum(abs2, V - scaled * Z; dims=1) + sum(abs2, Z; dims=1))
    return -0.5 .* (length(μ) * log(2π) + ld .+ q)
end

# Validate before indexing; exact conditioning on an output coordinate is singular.
function _structured_indices(d, indices)
    idx = collect(indices)
    all(i -> 1 <= i <= length(d), idx) || throw(ArgumentError("Indices out of bounds"))
    length(unique(idx)) == length(idx) || throw(ArgumentError("Indices must be unique"))
    return idx
end
