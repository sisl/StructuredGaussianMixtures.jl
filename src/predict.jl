"""
    predict(dist::MvNormal, x::AbstractVector, input_indices::Union{Vector{Int},AbstractRange}, output_indices::Union{Vector{Int},AbstractRange})

Compute the conditional distribution of the output indices given the input indices using the Schur complement.
Returns a new MvNormal distribution representing the conditional distribution.

# Arguments
- `dist`: The multivariate normal distribution
- `x`: The observed values for the input indices
- `input_indices`: The indices of the observed variables
- `output_indices`: The indices of the variables to predict

# Returns
- A new MvNormal distribution representing the conditional distribution
"""
function predict(
    dist::MvNormal,
    x::AbstractVector,
    input_indices::Union{Vector{Int},AbstractRange},
    output_indices::Union{Vector{Int},AbstractRange},
)
    # Get the full mean and covariance
    μ = mean(dist)
    Σ = cov(dist)
    d = length(μ)

    # Verify indices are valid
    all(i -> 1 ≤ i ≤ d, input_indices) ||
        throw(ArgumentError("Input indices out of bounds"))
    all(i -> 1 ≤ i ≤ d, output_indices) ||
        throw(ArgumentError("Output indices out of bounds"))

    # Split the mean vector
    μ₁ = μ[input_indices]
    μ₂ = μ[output_indices]

    # Split the covariance matrix
    Σ₁₁ = Σ[input_indices, input_indices]
    Σ₁₂ = Σ[input_indices, output_indices]
    Σ₂₁ = Σ[output_indices, input_indices]
    Σ₂₂ = Σ[output_indices, output_indices]

    # Compute the Schur complement
    Σ₂₂₁ = Σ₂₂ - Σ₂₁ * (Σ₁₁ \ Σ₁₂)

    # Compute the conditional mean
    μ₂₁ = μ₂ + Σ₂₁ * (Σ₁₁ \ (x - μ₁))

    # Return the conditional distribution
    return MvNormal(μ₂₁, Σ₂₂₁)
end

"""
    predict(dist::LRDMvNormal, x::AbstractVector, input_indices::Union{Vector{Int},AbstractRange}, output_indices::Union{Vector{Int},AbstractRange})

Compute the conditional distribution of the output indices given the input indices using the Schur complement.
Returns a new LRDMvNormal distribution representing the conditional distribution.
This implementation is efficient for low-rank plus diagonal covariance structure.

# Arguments
- `dist`: The low-rank plus diagonal multivariate normal distribution
- `x`: The observed values for the input indices
- `input_indices`: The indices of the observed variables
- `output_indices`: The indices of the variables to predict

# Returns
- A new LRDMvNormal distribution representing the conditional distribution
"""
function predict(
    dist::LRDMvNormal,
    x::AbstractVector,
    input_indices::Union{Vector{Int},AbstractRange},
    output_indices::Union{Vector{Int},AbstractRange},
)
    d = length(dist.μ)

    # Verify indices are valid
    all(i -> 1 ≤ i ≤ d, input_indices) ||
        throw(ArgumentError("Input indices out of bounds"))
    all(i -> 1 ≤ i ≤ d, output_indices) ||
        throw(ArgumentError("Output indices out of bounds"))

    # Split the mean vector
    μ₁ = dist.μ[input_indices]
    μ₂ = dist.μ[output_indices]

    # Split the low-rank factor and diagonal
    F₁ = dist.F[input_indices, :]
    F₂ = dist.F[output_indices, :]
    D₁ = dist.D[input_indices]
    D₂ = dist.D[output_indices]

    # Compute D₁^(-1/2) * F₁ efficiently
    D₁_inv_sqrt = 1 ./ sqrt.(D₁)
    F₁_scaled = F₁ .* D₁_inv_sqrt

    # Compute (I + F₁' * D₁^(-1) * F₁)^(-1) efficiently
    I_plus_FF = I + F₁_scaled' * F₁_scaled
    I_plus_FF_inv = inv(I_plus_FF)

    # Compute the conditional mean efficiently
    x_centered = x - μ₁
    D₁_inv = 1 ./ D₁
    F₁_D₁_inv = F₁ .* D₁_inv
    μ₂₁ = μ₂ + F₂ * (I_plus_FF_inv * (F₁_scaled' * (D₁_inv_sqrt .* x_centered)))

    # Compute the conditional covariance structure efficiently
    # For the low-rank part: F₂ * (I - I_plus_FF_inv) * F₂'
    # Compute matrix square root using eigendecomposition
    λ, Q = eigen(I - I_plus_FF_inv)
    F₂_cond = F₂ * (Q * Diagonal(sqrt.(λ)))

    # For the diagonal part: D₂ + diag(F₂ * I_plus_FF_inv * F₂')
    D₂_cond = D₂ + diag(F₂ * (I_plus_FF_inv * F₂'))

    # Return the conditional distribution
    if length(output_indices) <= dist.rank
        return MvNormal(μ₂₁, F₂_cond * F₂_cond' + Diagonal(D₂_cond))
    else
        return LRDMvNormal(μ₂₁, F₂_cond, D₂_cond)
    end
end

"""
    marginal(dist::MvNormal, indices::Union{Vector{Int},AbstractRange})

Compute the marginal distribution over the specified indices.
Returns a new MvNormal distribution representing the marginal.

# Arguments
- `dist`: The multivariate normal distribution
- `indices`: The indices to marginalize over

# Returns
- A new MvNormal distribution representing the marginal
"""
function marginal(dist::MvNormal, indices::Union{Vector{Int},AbstractRange})
    μ = mean(dist)[indices]
    Σ = cov(dist)[indices, indices]
    return MvNormal(μ, Σ)
end

"""
    marginal(dist::LRDMvNormal, indices::Union{Vector{Int},AbstractRange})

Compute the marginal distribution over the specified indices.
Returns a new LRDMvNormal distribution representing the marginal.

# Arguments
- `dist`: The low-rank plus diagonal multivariate normal distribution
- `indices`: The indices to marginalize over

# Returns
- A new LRDMvNormal distribution representing the marginal
"""
function marginal(dist::LRDMvNormal, indices::Union{Vector{Int},AbstractRange})
    μ = dist.μ[indices]

    # If the number of indices is less than or equal to the rank, use full rank
    if length(indices) <= dist.rank
        # Compute the full covariance matrix for the marginal
        Σ = dist.F[indices, :] * dist.F[indices, :]' + Diagonal(dist.D[indices])
        return MvNormal(μ, Σ)
    else
        F = dist.F[indices, :]
        D = dist.D[indices]
        return LRDMvNormal(μ, F, D)
    end
end

"""
    predict(dist::MultivariateMixture, x::AbstractVector, input_indices::Union{Vector{Int},AbstractRange}, output_indices::Union{Vector{Int},AbstractRange})

Compute the conditional distribution of the output indices given the input indices for a mixture model.
Returns a new mixture model where each component is the conditional distribution of the corresponding component,
and the weights are updated based on the log density of x under the marginal distributions.

# Arguments
- `dist`: The multivariate mixture distribution
- `x`: The observed values for the input indices
- `input_indices`: The indices of the observed variables
- `output_indices`: The indices of the variables to predict

# Returns
- A new mixture model representing the conditional distribution
"""
function predict(
    dist::MultivariateMixture,
    x::AbstractVector,
    input_indices::Union{Vector{Int},AbstractRange},
    output_indices::Union{Vector{Int},AbstractRange},
)
    # Get the number of components
    n_components = length(dist.components)

    # Compute log densities of x under each component's marginal
    log_densities = zeros(n_components)
    for k in 1:n_components
        # Get the marginal distribution for the input indices
        marginal_dist = marginal(dist.components[k], input_indices)
        log_densities[k] = logpdf(marginal_dist, x)
    end

    # Compute new weights using softmax
    log_weights = log.(probs(dist.prior)) .+ log_densities
    log_weights .-= maximum(log_weights)  # For numerical stability
    new_weights = exp.(log_weights)
    new_weights ./= sum(new_weights)

    # Get conditional distributions for each component
    new_components = [
        predict(dist.components[k], x, input_indices, output_indices) for
        k in 1:n_components
    ]

    # Return the new mixture model
    return MixtureModel(new_components, new_weights)
end

"""
    predict(dist::Union{MvNormal,LRDMvNormal,MultivariateMixture}, x::AbstractVector; 
           input_indices::Union{Vector{Int},AbstractRange} = 1:length(x), 
           output_indices::Union{Vector{Int},AbstractRange} = length(x)+1:length(mean(dist)))

Compute the conditional distribution of the output indices given the input indices using the Schur complement.
Returns a new distribution representing the conditional distribution.

# Arguments
- `dist`: The multivariate normal distribution (MvNormal or LRDMvNormal)
- `x`: The observed values for the input indices
- `input_indices`: The indices of the observed variables (default: first length(x) indices)
- `output_indices`: The indices of the variables to predict (default: remaining indices)

# Returns
- A new distribution representing the conditional distribution
"""
function predict(
    dist::Union{MvNormal,LRDMvNormal,MultivariateMixture},
    x::AbstractVector;
    input_indices::Union{Vector{Int},AbstractRange}=1:length(x),
    output_indices::Union{Vector{Int},AbstractRange}=(length(x) + 1):length(mean(dist)),
)
    # Convert ranges to vectors if needed
    input_indices_vec = collect(input_indices)
    output_indices_vec = collect(output_indices)
    return predict(dist, x, input_indices_vec, output_indices_vec)
end