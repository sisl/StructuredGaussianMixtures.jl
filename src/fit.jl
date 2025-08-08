"""
    GMMFitMethod

Abstract type for Gaussian Mixture Model fitting methods.
"""
abstract type GMMFitMethod end
fit(gmmfit::GMMFitMethod, x::Matrix) = throw(MethodError(fit, (gmmfit, x)))
fit(gmmfit::GMMFitMethod, x::Matrix, weights::Vector) = throw(MethodError(fit, (gmmfit, x, weights)))

"""
    EM

Standard Expectation Maximization for fitting Gaussian Mixture Models.

# Fields
- `n_components`: Number of mixture components
- `method`: Initialization method (`:kmeans`, `:rand`, etc.)
- `kind`: Covariance structure (`:full`, `:diag`, etc.)
- `nInit`: Number of initializations
- `nIter`: Maximum number of iterations
- `nFinal`: Number of final iterations
"""
struct EM <: GMMFitMethod
    n_components::Int
    method::Symbol
    kind::Symbol
    nInit::Int
    nIter::Int
    nFinal::Int

    function EM(n_components::Int; 
    method::Symbol=:kmeans, 
    kind::Symbol=:full, 
    nInit::Int=50, 
    nIter::Int=10, 
    nFinal::Int=nIter)
        new(n_components, method, kind, nInit, nIter, nFinal)
    end
end


"""
    fit(fitmethod::EM, x::Matrix)

Fit a Gaussian Mixture Model using Expectation Maximization.

# Arguments
- `fitmethod`: The EM fitting method configuration
- `x`: The data matrix (n_samples × n_features)

# Returns
- A MixtureModel of MvNormal distributions

# Notes
- Uses GaussianMixtures.jl's GMM implementation
- Supports different initialization methods and covariance structures
"""
function fit(fitmethod::EM, x::Matrix)
   gmm = GMM(fitmethod.n_components, permutedims(x); method=fitmethod.method, kind=fitmethod.kind, nInit=fitmethod.nInit, nIter=fitmethod.nIter, nFinal=fitmethod.nFinal)
   return MixtureModel(gmm)
end

"""
    PCAEM

Mixture of Probabilistic Principal Component Analysis model.
Fits a GMM in PCA-reduced space and transforms back to original space.

# Fields
- `n_components`: Number of mixture components
- `rank`: Number of principal components to use
- `gmm_method`: Initialization method for GMM (`:kmeans`, `:rand`, etc.)
- `gmm_kind`: Covariance structure for GMM (`:full`, `:diag`, etc.)
- `gmm_nInit`: Number of GMM initializations
- `gmm_nIter`: Maximum number of GMM iterations
- `gmm_nFinal`: Number of final GMM iterations
"""
struct PCAEM <: GMMFitMethod
    n_components::Int
    rank::Int
    gmm_method::Symbol
    gmm_kind::Symbol
    gmm_nInit::Int
    gmm_nIter::Int
    gmm_nFinal::Int
    
    function PCAEM(n_components::Int, rank::Int; 
    gmm_method::Symbol=:kmeans, 
    gmm_kind::Symbol=:full, 
    gmm_nInit::Int=50, 
    gmm_nIter::Int=10, 
    gmm_nFinal::Int=gmm_nIter)
        new(n_components, rank, gmm_method, gmm_kind, gmm_nInit, gmm_nIter, gmm_nFinal)
    end
end

    
"""
    fit(fitmethod::PCAEM, x::Matrix)

Fit a Mixture of Probabilistic Principal Component Analysis model.
This method first performs PCA to reduce dimensionality, then fits a GMM in the reduced space,
and finally transforms the components back to the original space as LRDMvNormal distributions.

# Arguments
- `fitmethod`: The PCAEM fitting method configuration
- `x`: The data matrix (n_samples × n_features)

# Returns
- A MixtureModel of LRDMvNormal distributions

# Notes
- Uses PCA for dimensionality reduction
- Fits GMM in the reduced space
- Transforms components back to original space with low-rank plus diagonal structure
- The diagonal noise term is estimated from PCA reconstruction error
"""
function fit(fitmethod::PCAEM, x::Matrix)
    # run PCA on x
    pca = pca_fit(PCA, x; maxoutdim=fitmethod.rank);
    reduced_data = transform(pca, x);
    reconstructed_data = reconstruct(pca, reduced_data);
    error_data = x .- reconstructed_data
    D = vec(var(error_data, dims=2))
    μ = mean(pca)
    P = projection(pca)

    # fit GMM on the PCA scores
    gmm = GMM(fitmethod.n_components, permutedims(reduced_data); method=fitmethod.gmm_method, kind=fitmethod.gmm_kind, nInit=fitmethod.gmm_nInit, nIter=fitmethod.gmm_nIter, nFinal=fitmethod.gmm_nFinal)
    gmm = MixtureModel(gmm)

    # for each component of the GMM, make a LRDMvNormal distribution, with the following parameters:
    # μ = μ + P * comp.μ, where comp.μ is the mean of the k-th component of the GMM
    # F = P * comp.Σ^0.5, where comp.Σ is the covariance of the k-th component of the GMM
    # D = D
    lr_components = components(gmm)
    @assert length(lr_components) == fitmethod.n_components "The number of components in the GMM must be equal to the number of components in the MPPCA"
    comps = Vector{LRDMvNormal}(undef, fitmethod.n_components)
    
    for (k, comp) in enumerate(lr_components)
        # Compute mean in original space
        μ_k = μ + P * mean(comp)
        
        # Compute low-rank factor using eigendecomposition for stability
        λ, Q = eigen(cov(comp))
        F_k = P * (Q * Diagonal(sqrt.(λ)))
        
        # Create the component
        comps[k] = LRDMvNormal(μ_k, F_k, D)
    end
    
    return MixtureModel(comps, probs(gmm))
end

"""
    FactorEM

Mixture of Factor Analyzers model.
Directly fits a mixture of low-rank plus diagonal Gaussian distributions.

# Fields
- `n_components`: Number of mixture components
- `rank`: Rank of the low-rank factor
- `gmm_method`: Initialization method for GMM (`:kmeans`, `:rand`, etc.)
- `gmm_nInit`: Number of GMM initializations
- `gmm_nIter`: Maximum number of GMM iterations
- `gmm_nFinal`: Number of final GMM iterations
"""
struct FactorEM <: GMMFitMethod
    n_components::Int
    rank::Int
    initialization_method::Symbol # :kmeans, :rand
    nInit::Int # number of initializations for the GMM
    nIter::Int # number of EM iterations for the GMM
    nInternalIter::Int # number of EM iterations for the internal factor analysis

    function FactorEM(n_components::Int, rank::Int; 
        initialization_method::Symbol=:kmeans, 
        nInit::Int=1, 
        nIter::Int=10, 
        nInternalIter::Int=10)
        new(n_components, rank, initialization_method, nInit, nIter, nInternalIter)
    end
end


"""
    fit(fitmethod::FactorEM, x::Matrix)

Fit a Mixture of Factor Analyzers model using Expectation Maximization.
This method directly fits a mixture of low-rank plus diagonal Gaussian distributions.

# Arguments
- `fitmethod`: The FactorEM fitting method configuration
- `x`: The data matrix (n_samples × n_features)

# Returns
- A MixtureModel of LRDMvNormal distributions

# Notes
- Directly fits the low-rank plus diagonal structure
- More computationally intensive than PCAEM but potentially more accurate
- Not yet implemented
"""
fit(fitmethod::FactorEM, x::Matrix) = fit(fitmethod, x, ones(size(x, 2)) / size(x, 2))

"""
    fit(fitmethod::FactorEM, x::Matrix, weights::Vector)

Fit a Mixture of Factor Analyzers model using Expectation Maximization with weighted data points.

# Arguments
- `fitmethod`: The FactorEM fitting method configuration
- `x`: The data matrix (n_samples × n_features)
- `weights`: Vector of weights for each data point

# Returns
- A MixtureModel of LRDMvNormal distributions

# Notes
- Supports weighted data points for importance sampling or missing data scenarios
- Weights are automatically normalized to sum to 1
- Uses the same EM algorithm but with weighted responsibilities
"""
function fit(fitmethod::FactorEM, x::Matrix, weights::Vector)
    norm_weights = weights ./ sum(weights)
    best_ll = -Inf
    best_gmm = nothing
    for i in 1:fitmethod.nInit
        gmm = initialize_gmm(fitmethod.initialization_method, fitmethod.n_components, fitmethod.rank, x)

        # run EM
        for j in 1:fitmethod.nIter

            # E-step
            log_resp = e_step(gmm, x)

            # M-step
            m_step!(gmm, x, log_resp, norm_weights; nInternalIter=fitmethod.nInternalIter)
        end

        # compute final logprobs to return the best fit
        lls = logpdf(gmm, x)
        ll = lls' * norm_weights
        if ll > best_ll
            best_ll = ll
            best_gmm = gmm
        end
    end

    return best_gmm
end

function initialize_gmm(method::Symbol, n_components::Int, rank::Int, x::Matrix; epsilon::Float64=0.01)
    n_features, n_samples = size(x)
    global_var = var(x, dims=2)[:]
    if method == :kmeans
        # Run k-means on x
        kmeans_result = kmeans(x, n_components)
        assigns = assignments(kmeans_result)
        centers = kmeans_result.centers
        
        # Initialize components
        components = Vector{LRDMvNormal}(undef, n_components)
        weights = zeros(n_components)
        
        for k in 1:n_components
            # Get data points in this cluster
            cluster_mask = assigns .== k
            cluster_data = x[:, cluster_mask]
            
            if isempty(cluster_data)
                # If cluster is empty, use small random initialization
                μ = centers[:, k]
                F = zeros(n_features, rank)
                D = global_var
            else
                # Compute mean and variance for this cluster
                μ = mean(cluster_data, dims=2)[:]
                D = var(cluster_data, dims=2)[:]
                
                # Initialize low-rank factor with small random values
                F = zeros(n_features, rank)
            end
            components[k] = LRDMvNormal(μ, F, D)
            weights[k] = count(cluster_mask) / n_samples
        end
        
        return MixtureModel(components, weights)
    elseif method == :rand
        # Choose n_components random rows of x as means
        rows = rand(1:n_samples, n_components)
        means = x[:,rows]
        
        # Initialize components with random low-rank structure
        components = Vector{LRDMvNormal}(undef, n_components)
        for k in 1:n_components
            μ = means[:, k]
            F = zeros(n_features, rank)
            D = global_var
            components[k] = LRDMvNormal(μ, F, D)
        end
        
        # Use uniform weights
        weights = ones(n_components) / n_components
        return MixtureModel(components, weights)
    else
        throw(ArgumentError("Invalid initialization method: $method"))
    end
end


function e_step(gmm::MixtureModel, x::Matrix)
    n_features, n_samples = size(x)
    n_components = length(components(gmm))
    log_resp = Matrix{eltype(x)}(undef, n_samples, n_components)
    for i in 1:n_samples
        log_weights = log.(probs(gmm))
        for j in 1:n_components
            log_resp[i, j] = logpdf(components(gmm)[j], x[:,i]) + log_weights[j]
        end
    end
    
    # Normalize log responsibilities in a numerically stable way
    max_logs = maximum(log_resp, dims=2)
    log_resp .-= max_logs .+ log.(sum(exp.(log_resp .- max_logs), dims=2))
    return log_resp
end


function m_step!(gmm::MixtureModel, x::Matrix, log_resp::Matrix, point_weights::Vector; nInternalIter::Int=10)
    n_samples, n_components = size(log_resp)
    n_features = length(components(gmm)[1].μ)
    
    # Convert log responsibilities to responsibilities
    resp = exp.(log_resp)
    
    # Update mixture weights
    gmm.prior.p .= resp' * point_weights
    
    # Update means and covariance structure for each component
    for j in 1:n_components
        comp = components(gmm)[j]
        weights = resp[:, j] .* point_weights
        sum_weights = sum(weights)
        # Update mean in-place
        μ = sum(weights' .* x, dims=2)[:] / sum_weights
        gmm.components[j].μ .= μ
        
        # Compute residuals
        r = x .- μ
        
        # Compute weighted residual covariance diagonal
        C_r = sum(weights' .* (r.^2), dims=2)[:] / sum_weights
        
        # Initialize D to residual covariance diagonal
        D = deepcopy(C_r)
        
        # Initialize F to previous value (or clipped identity if first iteration)
        F = comp.F
        if all(iszero, F)
            F .= 0.1 * Matrix{Float64}(I, n_features, size(F, 2))
        end
        
        # Inner EM iterations for F and D
        for iter in 1:nInternalIter
            # Inner E-step
            # Compute G = (F'D^-1F + I)^-1
            G = inv(F' * ( 1 ./ D  .* F) + I)

            # Compute expected latent variables s
            s = G * F' * ((1 ./ D) .* r)
            
            # Compute weighted expectations
            C_rs = (weights' .* r) * s' / sum_weights
            C_ss = (weights' .* s) * s' / sum_weights + G
            
            # Inner M-step
            # Update F
            F .= C_rs * inv(C_ss)
            
            # Update D
            D .= C_r + sum((F * C_ss) .* F, dims=2)[:] - 2 * sum(C_rs .* F, dims=2)[:]
        end

        # Update the component
        gmm.components[j].F .= F
        gmm.components[j].D .= D
    end
end
