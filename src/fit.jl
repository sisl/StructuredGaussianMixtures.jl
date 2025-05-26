"""
    GMMFitMethod

Abstract type for Gaussian Mixture Model fitting methods.
"""
abstract type GMMFitMethod end
fit(gmmfit::GMMFitMethod, x::Matrix) = throw(MethodError(fit, (gmmfit, x)))

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
end
EM(n_components::Int; 
    method::Symbol=:kmeans, 
    kind::Symbol=:full, 
    nInit::Int=50, 
    nIter::Int=10, 
    nFinal::Int=nIter) = EM(n_components, method, kind, nInit, nIter, nFinal)

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
end
PCAEM(n_components::Int, rank::Int; 
    gmm_method::Symbol=:kmeans, 
    gmm_kind::Symbol=:full, 
    gmm_nInit::Int=50, 
    gmm_nIter::Int=10, 
    gmm_nFinal::Int=gmm_nIter) = PCAEM(n_components, rank, gmm_method, gmm_kind, gmm_nInit, gmm_nIter, gmm_nFinal)
    
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
    gmm_method::Symbol
    gmm_nInit::Int
    gmm_nIter::Int
    gmm_nFinal::Int
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
function fit(fitmethod::FactorEM, x::Matrix)
    error("Not implemented")
end