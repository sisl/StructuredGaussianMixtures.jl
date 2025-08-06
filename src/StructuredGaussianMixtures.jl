__precompile__(false)

module StructuredGaussianMixtures

using Distributions
using LinearAlgebra
using Statistics
using Random
using MultivariateStats: PCA, fit as pca_fit, transform, reconstruct, projection, mean
using GaussianMixtures
import GaussianMixtures: covar
using Clustering

## conversion to MixtureModel since GaussianMixtures.jl fails for d=1
function Distributions.MixtureModel(gmm::GMM{T}) where {T<:AbstractFloat}
    # if gmm.d == 1
    #     mixtures = [Normal(gmm.μ[i,1], covar(gmm.Σ[i])) for i=1:gmm.n]
    if kind(gmm) == :full
        mixtures = [MvNormal(vec(gmm.μ[i,:]), covar(gmm.Σ[i])) for i=1:gmm.n]
    else
        mixtures = [MvNormal(vec(gmm.μ[i,:]), sqrt.(vec(gmm.Σ[i,:]))) for i=1:gmm.n]
    end
    MixtureModel(mixtures, gmm.w)
end

include("lrdmvnormal.jl")
export LRDMvNormal, rank, low_rank_factor, diagonal

include("fit.jl")
export fit, GMMFitMethod, EM, PCAEM, FactorEM

include("predict.jl")
export predict, marginal

end # module
