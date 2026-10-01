__precompile__(false)

module StructuredGaussianMixtures

using Distributions
import Distributions: fit
using LinearAlgebra
using Statistics
using Random
using Arpack: eigs
include("factoroperations.jl")
include("lrdmvnormal.jl")
export LRDMvNormal, rank, low_rank_factor, diagonal

include("latentmvnormal.jl")
export LatentMvNormal, loading, latent_covariance_factor, latent_covariance

include("fitspecs.jl")
include("fitdata.jl")
include("covariancefit.jl")
include("initialization.jl")
include("em.jl")
include("fit.jl")
include("pca.jl")
export fit, fit!, initialize, workspace, responsibilities
export FullCovariance,
    DiagonalCovariance, LowRankDiagonal, LatentCovariance, MixtureSpec, Tied
export Exact, CovarianceEM, EM, PCAEM, KMeansInit, RandomInit, RandomLoading
export FitReport, GaussianWorkspace, MixtureWorkspace, PCAWorkspace

include("predict.jl")
include("bandedprecision.jl")
export BandedPrecision, BandedPrecisionMvNormal
export predict, marginal

end # module
