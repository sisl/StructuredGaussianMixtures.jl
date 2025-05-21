module StructuredGaussianMixtures

using Distributions
using LinearAlgebra
using Statistics

include("lrdmvnormal.jl")
export LRDMvNormal

include("fit.jl")
export fit
include("predict.jl")
export predict, marginal

end # module
