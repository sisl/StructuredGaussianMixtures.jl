using StructuredGaussianMixtures
using Test

# Run all test suites
include("test_lrdmvnormal.jl")
include("test_fitting.jl")
include("test_prediction.jl")

include("test_latentmvnormal.jl")

include("test_structured_interface.jl")

include("test_latentfit.jl")
