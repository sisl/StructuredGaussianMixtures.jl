using Documenter
using Random
using StructuredGaussianMixtures
using Distributions

makedocs(
    sitename = "StructuredGaussianMixtures",
    format = Documenter.HTML(prettyurls = false),
    modules = [StructuredGaussianMixtures],
    pages = [
        "Home" => "index.md",
        "Fitting Methods" => [
            "Overview" => "fitting.md",
            "LRDMvNormal" => "lrdmvnormal.md"
        ],
        "Prediction" => "prediction.md",
        "Examples" => "examples.md",
    ],
    remotes = nothing
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
