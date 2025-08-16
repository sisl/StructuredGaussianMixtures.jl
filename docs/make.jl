using Documenter
using Random
using StructuredGaussianMixtures
using Distributions

makedocs(
    sitename = "StructuredGaussianMixtures",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
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

deploydocs(
    repo = "github.com/sisl/StructuredGaussianMixtures.jl",
    push_preview = true
)
