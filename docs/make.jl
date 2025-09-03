using Documenter
using Random
using StructuredGaussianMixtures
using Distributions

makedocs(;
    sitename="StructuredGaussianMixtures",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true"),
    modules=[StructuredGaussianMixtures],
    pages=[
        "Home" => "index.md",
        "Fitting Methods" =>
            ["Fitting" => "fitting.md", "Structured Gaussians" => "lrdmvnormal.md"],
        "Prediction" => "prediction.md",
        "Examples" => "examples.md",
    ],
    remotes=nothing,
)

if get(ENV, "CI", "false") == "true"
    deploydocs(; repo="github.com/sisl/StructuredGaussianMixtures.jl", push_preview=true)
end
