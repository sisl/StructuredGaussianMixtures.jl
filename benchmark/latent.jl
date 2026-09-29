# Run: julia --project=. benchmark/latent.jl
# For a baseline checkout: julia --project=/path/to/baseline benchmark/latent.jl
# No benchmarking dependencies; warm each closure before measuring median elapsed
# time and minimum allocated bytes. Compare processes with identical BLAS threads.
using StructuredGaussianMixtures, Distributions, LinearAlgebra, Random, Statistics
BLAS.set_num_threads(1)
rng = MersenneTwister(731)
println("Julia ", VERSION, "; BLAS threads=", BLAS.get_num_threads())
println("case,median_ms,min_bytes")
function measure(name, f; repeats=15)
    f()
    f()
    times = Float64[]
    bytes = Int[]
    for _ in 1:repeats
        GC.gc()
        result = @timed f()
        push!(times, result.time * 1000)
        push!(bytes, result.bytes)
    end
    return println(name, ",", round(median(times); digits=4), ",", minimum(bytes))
end
for (p, r, n) in ((100, 5, 500), (500, 10, 1000))
    μ, L, D, B = randn(rng, p), randn(rng, p, r), ones(p), randn(rng, r, r)
    X = randn(rng, p, n)
    g = LRDMvNormal(μ, L * B, D)
    obs, target = collect(1:2:p), collect(2:2:p)
    x = X[obs, 1]
    measure("lrd_scalar_$(p)_$(r)", () -> logpdf(g, X[:, 1]))
    measure("lrd_batch_$(p)_$(r)_$(n)", () -> logpdf(g, X))
    measure("lrd_condition_$(p)_$(r)", () -> predict(g, x, obs, target))
    if isdefined(StructuredGaussianMixtures, :LatentMvNormal)
        latent = LatentMvNormal(μ, L, D, B)
        measure("latent_scalar_$(p)_$(r)", () -> logpdf(latent, X[:, 1]))
        measure("latent_batch_$(p)_$(r)_$(n)", () -> logpdf(latent, X))
        measure("latent_condition_$(p)_$(r)", () -> predict(latent, x, obs, target))
        measure("latent_sample_$(p)_$(r)_$(n)", () -> rand(rng, latent, n))
    end
end
X = randn(rng, 60, 300)
measure(
    "pca_fit_60_5_300",
    () -> begin
        Random.seed!(34)
        StructuredGaussianMixtures.fit(PCAEM(3, 5; gmm_nInit=2, gmm_nIter=5), X)
    end;
    repeats=5,
)
measure(
    "factor_fit_60_5_300",
    () -> begin
        Random.seed!(34)
        StructuredGaussianMixtures.fit(
            FactorEM(3, 5; nInit=1, nIter=3, nInternalIter=3), X
        )
    end;
    repeats=5,
)
