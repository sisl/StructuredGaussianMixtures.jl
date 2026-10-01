# Structure, optimization, and mutable fitting state are deliberately separate.
abstract type GaussianStructure end
"""Unrestricted positive definite covariance."""
struct FullCovariance <: GaussianStructure end
"""Independent coordinates with positive variances."""
struct DiagonalCovariance <: GaussianStructure end
"""Equal positive variance in every coordinate: `σ² I`."""
struct IsotropicCovariance <: GaussianStructure end
"""Covariance `F*F' + Diagonal(D)` with `r` loading columns."""
struct LowRankDiagonal <: GaussianStructure
    r::Int
    function LowRankDiagonal(r::Integer)
        r >= 0 || throw(ArgumentError("rank must be nonnegative"))
        return new(r)
    end
end
"""Covariance `F*A*F' + Diagonal(D)` with `r` latent coordinates."""
struct LatentCovariance{S<:GaussianStructure} <: GaussianStructure
    r::Int
    latent::S
    function LatentCovariance(r::Integer; latent=FullCovariance())
        r > 0 || throw(ArgumentError("latent rank must be positive"))
        latent isa Union{FullCovariance,DiagonalCovariance} ||
            throw(ArgumentError("latent covariance must be full or diagonal"))
        return new{typeof(latent)}(r, latent)
    end
end
"""Declare shared parameter names, e.g. `Tied(:F, :D)` for PCAEM."""
struct Tied
    parameters::Tuple{Vararg{Symbol}}
    function Tied(parameters::Symbol...)
        length(unique(parameters)) == length(parameters) ||
            throw(ArgumentError("duplicate tied parameter"))
        return new(parameters)
    end
end
"""Mixture structure, component count, and parameter sharing (independent by default)."""
struct MixtureSpec{S<:GaussianStructure}
    covariance::S
    k::Int
    tied::Tied
    function MixtureSpec(s::S, k::Integer; tied=Tied()) where {S<:GaussianStructure}
        k > 0 || throw(ArgumentError("component count must be positive"))
        return new{S}(s, k, tied)
    end
end

abstract type CovarianceMethod end
abstract type MixtureMethod end
abstract type Initialization end
"""Initialize mixture centers using weighted k-means; `maxiter` is not a restart count."""
struct KMeansInit <: Initialization
    maxiter::Int
    function KMeansInit(; maxiter=50)
        maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
        return new(maxiter)
    end
end
"""Choose mixture centers from observations, proportional to observation weights."""
struct RandomInit <: Initialization end
"""Initialize an LRD covariance from a small nonzero random loading and marginal variances."""
struct RandomLoading <: Initialization end
"""Weighted covariance MLE, with an optional additive diagonal ridge."""
struct Exact <: CovarianceMethod
    regularization::Float64
    function Exact(; regularization=1e-6)
        isfinite(regularization) && regularization >= 0 ||
            throw(ArgumentError("regularization must be finite and nonnegative"))
        return new(regularization)
    end
end
"""Inner factor-analysis EM; `variance_floor` constrains residual variances."""
struct CovarianceEM{I<:Initialization} <: CovarianceMethod
    maxiter::Int
    tol::Float64
    variance_floor::Float64
    init::I
    function CovarianceEM(; maxiter=20, tol=1e-6, variance_floor=1e-6, init=RandomLoading())
        maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
        isfinite(tol) && tol >= 0 ||
            throw(ArgumentError("tol must be finite and nonnegative"))
        isfinite(variance_floor) && variance_floor >= 0 ||
            throw(ArgumentError("variance_floor must be finite and nonnegative"))
        init isa RandomLoading ||
            throw(ArgumentError("CovarianceEM requires RandomLoading initialization"))
        return new{typeof(init)}(maxiter, tol, variance_floor, init)
    end
end
"""Native weighted mixture EM. `fit` performs restarts; `fit!` continues existing parameters.
Collapsed components cause a reported failed run (`min_mass` is a normalized mass).
"""
struct EM{C<:CovarianceMethod,I<:Initialization} <: MixtureMethod
    covariance_method::C
    init::I
    maxiter::Int
    tol::Float64
    n_init::Int
    min_mass::Float64
    function EM(;
        covariance_method=Exact(),
        init=KMeansInit(),
        maxiter=100,
        tol=1e-6,
        n_init=1,
        min_mass=1e-12,
    )
        maxiter >= 0 && n_init > 0 ||
            throw(ArgumentError("invalid iteration or restart count"))
        isfinite(tol) && tol >= 0 ||
            throw(ArgumentError("tol must be finite and nonnegative"))
        0 <= min_mass < 1 || throw(ArgumentError("min_mass must be in [0,1)"))
        init isa Union{KMeansInit,RandomInit} ||
            throw(ArgumentError("unsupported mixture initialization"))
        return new{typeof(covariance_method),typeof(init)}(
            covariance_method, init, maxiter, tol, n_init, min_mass
        )
    end
end
"""PCA projection followed by `latent_method`; requires LatentCovariance and Tied(:F,:D)."""
struct PCAEM{M<:EM} <: MixtureMethod
    latent_method::M
    residual_floor::Float64
    function PCAEM(; latent_method=EM(), residual_floor=1e-6)
        latent_method.covariance_method isa Exact ||
            throw(ArgumentError("PCAEM latent method must use Exact"))
        isfinite(residual_floor) && residual_floor >= 0 ||
            throw(ArgumentError("residual_floor must be finite and nonnegative"))
        return new{typeof(latent_method)}(latent_method, residual_floor)
    end
end

"""Convergence diagnostics; objectives are normalized weighted log likelihoods.
`status` is :initialized, :converged, :iteration_limit, or :failed.
PCAEM reports its reduced-space objective, with observed likelihood separately.
"""
mutable struct FitReport
    status::Symbol
    iterations::Int
    objective::Float64
    history::Vector{Float64}
    message::String
    objective_kind::Symbol
    observed_objective::Union{Nothing,Float64}
    runs::Vector{FitReport}
end
function FitReport(; kind=:observed_loglikelihood)
    return FitReport(:initialized, 0, -Inf, Float64[], "", kind, nothing, FitReport[])
end
"""Mutable initialized single-Gaussian fitting state; diagnostics are in `.report`."""
mutable struct GaussianWorkspace{S<:GaussianStructure}
    spec::S
    model::Distributions.AbstractMvNormal
    report::FitReport
end
"""Mutable initialized mixture state; nested covariance diagnostics are in `.inner_reports`."""
mutable struct MixtureWorkspace{S<:MixtureSpec}
    spec::S
    model::MixtureModel
    report::FitReport
    inner_reports::Vector{FitReport}
end
"""PCA fitting state retaining its projection, offset, noise, and latent workspace."""
mutable struct PCAWorkspace{S<:MixtureSpec}
    spec::S
    F::Matrix{Float64}
    offset::Vector{Float64}
    D::Vector{Float64}
    latent::MixtureWorkspace
    model::MixtureModel
    report::FitReport
end
