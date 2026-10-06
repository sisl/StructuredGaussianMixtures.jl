"""Positive definite stationary covariance with constant diagonals (one lag per coordinate separation)."""
struct ToeplitzCovariance <: GaussianStructure end

"""Iterative Toeplitz covariance likelihood fitting by limited-memory BFGS with SPD Armijo steps.
`regularization` adds a ridge to the empirical scatter (a precision-trace penalty).
`tol` controls the scaled lag-gradient norm; `maxiter` and `max_backtracks` bound work.
`memory` controls the number of retained curvature pairs. No global optimum is
guaranteed. Fitting uses dense sufficient statistics and O(p³) accepted-step gradients.
"""
struct ToeplitzMLE <: CovarianceMethod
    regularization::Float64
    maxiter::Int
    tol::Float64
    max_backtracks::Int
    memory::Int
    function ToeplitzMLE(; regularization=1e-6, maxiter=200, tol=1e-6, max_backtracks=60, memory=10)
        isfinite(regularization) && regularization>=0 ||
            throw(ArgumentError("regularization must be finite and nonnegative"))
        maxiter>=0 && max_backtracks>0 || throw(ArgumentError("invalid iteration limit"))
        isfinite(tol) && tol>=0 ||
            throw(ArgumentError("tol must be finite and nonnegative"))
        memory>0 || throw(ArgumentError("memory must be positive"))
        return new(regularization, maxiter, tol, max_backtracks, memory)
    end
end

_toeplitz_matrix(c) = [c[abs(i - j) + 1] for i in eachindex(c), j in eachindex(c)]

# Durbin recursion: W maps centered observations to standardized innovations.
# Each new row is the best linear predictor from preceding coordinates.
function _toeplitz_whitener(c)
    p=length(c)
    W=zeros(p, p)
    variance=c[1]
    variance>0 || throw(PosDefException(1))
    W[1, 1]=inv(sqrt(variance))
    logdeterminant=log(variance)
    a=Float64[]
    for m in 1:(p - 1)
        reflection=(c[m + 1]-dot(a, c[m:-1:2]))/variance
        isfinite(reflection) && abs(reflection)<1 || throw(PosDefException(m+1))
        a=vcat(a-reflection*reverse(a), reflection)
        variance*=1-reflection^2
        isfinite(variance) && variance>0 || throw(PosDefException(m+1))
        scale=inv(sqrt(variance))
        W[m + 1, m + 1]=scale
        W[m + 1, 1:m] .= -reverse(a) .* scale
        logdeterminant+=log(variance)
    end
    return LowerTriangular(W), logdeterminant
end

"""`ToeplitzMvNormal(mean, first_column)` stores a real symmetric Toeplitz covariance.
Inputs are copied. Scoring caches a Durbin innovations transform: O(p²) setup and
storage, O(p²) work per observation, and O(1) cached log determinant. Covariance
materialization and arbitrary conditioning return dense matrices/distributions.
Fields are internal; do not mutate their contents.
"""
struct ToeplitzMvNormal <: Distributions.AbstractMvNormal
    μ::Vector{Float64}
    c::Vector{Float64}
    W::LowerTriangular{Float64,Matrix{Float64}}
    logdeterminant::Float64
    function ToeplitzMvNormal(μ::AbstractVector, c::AbstractVector)
        length(μ)==length(c) ||
            throw(DimensionMismatch("mean and covariance dimension mismatch"))
        isempty(c) && throw(ArgumentError("dimension must be positive"))
        all(isfinite, μ) && all(isfinite, c) ||
            throw(ArgumentError("parameters must be finite"))
        owned=Float64.(c)
        ownedmean=Float64.(μ)
        all(isfinite, owned) && all(isfinite, ownedmean) ||
            throw(ArgumentError("parameters exceed Float64 range"))
        W, ld=_toeplitz_whitener(owned)
        return new(ownedmean, owned, W, ld)
    end
end
Distributions.partype(::ToeplitzMvNormal) = Float64
Distributions.length(g::ToeplitzMvNormal) = length(g.μ)
Distributions.size(g::ToeplitzMvNormal) = (length(g),)
Distributions.mean(g::ToeplitzMvNormal) = copy(g.μ)
Distributions.var(g::ToeplitzMvNormal) = fill(g.c[1], length(g))
Distributions.cov(g::ToeplitzMvNormal) = _toeplitz_matrix(g.c)
Distributions.logdetcov(g::ToeplitzMvNormal) = g.logdeterminant
# Explicit precision requests materialize the dense inverse covariance.
Distributions.invcov(g::ToeplitzMvNormal) = first(_toeplitz_precision(g.c))
function Distributions.sqmahal(g::ToeplitzMvNormal, x::AbstractVector)
    length(x)==length(g) || throw(DimensionMismatch("observation dimension mismatch"))
    return sum(abs2, g.W*(x-g.μ))
end
function Distributions.sqmahal!(out::AbstractVector, g::ToeplitzMvNormal, X::AbstractMatrix)
    size(X, 1)==length(g) || throw(DimensionMismatch("observation dimension mismatch"))
    length(out)==size(X, 2) || throw(DimensionMismatch("output length mismatch"))
    out .= vec(sum(abs2, g.W*(X .- g.μ); dims=1))
    return out
end
function Distributions.logpdf(g::ToeplitzMvNormal, x::AbstractVector)
    return -0.5*(length(g)*log(2π)+g.logdeterminant+Distributions.sqmahal(g, x))
end
function Distributions.logpdf(g::ToeplitzMvNormal, X::AbstractMatrix)
    size(X, 1)==length(g) || throw(DimensionMismatch("observation dimension mismatch"))
    return -0.5 .*
           (length(g)*log(2π)+g.logdeterminant .+ vec(sum(abs2, g.W*(X .- g.μ); dims=1)))
end
function Distributions._logpdf!(
    out::AbstractArray{<:Real}, g::ToeplitzMvNormal, X::AbstractMatrix{<:Real}
)
    length(out)==size(X, 2) || throw(DimensionMismatch("output length mismatch"))
    out .= logpdf(g, X)
    return out
end
function Distributions._rand!(
    rng::AbstractRNG, g::ToeplitzMvNormal, x::Union{AbstractVector,AbstractMatrix}
)
    size(x, 1)==length(g) || throw(DimensionMismatch("sample dimension mismatch"))
    randn!(rng, x)
    ldiv!(g.W, x)
    x .+= g.μ
    return x
end
_remean(g::ToeplitzMvNormal, μ) = ToeplitzMvNormal(μ, g.c)
function marginal(g::ToeplitzMvNormal, indices::Union{Vector{Int},AbstractRange})
    idx=_structured_indices(g, indices)
    isempty(idx) && throw(ArgumentError("marginal indices must be nonempty"))
    return MvNormal(g.μ[idx], Symmetric([g.c[abs(i - j) + 1] for i in idx, j in idx]))
end
function predict(
    g::ToeplitzMvNormal,
    x::AbstractVector,
    input::Union{Vector{Int},AbstractRange},
    output::Union{Vector{Int},AbstractRange},
)
    all(isfinite, x) || throw(ArgumentError("observed values must be finite"))
    obs=_structured_indices(g, input)
    target=_structured_indices(g, output)
    length(x)==length(obs) ||
        throw(DimensionMismatch("observed values and indices must match"))
    isempty(target) && throw(ArgumentError("output indices must be nonempty"))
    isempty(intersect(obs, target)) ||
        throw(ArgumentError("input and output indices must be disjoint"))
    isempty(obs) && return marginal(g, target)
    observed=Symmetric([g.c[abs(i - j) + 1] for i in obs, j in obs])
    cross=[g.c[abs(i - j) + 1] for i in target, j in obs]
    target_covariance=[g.c[abs(i - j) + 1] for i in target, j in target]
    factor=cholesky(observed)
    μ=g.μ[target]+cross*(factor \ (x-g.μ[obs]))
    conditional=target_covariance-cross*(factor \ cross')
    return MvNormal(μ, Symmetric(conditional))
end
function predict(
    g::ToeplitzMvNormal,
    x::AbstractVector;
    input_indices=1:length(x),
    output_indices=(length(x) + 1):length(g),
)
    return predict(g, x, collect(input_indices), collect(output_indices))
end

function _check(s::ToeplitzCovariance, m::CovarianceMethod, p)
    return m isa ToeplitzMLE ||
           throw(ArgumentError("ToeplitzCovariance requires ToeplitzMLE"))
end
function initialize(
    s::ToeplitzCovariance,
    m::ToeplitzMLE,
    X::AbstractMatrix;
    weights=nothing,
    rng=Random.default_rng(),
)
    X, w=_data(X, weights)
    μ=_mean(X, w)
    v=sum(_variance(X .- μ, w))/size(X, 1)+m.regularization
    v>0 || throw(ArgumentError("positive data variance or regularization required"))
    c=zeros(size(X, 1))
    c[1]=v
    return GaussianWorkspace(s, ToeplitzMvNormal(μ, c), FitReport())
end
function _validate_model(
    s::ToeplitzCovariance, m::CovarianceMethod, g::Distributions.AbstractMvNormal
)
    _check(s, m, length(g))
    g isa ToeplitzMvNormal ||
        throw(ArgumentError("model does not match ToeplitzCovariance"))
    return nothing
end
function workspace(
    s::ToeplitzCovariance, m::CovarianceMethod, g::Distributions.AbstractMvNormal
)
    _validate_model(s, m, g)
    return GaussianWorkspace(s, deepcopy(g), FitReport())
end

# Durbin gives the first inverse column. The Gohberg–Semencul displacement
# recurrence fills the remaining entries in O(p²), without a dense inversion.
function _toeplitz_precision(c)
    p=length(c)
    variance=c[1]
    isfinite(variance) && variance>0 || throw(PosDefException(1))
    ld=log(variance)
    a=Float64[]
    for m in 1:(p-1)
        reflection=(c[m+1]-dot(a, @view(c[m:-1:2])))/variance
        isfinite(reflection) && abs(reflection)<1 || throw(PosDefException(m+1))
        a=vcat(a-reflection*reverse(a), reflection)
        variance*=1-reflection^2
        isfinite(variance) && variance>0 || throw(PosDefException(m+1))
        ld+=log(variance)
    end
    x=vcat(1.0, -a) ./ variance
    Q=Matrix{Float64}(undef, p, p)
    Q[:,1]=x
    Q[1,:]=x
    for j in 2:p, i in j:p
        Q[i,j]=Q[i-1,j-1]+(x[i]*x[j]-x[p-i+2]*x[p-j+2])/x[1]
        Q[j,i]=Q[i,j]
    end
    return Q, ld
end

function _toeplitz_objective(c, S)
    precision, ld=_toeplitz_precision(c)
    return ld+sum(precision .* S), precision
end

# The empirical scatter is not Toeplitz. Its dense products still cost O(p³),
# but are evaluated only once per accepted iterate, never during backtracking.
function _toeplitz_gradient(precision, S)
    G=precision-precision*S*precision
    return [lag==0 ? tr(G) : 2sum(diag(G, lag)) for lag in 0:(size(S,1)-1)]
end
function _toeplitz_objective_gradient(c, S)
    objective, precision=_toeplitz_objective(c, S)
    return objective, _toeplitz_gradient(precision, S)
end

function _toeplitz_direction(gradient, steps, changes)
    q=copy(gradient)
    alpha=zeros(length(steps))
    for j in length(steps):-1:1
        alpha[j]=dot(steps[j],q)/dot(steps[j],changes[j])
        q .-= alpha[j] .* changes[j]
    end
    if !isempty(steps)
        q .*= dot(last(steps),last(changes))/sum(abs2,last(changes))
    end
    for j in eachindex(steps)
        beta=dot(changes[j],q)/dot(steps[j],changes[j])
        q .+= (alpha[j]-beta) .* steps[j]
    end
    return -q
end
function _covariance(s::ToeplitzCovariance, m::ToeplitzMLE, current, R, w)
    scatter=(R .* w')*R'
    target=scatter+m.regularization*I
    # Scale covariance by average variance to make stopping and step sizes
    # insensitive to a common change in observation units.
    scale=tr(target)/size(R, 1)
    scale>0 || throw(ArgumentError("positive data variance or regularization required"))
    c=current.c ./ scale
    target=target ./ scale
    value, gradient=_toeplitz_objective_gradient(c, target)
    report=FitReport(; kind=if m.regularization==0
        :covariance_loglikelihood
    else
        :penalized_covariance_loglikelihood
    end)
    offset=length(c)*(log(2π)+log(scale))
    push!(report.history, -0.5*(offset+value))
    steps=Vector{Float64}[]
    changes=Vector{Float64}[]
    for iteration in 1:m.maxiter
        if m.tol>0 && norm(gradient, Inf)<=m.tol
            report.status=:converged
            break
        end
        direction=_toeplitz_direction(gradient, steps, changes)
        slope=dot(gradient,direction)
        if !isfinite(slope) || slope>=0
            empty!(steps)
            empty!(changes)
            direction=-gradient
            slope=-sum(abs2,gradient)
        end
        step=1.0
        accepted=false
        for backtrack in 1:m.max_backtracks
            candidate=c+step*direction
            try
                trial, trial_precision=_toeplitz_objective(candidate, target)
                if isfinite(trial) &&
                    trial<=value+1e-4*step*slope
                    trial_gradient=_toeplitz_gradient(trial_precision,target)
                    all(isfinite,trial_gradient) || break
                    displacement=candidate-c
                    change=trial_gradient-gradient
                    if dot(displacement,change)>1e-10*norm(displacement)*norm(change)
                        push!(steps,displacement)
                        push!(changes,change)
                        if length(steps)>m.memory
                            popfirst!(steps)
                            popfirst!(changes)
                        end
                    end
                    c=candidate
                    value=trial
                    gradient=trial_gradient
                    accepted=true
                    break
                end
            catch err
                err isa PosDefException || rethrow()
            end
            step*=0.5
        end
        if !accepted
            report.status=:failed
            report.message="Toeplitz likelihood line search exhausted without an SPD descent step"
            break
        end
        report.iterations=iteration
        push!(report.history, -0.5*(offset+value))
    end
    if report.status==:initialized
        report.status=m.tol>0 && norm(gradient, Inf)<=m.tol ? :converged : :iteration_limit
    end
    report.objective=last(report.history)
    g=ToeplitzMvNormal(zeros(length(c)), c .* scale)
    report.observed_objective=_objective(g, R, w)
    return g, report
end
