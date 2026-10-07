"""Precision matrix with half-bandwidth `bandwidth` in the supplied feature order."""
struct BandedPrecision <: GaussianStructure
    bandwidth::Int
    function BandedPrecision(bandwidth::Integer)
        bandwidth >= 0 || throw(ArgumentError("bandwidth must be nonnegative"))
        return new(bandwidth)
    end
end

"""
    BandedPrecisionMvNormal(mean, precision; bandwidth)

Gaussian storing the lower precision bands and their Cholesky factor in O(p*b)
space. `bandwidth` is the half-bandwidth; entries outside it must be zero.
Inputs are copied. `cov` and `invcov` explicitly materialize dense matrices;
scoring, sampling and precision solves use the compact factors.
"""
struct BandedPrecisionMvNormal <: Distributions.AbstractMvNormal
    μ::Vector{Float64}
    bands::Matrix{Float64}
    factor::Matrix{Float64}
    logdet_precision::Float64
    function BandedPrecisionMvNormal(
        μ::AbstractVector, bands::Matrix{Float64}, ::Val{:bands}
    )
        p=length(μ)
        p>0 && size(bands, 1)==p && 1<=size(bands, 2)<=p ||
            throw(DimensionMismatch("invalid precision bands"))
        all(isfinite, μ) && all(isfinite, bands) ||
            throw(ArgumentError("parameters must be finite"))
        B=copy(bands)
        b=size(B, 2)-1
        L=zeros(p, b+1)
        for i in 1:p
            for j in max(1, i - b):i
                value=B[i, i - j + 1]
                for k in max(1, i - b, j - b):(j - 1)
                    value-=L[i, i - k + 1]*L[j, j - k + 1]
                end
                if i==j
                    value>0 || throw(PosDefException(i))
                    L[i, 1]=sqrt(value)
                else
                    L[i, i - j + 1]=value/L[j, 1]
                end
            end
        end
        return new(Vector{Float64}(μ), B, L, 2sum(log, view(L, :, 1)))
    end
end
function BandedPrecisionMvNormal(μ::AbstractVector, Q::AbstractMatrix; bandwidth::Integer)
    p=length(μ)
    size(Q)==(p, p) || throw(DimensionMismatch("precision and mean dimensions differ"))
    0<=bandwidth<p || throw(ArgumentError("bandwidth must be between 0 and p-1"))
    issymmetric(Q) || throw(ArgumentError("precision must be symmetric"))
    all(isfinite, Q) || throw(ArgumentError("precision must be finite"))
    B=zeros(p, bandwidth+1)
    for i in 1:p, j in 1:i
        if i-j<=bandwidth
            B[i, i - j + 1]=Q[i, j]
        elseif !iszero(Q[i, j])
            throw(ArgumentError("nonzero precision outside declared band"))
        end
    end
    return BandedPrecisionMvNormal(μ, B, Val(:bands))
end
Distributions.length(g::BandedPrecisionMvNormal) = length(g.μ)
Distributions.size(g::BandedPrecisionMvNormal) = (length(g),)
Distributions.mean(g::BandedPrecisionMvNormal) = copy(g.μ)
Distributions.params(g::BandedPrecisionMvNormal) = (mean(g), invcov(g))
Distributions.logdetcov(g::BandedPrecisionMvNormal) = -g.logdet_precision
Base.eltype(::Type{BandedPrecisionMvNormal}) = Float64
Base.eltype(::BandedPrecisionMvNormal) = Float64
function _bandentry(B, i, j)
    i<j && return _bandentry(B, j, i)
    return i-j<size(B, 2) ? B[i, i - j + 1] : 0.0
end
function Distributions.invcov(g::BandedPrecisionMvNormal)
    p=length(g)
    return [_bandentry(g.bands, i, j) for i in 1:p, j in 1:p]
end
# Solve Q*x=v through two band triangular substitutions.
function _bandsolve(g::BandedPrecisionMvNormal, v::AbstractVector)
    length(v)==length(g) || throw(DimensionMismatch("precision solve dimension mismatch"))
    x=Vector{Float64}(v)
    L=g.factor
    p=length(g)
    b=size(L, 2)-1
    for i in 1:p
        for j in max(1, i - b):(i - 1)
            x[i]-=L[i, i - j + 1]*x[j]
        end
        x[i]/=L[i, 1]
    end
    for i in p:-1:1
        for j in (i + 1):min(p, i + b)
            x[i]-=L[j, j - i + 1]*x[j]
        end
        x[i]/=L[i, 1]
    end
    return x
end
function _bandcolumns(g, ids)
    C=Matrix{Float64}(undef, length(g), length(ids))
    e=zeros(length(g))
    for (j, i) in enumerate(ids)
        fill!(e, 0)
        e[i]=1
        C[:, j]=_bandsolve(g, e)
    end
    return C
end
function Distributions.cov(g::BandedPrecisionMvNormal)
    return Matrix(Symmetric(_bandcolumns(g, 1:length(g))))
end
Distributions.var(g::BandedPrecisionMvNormal) = diag(cov(g))
function Distributions.sqmahal(g::BandedPrecisionMvNormal, x::AbstractVector)
    length(x)==length(g) || throw(DimensionMismatch("observation dimension mismatch"))
    L=g.factor
    p=length(g)
    b=size(L, 2)-1
    value=0.0
    for j in 1:p
        v=0.0
        for i in j:min(p, j + b)
            v+=L[i, i - j + 1]*(x[i]-g.μ[i])
        end
        value+=v*v
    end
    return value
end
function Distributions.logpdf(g::BandedPrecisionMvNormal, x::AbstractVector)
    return -0.5*(length(g)*log(2π)+logdetcov(g)+sqmahal(g, x))
end
function Distributions._logpdf!(
    out::AbstractVector{<:Real}, g::BandedPrecisionMvNormal, X::AbstractMatrix{<:Real}
)
    size(X, 1)==length(g) && length(out)==size(X, 2) ||
        throw(DimensionMismatch("score dimensions differ"))
    for j in axes(X, 2)
        out[j]=logpdf(g, view(X, :, j))
    end
    return out
end
function Distributions._rand!(
    rng::AbstractRNG, g::BandedPrecisionMvNormal, x::AbstractVector
)
    length(x)==length(g) || throw(DimensionMismatch("sample dimension mismatch"))
    randn!(rng, x)
    p=length(g)
    L=g.factor
    b=size(L, 2)-1
    for i in p:-1:1
        for j in (i + 1):min(p, i + b)
            x[i]-=L[j, j - i + 1]*x[j]
        end
        x[i]/=L[i, 1]
    end
    return x .+= g.μ
end
function Distributions._rand!(
    rng::AbstractRNG, g::BandedPrecisionMvNormal, X::AbstractMatrix
)
    for j in axes(X, 2)
        Distributions._rand!(rng, g, view(X, :, j))
    end
    return X
end
_remean(g::BandedPrecisionMvNormal, μ) = BandedPrecisionMvNormal(μ, g.bands, Val(:bands))
function _check(s::BandedPrecision, m::CovarianceMethod, p)
    m isa Exact || throw(ArgumentError("banded precision requires Exact"))
    return s.bandwidth<p || throw(ArgumentError("bandwidth must be smaller than dimension"))
end
# Decomposable band graph: weighted regressions on each variable's preceding b
# neighbors give the exact constrained MLE. Ridge means replacing S by S+λI.
function _covariance(s::BandedPrecision, m::Exact, current, R, w)
    p=size(R, 1)
    b=s.bandwidth
    S=zeros(p, b+1)
    for i in 1:p, delta in 0:min(b, i - 1)
        S[i, delta + 1]=dot(view(R, i, :), w .* view(R, i-delta, :))+(
            delta==0 ? m.regularization : 0
        )
    end
    Q=zeros(p, b+1)
    for i in 1:p
        ids=max(1, i - b):(i - 1)
        c=[_bandentry(S, i, j) for j in ids]
        localS=[_bandentry(S, j, k) for j in ids, k in ids]
        beta=isempty(ids) ? Float64[] : cholesky(Symmetric(localS))\c
        residual=S[i, 1]-dot(c, beta)
        residual>0 || throw(PosDefException(i))
        indices=[collect(ids); i]
        coefficients=[-beta; 1.0]
        for a in eachindex(indices), d in 1:a
            row=indices[a]
            col=indices[d]
            Q[row, row - col + 1]+=coefficients[a]*coefficients[d]/residual
        end
    end
    g=BandedPrecisionMvNormal(zeros(p), Q, Val(:bands))
    report=FitReport(; kind=:covariance_loglikelihood)
    report.status=:converged
    report.iterations=1
    report.objective=_objective(g, R, w)
    push!(report.history, report.objective)
    return g, report
end
function workspace(
    s::BandedPrecision, m::CovarianceMethod, g::Distributions.AbstractMvNormal
)
    _validate_model(s, m, g)
    return GaussianWorkspace(s, deepcopy(g), FitReport())
end
function _validate_model(s::BandedPrecision, m::CovarianceMethod, g)
    _check(s, m, length(g))
    g isa BandedPrecisionMvNormal && size(g.bands, 2)==s.bandwidth+1 ||
        throw(ArgumentError("model does not match banded precision specification"))
    return nothing
end
function _bandindices(g, ids)
    ids=collect(Int, ids)
    all(i->1<=i<=length(g), ids) && length(unique(ids))==length(ids) ||
        throw(ArgumentError("indices must be unique and in bounds"))
    return ids
end
function marginal(g::BandedPrecisionMvNormal, ids::Union{Vector{Int},AbstractRange})
    ids=_bandindices(g, ids)
    isempty(ids) && throw(ArgumentError("marginal indices must be nonempty"))
    C=_bandcolumns(g, ids)
    return MvNormal(g.μ[ids], Symmetric(C[ids, :]))
end
function predict(
    g::BandedPrecisionMvNormal,
    x::AbstractVector,
    inputs::Union{Vector{Int},AbstractRange},
    outputs::Union{Vector{Int},AbstractRange},
)
    inputs=_bandindices(g, inputs)
    outputs=_bandindices(g, outputs)
    length(x)==length(inputs) || throw(DimensionMismatch("observations and indices differ"))
    isempty(outputs) && throw(ArgumentError("outputs must be nonempty"))
    isempty(intersect(inputs, outputs)) ||
        throw(ArgumentError("inputs and outputs overlap"))
    all(isfinite, x) || throw(ArgumentError("observations must be finite"))
    if length(inputs)+length(outputs)==length(g)
        # With every omitted variable observed, conditional precision is a
        # principal submatrix; no covariance or Schur complement is needed.
        positions=zeros(Int, length(g))
        positions[outputs]=1:length(outputs)
        entries=Tuple{Int,Int,Float64}[]
        b=0
        # Visit only stored edges, including diagonal entries.
        for oldi in 1:length(g), delta in 0:min(size(g.bands, 2) - 1, oldi - 1)
            oldj=oldi-delta
            i=positions[oldi]
            j=positions[oldj]
            if i>0 && j>0
                a=max(i, j)
                c=min(i, j)
                value=g.bands[oldi, delta + 1]
                if !iszero(value)
                    b=max(b, a-c)
                    push!(entries, (a, c, value))
                end
            end
        end
        B=zeros(length(outputs), b+1)
        for (i, j, value) in entries
            B[i, i - j + 1]=value
        end
        observed=zeros(length(g))
        observed[inputs]=x-g.μ[inputs]
        product=zeros(length(g))
        for i in 1:length(g), delta in 0:min(size(g.bands, 2) - 1, i - 1)
            j=i-delta
            value=g.bands[i, delta + 1]
            product[i]+=value*observed[j]
            i!=j && (product[j]+=value*observed[i])
        end
        rhs=product[outputs]
        conditional=BandedPrecisionMvNormal(g.μ[outputs], B, Val(:bands))
        return _remean(conditional, g.μ[outputs]-_bandsolve(conditional, rhs))
    end
    isempty(inputs) && return marginal(g, outputs)
    ids=[inputs; outputs]
    C=_bandcolumns(g, ids)[ids, :]
    n=length(inputs)
    Cii=cholesky(Symmetric(C[1:n, 1:n]))
    Coi=C[(n + 1):end, 1:n]
    return MvNormal(
        g.μ[outputs]+Coi*(Cii\(x-g.μ[inputs])),
        Symmetric(C[(n + 1):end, (n + 1):end]-Coi*(Cii\Coi')),
    )
end
function predict(
    g::BandedPrecisionMvNormal,
    x::AbstractVector;
    input_indices=1:length(x),
    output_indices=(length(x) + 1):length(g),
)
    return predict(g, x, collect(input_indices), collect(output_indices))
end

Distributions.partype(::BandedPrecisionMvNormal) = Float64
function Distributions.sqmahal!(
    out::AbstractArray{<:Real}, g::BandedPrecisionMvNormal, X::AbstractMatrix{<:Real}
)
    size(X, 1)==length(g) && length(out)==size(X, 2) ||
        throw(DimensionMismatch("quadratic form dimensions differ"))
    for j in axes(X, 2)
        out[j]=sqmahal(g, view(X, :, j))
    end
    return out
end
