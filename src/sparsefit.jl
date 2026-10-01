"""Sparse positive-definite precision, with sparsity learned by the covariance method."""
struct SparsePrecision <: GaussianStructure end
"""
    GraphicalLasso(; penalty=0.1, regularization=1e-6, optimizer=nothing,
                    kkt_tol=1e-3, zero_tol=1e-4, warmstart=true)

Minimize `tr((S + regularization*I)*Q) - logdet(Q) + penalty*sum(abs,Q_offdiag)`.
Both symmetric off-diagonal entries count in the penalty. Load `Convex` and pass
an optimizer supporting semidefinite and exponential cones (e.g. `SCS.Optimizer`).
Configure solver tolerances/iteration limits in that optimizer. `zero_tol` drops
small off-diagonal entries; the resulting SPD matrix must pass the KKT check.
"""
struct GraphicalLasso{O} <: CovarianceMethod
    penalty::Float64
    regularization::Float64
    optimizer::O
    kkt_tol::Float64
    zero_tol::Float64
    warmstart::Bool
    function GraphicalLasso(;
        penalty=0.1,
        regularization=1e-6,
        optimizer=nothing,
        kkt_tol=1e-3,
        zero_tol=1e-4,
        warmstart=true,
    )
        all(x->isfinite(x)&&x>=0, (penalty, regularization, zero_tol)) || throw(
            ArgumentError(
                "penalty, regularization and zero_tol must be finite and nonnegative"
            ),
        )
        isfinite(kkt_tol) && kkt_tol>0 ||
            throw(ArgumentError("kkt_tol must be positive and finite"))
        return new{typeof(optimizer)}(
            penalty, regularization, optimizer, kkt_tol, zero_tol, warmstart
        )
    end
end
_glasso_backend(m) = false
function _check(::SparsePrecision, m::CovarianceMethod, p)
    m isa GraphicalLasso || throw(ArgumentError("SparsePrecision requires GraphicalLasso"))
    _glasso_backend(m) || throw(
        ArgumentError(
            "GraphicalLasso requires the optional Convex extension: load `using Convex` and supply an optimizer such as SCS.Optimizer",
        ),
    )
    m.optimizer===nothing && throw(
        ArgumentError(
            "GraphicalLasso requires an optimizer supporting semidefinite and exponential cones",
        ),
    )
    return nothing
end
_remean(g::SparsePrecisionMvNormal, μ) = SparsePrecisionMvNormal(μ, g.Q)
function _validate_model(s::SparsePrecision, m::CovarianceMethod, g)
    _check(s, m, length(g))
    g isa SparsePrecisionMvNormal ||
        throw(ArgumentError("model does not match SparsePrecision"))
    return nothing
end
function workspace(
    s::SparsePrecision, m::CovarianceMethod, g::Distributions.AbstractMvNormal
)
    _validate_model(s, m, g)
    return GaussianWorkspace(s, deepcopy(g), FitReport())
end
function initialize(
    s::SparsePrecision,
    m::CovarianceMethod,
    X::AbstractMatrix;
    weights=nothing,
    rng=Random.default_rng(),
)
    X, w=_data(X, weights)
    _check(s, m, size(X, 1))
    μ=_mean(X, w)
    v=_variance(X .- μ, w) .+ m.regularization
    all(>(0), v) || throw(ArgumentError("zero variance: use positive regularization"))
    return GaussianWorkspace(s, SparsePrecisionMvNormal(μ, spdiagm(0=>1 ./ v)), FitReport())
end
function _covariance(s::SparsePrecision, m::GraphicalLasso, current, R, w)
    S=Symmetric((R .* w')*R')
    return _glasso_solve(m, current, S)
end
function _glasso_solve(m, current, S)
    return throw(ArgumentError("Load `using Convex` to enable GraphicalLasso fitting"))
end
function _glasso_kkt(Q, S, λ)
    gradient=Matrix(S)-inv(Symmetric(Matrix(Q)))
    residual=0.0
    for j in axes(Q, 2), i in axes(Q, 1)
        value=if i==j
            abs(gradient[i, j])
        elseif Q[i, j]==0
            max(abs(gradient[i, j])-λ, 0)
        else
            abs(gradient[i, j]+λ*sign(Q[i, j]))
        end
        residual=max(residual, value)
    end
    return residual
end

# The generic driver still reports observed likelihood for comparison with other
# mixtures; penalized M-steps do not guarantee its monotonic improvement.
function fit!(
    state::MixtureWorkspace{MixtureSpec{SparsePrecision}},
    m::EM,
    X::AbstractMatrix;
    weights=nothing,
)
    result=invoke(fit!, Tuple{MixtureWorkspace,EM,AbstractMatrix}, state, m, X; weights)
    note="GraphicalLasso uses penalized covariance updates; observed likelihood need not increase"
    state.report.message=if isempty(state.report.message)
        note
    else
        state.report.message*"; "*note
    end
    return result
end
