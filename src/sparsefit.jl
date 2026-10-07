"""Sparse positive-definite precision, with sparsity learned by the covariance method."""
struct SparsePrecision <: GaussianStructure end
"""
    GraphicalLasso(; penalty=0.1, regularization=1e-6, optimizer=nothing,
                    kkt_tol=1e-3, zero_tol=1e-4, warmstart=true,
                    maxiter=100, inner_maxiter=1000)

Minimize `tr((S + regularization*I)*Q) - logdet(Q) + penalty*sum(abs,Q_offdiag)`.
Both symmetric off-diagonal entries count in the penalty; diagonals are unpenalized.
The default native block-coordinate solver has `maxiter=100` full sweeps and
`inner_maxiter=1000` coordinate sweeps per lasso block. `kkt_tol` controls the
final scaled KKT residual; exhaustion returns `:iteration_limit`. Warm starts
reuse a feasible covariance derived from the supplied current model.
Load `Convex` and pass `optimizer=...` to use the optional conic reference backend.
`zero_tol` drops small precision entries subject to SPD and KKT checks.
"""
struct GraphicalLasso{O} <: CovarianceMethod
    penalty::Float64
    regularization::Float64
    optimizer::O
    kkt_tol::Float64
    zero_tol::Float64
    maxiter::Int
    inner_maxiter::Int
    warmstart::Bool
    function GraphicalLasso(;
        penalty=0.1,
        regularization=1e-6,
        optimizer=nothing,
        kkt_tol=1e-3,
        zero_tol=1e-4,
        warmstart=true,
        maxiter=100,
        inner_maxiter=1000,
    )
        all(x->isfinite(x)&&x>=0, (penalty, regularization, zero_tol)) || throw(
            ArgumentError(
                "penalty, regularization and zero_tol must be finite and nonnegative"
            ),
        )
        isfinite(kkt_tol) && kkt_tol>0 ||
            throw(ArgumentError("kkt_tol must be positive and finite"))
        maxiter > 0 && inner_maxiter > 0 || throw(ArgumentError("iteration limits must be positive"))
        return new{typeof(optimizer)}(
            penalty, regularization, optimizer, kkt_tol, zero_tol, maxiter, inner_maxiter, warmstart
        )
    end
end
_glasso_backend(m) = false
function _check(::SparsePrecision, m::CovarianceMethod, p)
    m isa GraphicalLasso || throw(ArgumentError("SparsePrecision requires GraphicalLasso"))
    if m.optimizer !== nothing && !_glasso_backend(m)
        throw(ArgumentError("Explicit optimizer requires the optional Convex extension: load `using Convex`"))
    end
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
function _glasso_solve(m::GraphicalLasso, current, S)
    m.optimizer === nothing && return _glasso_native(m, current, S)
    return _glasso_external(m, current, S)
end
_glasso_external(m, current, S) = throw(ArgumentError("Load `using Convex` for an explicit optimizer"))

# Dual block-coordinate graphical lasso. W is the covariance dual variable,
# diag(W)=diag(scatter), |W-scatter| <= lambda off diagonal. Each block update
# solves a lasso and preserves positive definiteness (up to roundoff).
# Thresholded-scatter components exactly partition the graphical-lasso solution.
# In particular, irrelevant low-variance coordinates never enter a dense solve.
function _glasso_components(S, penalty)
    remaining=trues(size(S,1))
    groups=Vector{Int}[]
    for start in axes(S,1)
        remaining[start] || continue
        group=[start]; remaining[start]=false
        cursor=1
        while cursor<=length(group)
            j=group[cursor]
            for i in axes(S,1)
                if remaining[i] && abs(S[i,j])>penalty
                    push!(group,i); remaining[i]=false
                end
            end
            cursor+=1
        end
        push!(groups,sort!(group))
    end
    return groups
end
function _glasso_native(m, current, S)
    scatter=Matrix(S)+m.regularization*I
    threshold=m.kkt_tol*(1+maximum(abs,scatter))
    warmQ=m.warmstart && current!==nothing ? Matrix(current.Q) : nothing
    warmres=warmQ===nothing ? Inf : _glasso_kkt(warmQ,scatter,m.penalty)
    if m.penalty==0 || warmres<=threshold
        Q=m.penalty==0 ? inv(cholesky(Symmetric(scatter))) : warmQ
        model=SparsePrecisionMvNormal(zeros(size(S,1)),sparse(Q))
        report=FitReport(;kind=:penalized_covariance_loglikelihood)
        residual=m.penalty==0 ? _glasso_kkt(Q,scatter,0) : warmres
        report.status=residual<=threshold ? :converged : :iteration_limit
        report.iterations=m.penalty==0 ? 1 : 0
        penalty=m.penalty*(sum(abs,Q)-sum(abs,diag(Q)))
        report.objective=-0.5*(size(S,1)*log(2π)-logdet(model.factor)+sum(scatter .* Q)+penalty)
        report.observed_objective=-0.5*(size(S,1)*log(2π)-logdet(model.factor)+sum(S .* Q))
        push!(report.history,report.objective)
        source=m.penalty==0 ? "zero-penalty direct precision solve" : "warm precision already satisfies KKT"
        report.message="$source; KKT residual=$residual; tolerance=$threshold"
        return model,report
    end
    groups=_glasso_components(S,m.penalty)
    length(groups)==1 && return _glasso_native_core(m,current,S)
    p=size(S,1)
    Q=zeros(p,p)
    reports=FitReport[]
    for group in groups
        initial=current===nothing ? nothing : SparsePrecisionMvNormal(zeros(length(group)),current.Q[group,group])
        model, inner=_glasso_native_core(m,initial,S[group,group])
        Q[group,group] .= Matrix(model.Q)
        push!(reports,inner)
    end
    model=SparsePrecisionMvNormal(zeros(p),sparse(Q))
    report=FitReport(;kind=:penalized_covariance_loglikelihood)
    report.iterations=maximum(r.iterations for r in reports)
    report.status=all(r.status==:converged for r in reports) ? :converged : :iteration_limit
    report.history=[sum(r.history[min(i,length(r.history))] for r in reports) for i in 1:report.iterations]
    report.objective=sum(r.objective for r in reports)
    report.observed_objective=sum(r.observed_objective for r in reports)
    residual=_glasso_kkt(Q,Matrix(S)+m.regularization*I,m.penalty)
    threshold=m.kkt_tol*(1+maximum(abs,Matrix(S)+m.regularization*I))
    report.status==:converged && residual>threshold && (report.status=:iteration_limit)
    report.message="native block-coordinate graphical lasso; screened blocks=$(length(groups)), largest=$(maximum(length,groups)); KKT residual=$residual; tolerance=$threshold; largest block diagnostics: "*reports[argmax(length.(groups))].message
    return model,report
end
function _glasso_native_core(m, current, S)
    p=size(S, 1)
    scatter=Matrix(S)+m.regularization*I
    all(>(0), diag(scatter)) || throw(ArgumentError("zero variance: use positive regularization"))
    W=copy(scatter)
    # A diagonal-loading convex combination is strictly feasible even for a
    # singular scatter, provided lambda>0. Its diagonal remains unpenalized.
    offmax=maximum(abs, scatter-Diagonal(diag(scatter)))
    alpha=offmax==0 ? 1.0 : min(1.0, m.penalty/offmax)
    W .*= 1-alpha
    W[diagind(W)] .= diag(scatter)
    if m.warmstart && current !== nothing
        candidate=Matrix(cov(current))
        candidate .= clamp.(candidate, scatter .- m.penalty, scatter .+ m.penalty)
        candidate[diagind(candidate)] .= diag(scatter)
        isposdef(Symmetric(candidate)) && (W=candidate)
    end
    isposdef(Symmetric(W)) || throw(ArgumentError("singular scatter with zero penalty: use positive regularization"))
    report=FitReport(; kind=:penalized_covariance_loglikelihood)
    beta=zeros(p-1); residual=zeros(p-1); block=zeros(p-1,p-1)
    # Retain the solved lasso coefficients, including their exact zeros. An
    # inverse covariance reconstruction each sweep introduces dense roundoff
    # entries and discards the block solver's own warm-start information.
    coefficients=zeros(p,p)
    if m.warmstart && current!==nothing
        for j in 1:p
            coefficients[:,j] .= -Vector(current.Q[:,j]) ./ current.Q[j,j]
            coefficients[j,j]=0
        end
    end
    threshold=m.kkt_tol*(1+maximum(abs, scatter))
    Q=inv(Symmetric(W))
    finalres=Inf
    coordinate_sweeps=0
    for sweep in 1:m.maxiter
        for j in 1:p
            idx=[1:j-1; j+1:p]
            copyto!(block, view(W, idx, idx))
            beta .= view(coefficients,idx,j)
            mul!(residual, block, beta)
            residual .= view(scatter, idx, j) .- residual
            for inner in 1:m.inner_maxiter
                coordinate_sweeps+=1
                delta=0.0
                for i in eachindex(beta)
                    old=beta[i]
                    z=residual[i]+block[i,i]*old
                    new=sign(z)*max(abs(z)-m.penalty, 0)/block[i,i]
                    change=new-old
                    if change!=0
                        beta[i]=new
                        BLAS.axpy!(-change, view(block,:,i), residual)
                        delta=max(delta, abs(change)*block[i,i])
                    end
                end
                delta <= threshold*0.01 && break
            end
            coefficients[idx,j] .= beta
            mul!(residual, block, beta)
            # An prematurely stopped lasso can violate the Schur complement.
            # Preserve the last SPD iterate rather than publishing that update.
            if dot(beta, residual) < W[j,j]
                W[idx,j] .= residual
                W[j,idx] .= residual
            end
        end
        Q=Matrix(inv(Symmetric(W)))
        # Numerical inversion leaves tiny entries at the lasso zeros. Only
        # accept thresholding when both SPD and KKT hold. An aggressive
        # zero_tol must not destroy a valid unthresholded solution.
        candidate=copy(Q)
        for j in 1:p, i in 1:p
            i!=j && (abs(candidate[i,j])<=m.zero_tol ||
                (coefficients[i,j]==0 && coefficients[j,i]==0)) && (candidate[i,j]=0)
        end
        candidate_residual=isposdef(Symmetric(candidate)) ?
            _glasso_kkt(candidate, scatter, m.penalty) : Inf
        if candidate_residual<=threshold
            Q=candidate
            finalres=candidate_residual
        else
            finalres=_glasso_kkt(Q, scatter, m.penalty)
        end
        objective=-0.5*(p*log(2π)-logdet(Symmetric(Q))+sum(scatter .* Q)+m.penalty*(sum(abs,Q)-sum(abs,diag(Q))))
        push!(report.history, objective)
        report.iterations=sweep
        report.objective=objective
        if finalres<=threshold
            report.status=:converged
            break
        end
        report.status=:iteration_limit
    end
    model=SparsePrecisionMvNormal(zeros(p), sparse(Q))
    report.observed_objective=-0.5*(p*log(2π)-logdet(model.factor)+sum(S .* Q))
    report.message="native block-coordinate graphical lasso; coordinate sweeps=$coordinate_sweeps; KKT residual=$finalres; tolerance=$threshold"
    return model, report
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
