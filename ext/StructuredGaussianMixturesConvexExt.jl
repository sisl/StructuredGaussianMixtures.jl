__precompile__(false)

module StructuredGaussianMixturesConvexExt
using StructuredGaussianMixtures
using Convex
using LinearAlgebra
using SparseArrays
const SGM=StructuredGaussianMixtures
SGM._glasso_backend(::SGM.GraphicalLasso) = true
function SGM._glasso_solve(m::SGM.GraphicalLasso, current, S)
    p=size(S, 1)
    scatter=Matrix(S)+m.regularization*I
    Q=Convex.Semidefinite(p)
    mask=ones(p, p)-Matrix{Float64}(I, p, p)
    problem=Convex.minimize(
        -Convex.logdet(Q)+sum(scatter .* Q)+m.penalty*sum(abs(mask .* Q))
    )
    if m.warmstart && current!==nothing
        Convex.set_value!(Q, Matrix(current.Q))
    end
    Convex.solve!(
        problem, m.optimizer; silent=true, warmstart=m.warmstart && current!==nothing
    )
    problem.status==Convex.MOI.OPTIMAL ||
        throw(ArgumentError("GraphicalLasso solver did not converge: $(problem.status)"))
    estimate=Convex.evaluate(Q)
    estimate===nothing &&
        throw(ArgumentError("GraphicalLasso solver returned no precision"))
    estimate=Matrix(Symmetric((estimate+estimate')/2))
    for j in 1:p, i in 1:p
        i!=j && abs(estimate[i, j])<=m.zero_tol && (estimate[i, j]=0)
    end
    # Never silently repair an indefinite solution or accept thresholding that
    # invalidates the optimizer's result. Verify the matrix actually returned.
    model=SGM.SparsePrecisionMvNormal(zeros(p), sparse(estimate))
    residual=SGM._glasso_kkt(estimate, scatter, m.penalty)
    residual<=m.kkt_tol*(1+maximum(abs, scatter)) || throw(
        ArgumentError(
            "GraphicalLasso KKT residual $residual exceeds tolerance; tighten solver tolerances or adjust zero_tol",
        ),
    )
    penalty=sum(abs, estimate)-sum(abs, diag(estimate))
    report=SGM.FitReport(; kind=:penalized_covariance_loglikelihood)
    report.status=:converged
    report.iterations=1
    report.objective=-0.5*(
        p*log(2π)-logdet(model.factor)+sum(scatter .* estimate)+m.penalty*penalty
    )
    report.observed_objective=-0.5*(p*log(2π)-logdet(model.factor)+sum(S .* estimate))
    push!(report.history, report.objective)
    report.message="Convex status=$(problem.status); KKT residual=$residual; iterations counts solver calls"
    return model, report
end
end
