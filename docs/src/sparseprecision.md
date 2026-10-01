# Sparse precision Gaussians

`SparsePrecisionMvNormal(μ, Q)` stores a sparse precision matrix and its sparse
Cholesky factor. It needs only Julia's `SparseArrays` standard library beyond the
base package dependencies. `Distributions.invcov(g)` returns a copy; `cov(g)` explicitly
materializes a dense covariance. Scoring uses sparse matrix products and a cached
log determinant. Sampling uses the permuted sparse triangular factor.

```@example sparse
using StructuredGaussianMixtures, SparseArrays, Distributions, Random
Q = spdiagm(-1 => fill(-0.2, 4), 0 => fill(2.0, 5), 1 => fill(-0.2, 4))
g = SparsePrecisionMvNormal(zeros(5), Q)
logpdf(g, zeros(5))
posterior = predict(g, [0.5], [1], [2, 3, 4, 5])
Distributions.invcov(posterior)
```

Conditioning on coordinates and retaining all remaining coordinates preserves a
sparse precision representation: its precision is the corresponding principal
block. If additional coordinates are marginalized, selected covariance solves
produce a dense `MvNormal`. Arbitrary marginalization can create fill-in; we do
not promise a sparse result. Selected solves require a dimension × selected-count
workspace, not a full covariance unless every coordinate is selected.

## Optional graphical lasso fitting

Install `Convex` and a conic solver separately. They are **not base dependencies**.
Loading Convex activates the package extension (Julia 1.9 or newer); no explicit
extension import is needed. The following example requires optional packages and
is not run by the base documentation build:

```julia
using StructuredGaussianMixtures, Convex, SCS, Random
optimizer = Convex.MOI.OptimizerWithAttributes(
    SCS.Optimizer, "eps_abs" => 1e-7, "eps_rel" => 1e-7, "max_iters" => 100000,
)
method = GraphicalLasso(penalty=0.1, optimizer=optimizer, zero_tol=1e-5)
X = randn(MersenneTwister(4), 5, 200)
result = fit(SparsePrecision(), method, X; report=true)
state = workspace(SparsePrecision(), method, result.model)
fit!(state, method, X)  # supplies the current precision as a primal warm start
mixture = fit(MixtureSpec(SparsePrecision(), 2),
              EM(covariance_method=method, maxiter=10), X)
```

The method minimizes

```math
\operatorname{tr}((S+\eta I)Q)-\log\det Q
+\lambda\sum_{i\ne j}|Q_{ij}|,
```

where `S` is normalized weighted centered scatter, `η = regularization` and
`λ = penalty`. Both symmetric off-diagonal entries count. Diagonals are not L1
penalized. Positive ridge regularization helps when scatter is singular. This
Convex backend forms dense scatter and a conic optimization problem; it is a
reference backend for modest dimensions, not a large-scale graphical-lasso solver.

The optimizer must support semidefinite and exponential cones. Solver tolerances
and iteration limits belong in the supplied optimizer. Only an `OPTIMAL` solver
status is accepted. The returned precision must be positive definite and pass an
independent KKT residual check. Off-diagonal magnitudes at most `zero_tol` are
zeroed before checking; a large threshold can invalidate the solution and is
rejected. Tighten solver tolerances if residual checks fail.

Single-Gaussian reports label their objective `:penalized_covariance_loglikelihood`:
minus half the above expression, including the Gaussian normalization constant.
`observed_objective` holds the unpenalized weighted log likelihood. `iterations`
counts solver calls (one per covariance fit); the message contains solver status
and KKT residual. Warm starts pass current precision values; solver-specific dual
state is not retained, and speedup depends on solver support.

Mixture reports retain **observed log likelihood**, with a message explaining that
penalized M-steps need not improve it monotonically. Outer convergence/restart
selection still use that observed objective; this is a regularized fitting
heuristic, not a claim to optimize one fixed globally penalized mixture objective.
Tied sparse precision fitting is not supported yet.

```@docs
SparsePrecisionMvNormal
SparsePrecision
GraphicalLasso
Distributions.invcov
marginal(::SparsePrecisionMvNormal, ::Union{Vector{Int},AbstractRange})
predict(::SparsePrecisionMvNormal, ::AbstractVector, ::Union{Vector{Int},AbstractRange}, ::Union{Vector{Int},AbstractRange})
```
