# Sparse precision Gaussians

`SparsePrecisionMvNormal(μ, Q)` stores a sparse precision matrix and its sparse
Cholesky factor. It needs only Julia's `SparseArrays` standard library beyond the
base package dependencies. `Distributions.invcov(g)` returns a copy; `cov(g)` explicitly
materializes a dense covariance. Scoring uses sparse matrix products and a cached
log determinant. `Distributions.sqmahal(g, X)` and `sqmahal!(out, g, X)` expose
the same batched quadratic-form kernel without computing log densities. Sampling uses the permuted sparse triangular factor.

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

## Graphical lasso fitting

The default solver is native dual block-coordinate graphical lasso. It requires
no additional dependencies. It follows the block updates of
[Friedman, Hastie, and Tibshirani](https://pmc.ncbi.nlm.nih.gov/articles/PMC3019769/),
with unpenalized diagonals, and uses dense working matrices while learning a
sparse precision. Storage is quadratic; runtime depends on sparsity, conditioning,
and the number of block/lasso sweeps. It does not construct a semidefinite program.

```julia
using StructuredGaussianMixtures, Random
method = GraphicalLasso(penalty=0.1, maxiter=100, kkt_tol=1e-5)
X = randn(MersenneTwister(4), 5, 200)
result = fit(SparsePrecision(), method, X; report=true)
state = workspace(SparsePrecision(), method, result.model)
fit!(state, method, X)
mixture = fit(MixtureSpec(SparsePrecision(), 2),
              EM(covariance_method=method, maxiter=10), X)
```

The native solver uses a feasible positive-definite covariance warm start when
available; otherwise it constructs one from the scatter. `maxiter` limits full
block sweeps, and `inner_maxiter` limits coordinate sweeps within each lasso.
Convergence requires the final returned precision to pass the KKT check.
Exhaustion returns an SPD estimate with `:iteration_limit`, never a convergence
claim. Small entries are dropped using `zero_tol` only when both SPD and KKT
checks pass; otherwise the unthresholded precision is retained. Report iterations count
full block sweeps and history records the penalized objective.

For small reference problems, install `Convex` and a conic solver separately and
pass `optimizer=Convex.MOI.OptimizerWithAttributes(SCS.Optimizer, ...)` after
`using Convex, SCS`. Loading Convex does not change the default native solver.
The optional conic backend remains useful for independent correctness checks.

The method minimizes

```math
\operatorname{tr}((S+\eta I)Q)-\log\det Q
+\lambda\sum_{i\ne j}|Q_{ij}|,
```

where `S` is normalized weighted centered scatter, `η = regularization` and
`λ = penalty`. Both symmetric off-diagonal entries count. Diagonals are not L1
penalized. Positive ridge regularization helps when scatter is singular. The optional
Convex backend forms dense scatter and a conic optimization problem; it is a
reference backend for modest dimensions, not a large-scale graphical-lasso solver.

For the optional backend, the optimizer must support semidefinite and exponential cones. Solver tolerances
and iteration limits belong in the supplied optimizer. Only an `OPTIMAL` solver
status is accepted. The returned precision must be positive definite and pass an
independent KKT residual check. Off-diagonal magnitudes at most `zero_tol` are
zeroed before checking; a large threshold can invalidate the solution and is
rejected. Tighten solver tolerances if residual checks fail.

Single-Gaussian reports label their objective `:penalized_covariance_loglikelihood`:
minus half the above expression, including the Gaussian normalization constant.
`observed_objective` holds the unpenalized weighted log likelihood. For the optional backend, `iterations`
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
