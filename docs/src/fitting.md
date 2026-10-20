# Fitting

Structure specifies the model family; the method specifies how to fit it.
Data matrices are **features × observations**. `fit` returns a fitted distribution;
`report=true` instead returns a named tuple with `model` and `report`.

## Structures and supported methods

| Structure | Method | Output |
|---|---|---|
| `FullCovariance()` | `Exact()` alone, or `EM(covariance_method=Exact())` | `MvNormal` |
| `DiagonalCovariance()` | `Exact()` alone, or `EM(covariance_method=Exact())` | diagonal `MvNormal` |
| `IsotropicCovariance()` | `Exact()` alone, or `EM(covariance_method=Exact())` | isotropic `MvNormal` |
| `LowRankDiagonal(r)` | `CovarianceEM()` alone, or `EM(covariance_method=CovarianceEM())` | `LRDMvNormal` |
| `LatentCovariance(r)` | `CovarianceEM()` alone | `LatentMvNormal` |
| `LatentCovariance(r)` in `MixtureSpec(...; tied=Tied(:F,:D))` | `EM(covariance_method=CovarianceEM())` or `PCAEM()` | `LatentMvNormal` components |

`MixtureSpec(structure, k)` defaults to independent components. `Tied` describes
constraints, not array aliasing. Full, diagonal, and isotropic structures support
`Tied(:covariance)` with `EM(covariance_method=Exact())`. Shared `F,D` is supported
through PCAEM and joint covariance EM; other tied configurations are rejected
explicitly. PCAEM additionally constrains means to an affine PCA subspace.
`LatentCovariance(r; latent=DiagonalCovariance())` requests diagonal reduced-space
covariances; its default is full covariance. No one-to-one mapping between specs
and distribution types is required (full and diagonal both use `MvNormal`).

```@docs
FullCovariance
DiagonalCovariance
IsotropicCovariance
LowRankDiagonal
LatentCovariance
MixtureSpec
Tied
Exact
CovarianceEM
EM
PCAEM
KMeansInit
RandomInit
RandomLoading
```

## Single Gaussians and mixtures

```@example fitting
using StructuredGaussianMixtures, Distributions, Random, Statistics
rng = MersenneTwister(12)
X = randn(rng, 8, 200)
w = rand(rng, 200)
gaussian = fit(FullCovariance(), Exact(regularization=1e-6), X; weights=w)
spec = MixtureSpec(LowRankDiagonal(2), 3)
method = EM(covariance_method=CovarianceEM(maxiter=10), maxiter=20, n_init=2)
result = fit(spec, method, X; weights=w, rng, report=true)
gmm = result.model
result.report.status, result.report.objective
```

Single-Gaussian fitting estimates a weighted mean and calls the covariance solver
directly, without mixture responsibilities or outer EM. Exact fitting reports one
closed-form update. LRD reports its inner covariance iterations. The mixture M-step
uses that same path for each independent component. Tied covariances instead use
a joint solve, keeping component means separate. Centered weighted observations are passed
to the covariance implementation; LRD never requires a dense scatter matrix.

Weights must be finite and nonnegative with positive total mass; they are normalized.
Zero-weight samples are removed before initialization and fitting. All supplied data
must be finite, including zero-weight columns. Exact covariance estimates use the
MLE denominator (total weight), not the unbiased sample covariance denominator.
Scaling weights does not change the fit. Integer-weight/replication equivalence
holds for updates from equivalent initialized parameters; randomized initialization
need not draw the same centers from differently represented datasets.

## Initialization and continuation

```@example fitting
state = initialize(spec, method, X; weights=w, rng)
fit!(state, method, X; weights=w)
first_objective = state.report.objective
fit!(state, method, X; weights=w)

# Copy a fitted model into independent mutable fitting state.
state = workspace(spec, method, gmm)
fit!(state, method, X; weights=w)
```

`fit` uses method-owned initialization and restarts; `fit!` performs one continuation
run using current parameters. It does not call initializers, run restarts, or reset
LRD residual variances, even if those settings differ in the supplied method.
`initialize` creates one state, independent of `n_init`. A report describes the most
recent invocation; its history starts with the objective before that invocation.
`tol=0` disables early stopping for fixed-iteration comparisons. `maxiter=0` evaluates
the initial model without updates. Workspaces own copied parameters; models returned
by `fit!` are the current fitted snapshot. Workspace fields are implementation details.

`KMeansInit(maxiter=50)` performs weighted Lloyd iterations from randomly selected
positive-weight observations. Empty clusters retain their previous center.
`RandomInit()` uses those selected observations directly. Both initialize covariances
from global weighted data and equal mixture masses. `RandomLoading()` initializes
small nonzero LRD or latent loadings, with latent `A=I`; zero loadings would be a
stationary point of covariance EM. It is a random starting point, not a PCA fit.
Use explicit `rng` for reproducibility; continuation does not consume randomness.

```@docs
StructuredGaussianMixtures.fit(::StructuredGaussianMixtures.GaussianStructure, ::StructuredGaussianMixtures.CovarianceMethod, ::AbstractMatrix)
fit!
initialize
workspace
GaussianWorkspace
MixtureWorkspace
PCAWorkspace
FitReport
responsibilities
```

## PCAEM lifecycle and objectives

```@example fitting
pca_spec = MixtureSpec(LatentCovariance(2), 3; tied=Tied(:F,:D))
pca_method = PCAEM(latent_method=EM(maxiter=20, n_init=2), residual_floor=1e-6)
pca = fit(pca_spec, pca_method, X; rng, report=true)
pca.report.objective_kind, pca.report.observed_objective
pca_state = initialize(pca_spec, pca_method, X; rng)
fit!(pca_state, pca_method, X)
fit!(pca_state, pca_method, X) # Retains F, D and the projection offset.
```

PCAEM computes weighted principal directions, fits the projected observations and
estimates a shared residual diagonal. It is a projected estimator, **not joint
observed-space maximum likelihood**. Its report labels the projected likelihood;
`observed_objective` separately evaluates the reconstructed observation model.
Nested restart reports describe the projected fit. Retain `PCAWorkspace` to continue:
a fitted observation model alone does not uniquely identify the original PCA offset.
The PCA rank must not exceed the numerical centered-data rank or
`min(features, positive-weight observations - 1)`.

PCA residual variances now use normalized weighted squared residuals (MLE), aligning
weight semantics with the other solvers. Previously the unweighted estimator used
sample variance: without flooring, the new residuals are `(n-1)/n` times the old ones.

## Convergence, regularization and failures

`Exact(regularization=λ)` adds `λI` to the MLE covariance. `CovarianceEM(variance_floor=λ)`
constrains residual variances to at least `λ`. `PCAEM(residual_floor=λ)` floors shared
reconstruction noise. These settings are separate from outer EM convergence options.
Inner covariance EM uses small Cholesky solves and preserves the previous covariance
between M-steps. The covariance objective is evaluated at zero-mean residuals.

EM stops when the absolute objective change is at most `tol * (1 + abs(previous))`.
Reports distinguish `:converged`, `:iteration_limit`, `:failed` and `:initialized`.
The objective is a normalized weighted log likelihood, not a total log likelihood.
Responsibilities use log-domain normalization and batched component scores.
An outer iteration commits parameters only after all component updates succeed.

A component with mass at or below `min_mass` fails the run explicitly; it is not
silently reinitialized. Numerical covariance failures are also reported. `fit!`
returns the last valid model with a failed report; callers must inspect that report.
An initial numerical scoring failure also produces `:failed`: the model is
unchanged, the history is empty, and the objective remains `-Inf`. An M-step
failure retains the last successfully evaluated model and objective.
Fresh `fit` excludes failed restarts, selects the largest final objective among
successful runs, and throws if all fail. Reports retain each restart's diagnostics.
Invalid configuration/data raise errors rather than being treated as failed restarts.

Unregularized exact EM on well-conditioned data should improve likelihood.
Regularization and PCAEM do not imply monotonic observed likelihood; a small change
is the stopping criterion, not a claim of global optimality. Iteration limits apply
per invocation. Full/diagonal solves need positive definite fitted covariance;
use positive regularization for singular data.

## Migration

The former fitting interface is intentionally removed:

| Previous call | New call |
|---|---|
| `fit(EM(k), X)` | `fit(MixtureSpec(FullCovariance(), k), EM(), X)` |
| `fit(EM(k; kind=:diag), X)` | `fit(MixtureSpec(DiagonalCovariance(), k), EM(), X)` |
| `fit(FactorEM(k,r), X)` | `fit(MixtureSpec(LowRankDiagonal(r), k), EM(covariance_method=CovarianceEM()), X)` |
| `fit(PCAEM(k,r), X)` | `fit(MixtureSpec(LatentCovariance(r), k; tied=Tied(:F,:D)), PCAEM(), X)` |

Pass weights as `weights=w`. Old outer `nIter` becomes `EM(maxiter=...)`;
`nInternalIter` becomes `CovarianceEM(maxiter=...)`. FactorEM's restart count becomes
`n_init`. The external GMM's `nInit` was a k-means iteration budget: use
`KMeansInit(maxiter=...)`, not `n_init`. External split initialization and `nFinal`
staging are not retained. Use explicit continuation for staged budgets.

The implementation no longer depends on GaussianMixtures.jl. Random initialization
and trajectories differ; predictions and supported model families remain available.
Distribution operations retain the PR1 interfaces described in the Gaussian guides.

## Joint latent covariance EM

`CovarianceEM` also fits `LatentCovariance(r)` as a single Gaussian or as a
mixture with exactly `Tied(:F, :D)`. Component means and latent covariances remain
independent; the loading and diagonal residual variances are learned jointly.

```@example fitting
spec = MixtureSpec(LatentCovariance(3), 4; tied=Tied(:F, :D))
method = EM(covariance_method=CovarianceEM(maxiter=5), maxiter=50)
state = initialize(spec, method, X; rng)
fit!(state, method, X; weights=w)
fit!(state, method, X; weights=w) # Continues all parameters without initialization.
posterior = responsibilities(state.model, X)

# A single weighted Gaussian uses the same covariance update.
g = fit(LatentCovariance(3), CovarianceEM(), X; weights=w, rng)

# Restrict each component's latent A to be diagonal:
diagonal_latent = MixtureSpec(
    LatentCovariance(3; latent=DiagonalCovariance()), 4; tied=Tied(:F, :D))
```

The covariance is `F*A_k*F' + Diagonal(D)`. At each outer M-step, component
responsibilities and updated means stay fixed during an inner joint EM fit.
Posterior latent first and second moments produce small sufficient statistics;
loading updates pool those statistics, each `A_k` updates independently, and the
residual diagonal pools expected squared residuals. No dense observation scatter
matrix is formed. Centered observations are processed in bounded blocks, reusing
scratch arrays across components and inner sweeps; the implementation does not
store a centered dataset for each component. The posterior-moment pass also
computes likelihood using a sum of squared residual and latent norms, avoiding
the subtraction of two large Woodbury quadratic terms.

For component mass ``m_k``, let ``C_k=E[xz^T]`` and ``H_k=E[zz^T]`` be normalized,
responsibility-weighted posterior moments of centered observations. One inner EM
sweep updates

```math
F_{new}=\left(\sum_k m_k C_k\right)\left(\sum_k m_k H_k\right)^{-1},
\qquad A_{k,new}=H_k.
```

For diagonal latent covariance, replace ``H_k`` by its diagonal in the ``A_k``
update. The shared residual variance is the mass-weighted diagonal of
``E[(x-F_{new}z)(x-F_{new}z)^T]``, clipped at `variance_floor`. The implementation
uses small Cholesky solves, not explicit inverses. These are ordinary latent
variable EM updates for the observed-space likelihood with fixed responsibilities;
there is no separate optimizer or ADMM loop.

`state.inner_reports` contains **one joint covariance report**, whose history is
the component-mass-weighted covariance log likelihood. It is not a per-component
report. One inner sweep already constitutes a generalized EM update; `maxiter`
controls the maximum number of inner sweeps **per outer M-step**. Start with a small
inner budget, inspect `inner_reports`, and use continuation or a larger budget when
needed; fully converging every inner fit can waste work while responsibilities are
still changing. Each new M-step starts from the current parameters. The default
remains 20 sweeps; no automatic schedule or acceleration is applied.
`variance_floor` applies to D, not A.
Latent covariance factors are fitted with positive-definite A; continuation rejects
singular A even though `LatentMvNormal` can represent it for other operations.

Unlike PCAEM, this method learns F and D through the observed likelihood and does
not constrain means to the initial PCA affine subspace. Full latent covariance has
nonunique coordinates: simultaneous transformations of F and A can preserve the
observation covariance. No coordinate normalization is imposed; compare covariance
and likelihood, rather than individual loading entries, across runs. EM reaches
stationary points and may converge slowly; neither `:converged` nor a fixed
iteration budget guarantees the global mixture optimum. Use explicit seeds and
`n_init` to assess different starts.

Fitted components retain independently owned copies of equal F and D arrays,
consistent with `LatentMvNormal` ownership. `workspace` validates those equalities,
and joint updates preserve them. This release does not support separate D with
shared F, separate F with shared D, or other partial-sharing combinations.

## Isotropic and shared covariance

```@example fitting
isotropic = fit(IsotropicCovariance(), Exact(), X; weights=w)
tied_spec = MixtureSpec(FullCovariance(), 3; tied=Tied(:covariance))
tied_method = EM(covariance_method=Exact(regularization=1e-5), maxiter=10)
tied_state = initialize(tied_spec, tied_method, X; weights=w, rng)
tied_model = fit!(tied_state, tied_method, X; weights=w)
length(tied_state.inner_reports) # one joint covariance report
```

Use `DiagonalCovariance()` or `IsotropicCovariance()` in the same mixture
specification for a shared diagonal or scalar covariance. Component means and
mixture weights remain independent. Each exact M-step pools the weighted scatter
**around each component's updated mean**, divides by total mass, then projects to
the requested covariance structure. Isotropic covariance is the average marginal
variance times the identity. `Exact.regularization` adds a variance ridge once
after pooling; it is not scaled by component mass.

A tied M-step produces one inner covariance report, whose objective is the
mass-weighted conditional Gaussian log likelihood (excluding mixture weights).
The resulting components reuse one covariance representation. Treat fitted
parameters as read-only: `workspace(spec, method, model)` copies the model and
checks equality of its component covariances before continuation. Initialization
and subsequent updates preserve the specified tying constraints.

The internal `_fit_components(spec, covariance_method, components, X, weights)`
operation is the joint M-step dispatch point. Its samples × components weight
matrix contains effective observation weights; column sums retain component
masses. Independent fitting delegates to single-Gaussian fitting, while tied
methods combine statistics before normalizing. It returns components and inner
reports, leaving responsibilities and outer convergence to the EM driver.
If an inner covariance solver reports `:failed`, outer EM retains the last valid
mixture and reports the inner failure instead of accepting that candidate.
