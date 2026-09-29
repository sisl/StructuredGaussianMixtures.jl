# Latent representation benchmarks

Run `julia --project=. benchmark/latent.jl` from the repository root. To reproduce
the baseline, run the same script with `--project=/path/to/main-checkout`.
Use the same resolved dependencies for both checkouts. The script fixes BLAS to
one thread, warms each closure twice, runs GC outside each measurement, and
reports median elapsed milliseconds and minimum allocated bytes across 15 runs
(5 for fitting). Compilation is excluded. These are local observations, not CI
performance thresholds; microsecond timings are sensitive to system load.

Measured with Julia 1.11.5 on macOS ARM64, baseline commit `315f965`, using the
same dependency manifest for both processes. `baseline.csv` and `latent.csv`
contain all timings and allocations. Data and fitting seeds are fixed in the script.

| Operation | Main (ms) | PR (ms) | Observation |
|---|---:|---:|---|
| LRD scalar, p=100/r=5 | 0.0280 | 0.0230 | Shared Cholesky kernel |
| LRD batch, p=100/r=5/n=500 | 3.3172 | 0.2190 | 15.1× faster |
| LRD scalar, p=500/r=10 | 0.0728 | 0.0358 | 2.0× faster |
| LRD batch, p=500/r=10/n=1000 | 56.9416 | 2.2312 | 25.5× faster |
| LRD conditioning, p=500/r=10 | 0.0615 | 0.0611 | Algorithm unchanged |
| PCAEM fit, p=60/r=5/n=300/K=3 | 0.9598 | 0.9231 | Similar runtime |
| FactorEM fit, p=60/r=5/n=300/K=3 | 15.5512 | 8.1281 | 1.9× faster; fitting updates unchanged |

The speedup comes from preparing once per batch and replacing repeated general
solves with small Cholesky solves. The quadratic is evaluated as a sum of squares
rather than a subtraction of large terms. The 500-dimensional LRD batch allocates
12,389,952 bytes versus 164,568,256 bytes on main.

For the new latent representation, the 500-dimensional batch takes 2.2830 ms and
allocates 12,455,568 bytes, close to the equivalent LRD batch. Scalar scoring takes
0.0412 ms versus 0.0358 ms for equivalent LRD, reflecting formation of `L * B`.
Latent conditioning takes 0.0738 ms and allocates 206,736 bytes versus 0.0611 ms
and 140,496 bytes for LRD. It retains and copies both the basis and latent factor,
so it has additional storage/ownership costs. Parameter preparation is not cached
across calls; persistent preparation plans are deferred to the fitting refactor.
