# Newton–Adjoint scaling experiment

This experiment compares the unchanged public solvers from `c5a98c7`: `adjointBfgs`, `adjointNewton`, and `adjointNewtonExact`. Complete Newton is measured with both the dense joint Lagrangian Hessian and its Hessian-vector action. The benchmark code is on a separate branch so the solver implementations are not tuned to win the comparison.

## Controlled problems

Let m denote the number of design variables and n the number of implicit states. A fixed, well-conditioned lower-bidiagonal matrix A has diagonal 1 and subdiagonal -1/4. The implicit equation is c=A(x-h(p))=0, represented as a **dense** ml-matrix matrix. Each state selects one design variable; assignments cycle through the m variables. This permits m<n and m>n.

The objective is f(p,x)=F(p)+w^T(x-h(p)). The weights sum to one for every represented design variable. Consequently the feasible reduced objective is exactly F, whose gradient is evaluated independently in the benchmark.

- `easy`: h_j(p)=p_i and F=1/2 sum (p_i-1)². All three methods can reach the solution in one iteration. This exposes additional curvature/sensitivity work when it buys no iteration reduction.
- `curved`: h_j(p)=p_i² and F=sum[e_i^4/4+e_i^3/3+3e_i²/2], e_i=p_i-1. The adjoint remains nonzero at the solution. Full Newton includes the state-curvature contribution, while the approximate model omits it.
- `heterogeneous`: the same nonlinear cost, with distinct initial components 1.5+0.3 sin(i phi), phi=(1+sqrt(5))/2. The trajectory is not a replicated scalar iteration, so a crossover measured in the uniform family need not carry over.

The uniform families start at 1.5; all initial states are feasible. The condition of the reduced problem stays bounded as dimensions grow. These controlled problems deliberately separate dimension overhead from changes in conditioning or convergence difficulty. They are not a representative sparse PDE workload or a claim about every high-dimensional objective. The state equations couple neighboring residuals but an explicit state solution is available for correctness checking.

Analytical first derivatives are provided equally to all solvers. The primary sweep uses analytical curvature; the `-fd` follow-up computes second curvature by central differences while keeping first derivatives analytical. It is not a study of fully numerical first derivatives. Constant state Jacobians and constant curvature blocks are cached consistently. Optional Hessians are allocated lazily, so BFGS does not pay for an unused state Hessian. Complete dense Hessians include the lambda-weighted constraint curvature; the action callback computes exactly the same operator without allocating its joint matrix.

## Accuracy, time, and memory

The common stopping target is true reduced-gradient RMS <=1e-6, parameter-error RMS <=1e-6, and actual current-state constraint RMS <=1e-12. RMS normalization keeps the requested per-component accuracy constant as dimension increases. A benchmark-only callback verifies all three and throws a private sentinel to end the solve immediately. This avoids comparing time and memory accumulated after one method has already met the target. Solver algorithms are unchanged.

Time and memory use separate child processes, invoked with plain Node against the compiled public API. A TypeScript loader is not part of the measured process. Cases run serially, and method order rotates deterministically.

- **Time:** warmup solves are discarded, then repeated solve times are measured. Explicit GC is called between solves, outside the measured interval. Setup and Node startup are excluded. Evaluation counters include objective, constraint, first-partial and curvature callbacks. Common target monitoring is included equally for every method.
- **Memory:** every sample is a fresh cold process with a single solve. `process.resourceUsage().maxRSS` gives the OS-recorded process high-water RSS, including peaks inside synchronous linear algebra. It is converted from KiB to MiB. An event-loop sampling timer would miss those synchronous peaks and is deliberately not used.
- Peak RSS includes Node, module imports, model inputs, JIT, solver data, and garbage not yet collected. It is not `heapUsed`, nor is the difference between final and initial heap size presented as a peak. Pre-input, post-input and final memory snapshots are retained as context. Compare absolute process peaks; baseline subtraction is not an exact solver-only allocation measure.
- Medians and interquartile ranges are retained. Quartile separation describes observed timing ranges; it is not a confidence interval or formal significance test. Borderline cases need additional warmups and repetitions.

The initial sweep spans m=8..512 and n=128..1024 on selected slices, not the entire Cartesian product. Process timeouts cover **all warmups and repeated solves in that worker**, not one solve. Timeout rows retain their status and are not converted into guessed milliseconds. Longer-budget reruns and crossover refinements are saved separately so the measurement protocol stays auditable. V8's old-heap cap is reported; it is not a hard RSS limit, because typed-array storage can be external to that heap.

The initial sweep used a 90-second worker limit; the long approximate case was rerun successfully with 240 seconds. The script now defaults to 240 seconds for future full sweeps. Reproducing the original time-limited protocol requires `--timeout-ms=90000`; saved protocols remain authoritative for `--resume`.

The official [Node process resource-usage documentation](https://nodejs.org/docs/latest-v24.x/api/process.html#processresourceusage) defines the peak-RSS units. Python/Matplotlib plots consume saved JSON and are generated only after timing workers have finished.

## Interpreting crossover and memory results

There is no dimension-only crossover. At minimum, retain the objective family, fixed state/design count, curvature representation, convergence target, and runtime environment when naming an observed boundary. Complete Newton can lose its iteration-count advantage when BFGS already solves the reduced problem cheaply. Conversely, increasing n at small m can favor complete Newton if it avoids repeated state/adjoint evaluations.

Both Newton variants explicitly solve for S with m right-hand sides. Dense state factorization costs O(n³), sensitivity backsolves O(n²m), and reduced Hessian storage O(m²). A dense joint Hessian adds O((m+n)²) storage and projection work. The action callback removes that joint allocation but still assembles/factors the dense m-by-m reduced Hessian; it is not matrix-free Newton-CG. Repeated allocations and GC make process peak RSS depend on iteration count as well as matrix dimensions.

An important implementation limitation is visible in `src/core/bfgsUpdate.ts`: the inverse-Hessian update uses two general dense matrix products. Its update is therefore O(m³) in this library. Standard rank-two BFGS algebra can be implemented in O(m²). The observed crossover against this implementation must not be advertised as a universal comparison against an optimized BFGS implementation. Sparse state solvers and limited-memory or matrix-free outer methods can also change the result substantially.

## Observed conditional boundaries

On the 2026-11-05 run (Node v24.22.1, Windows x64, Intel Core i9-14900K), the seven-repeat refinements of the uniform curved family bracketed the dense full-Hessian crossover against BFGS between 128 and 160 designs at 128 states, and between 224 and 256 designs at 256 states. The timing quartile ranges were separated at the favorable and unfavorable endpoints. These are measured brackets for this objective and implementation, not a dimension-only rule.

With analytical first derivatives but finite-difference curvature, full Newton was favorable at 16 designs and unfavorable at 32 designs, with 128 states. The action callback delayed or avoided the dense-Hessian penalty on many tested slices. A persistent, universal action-callback crossover was not established.

The heterogeneous-start comparison changed the conclusion at 512 designs and 128 states: complete Newton was favorable again because BFGS needed more iterations. At 16 designs and 1024 states, complete Newton was favorable in both time and total process peak RSS. Peak memory did not have a single monotone boundary: dense matrix sizes, repeated solves, temporary allocations, and GC all contributed.

The Japanese summary, exact timings/RSS, interquartile ranges, per-run accuracy checks, and original timeout record are retained under `benchmark-results/adjoint-scaling-summary.*` and the source reports. They are machine-specific outputs and remain gitignored.

## Reproduction

```sh
npm run benchmark:adjoint-scaling
# Targeted examples; each gets its own report and raw protocol:
node scripts/adjoint-scaling-benchmark.mjs --sizes=128x128,160x128,192x128 --families=curved --warmups=8 --repeats=7 --memory-repeats=0 --name=adjoint-scaling-refine
node scripts/adjoint-scaling-benchmark.mjs --sizes=16x1024 --families=curved --timeout-ms=240000 --name=adjoint-scaling-extended
```

Raw data and per-run reports are saved under gitignored `benchmark-results/`. `--resume` requires an identical saved protocol. Memory repetitions can be zero for a timing-only refinement; those runs do not create memory measurements. `scripts/plot-adjoint-scaling.py` creates PNG/SVG figures from one or more saved reports. Warm time and cold process memory must remain labeled separately in any exported plot or table.

`npm run benchmark:adjoint-scaling:followups` runs the longer-budget case, seven-repeat timing refinements, numerical-curvature comparisons, and heterogeneous starts serially. `node scripts/summarize-adjoint-scaling.mjs` combines them into a Japanese summary and a provenance-preserving JSON file. Refinements replace timing samples, while the independent primary memory measurements are retained. Optional plots require Python and Matplotlib; the solver benchmarks themselves only use Node and the existing npm dependencies.
