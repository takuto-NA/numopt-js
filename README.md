# numopt-js

A flexible numerical optimization library for JavaScript/TypeScript that works smoothly in browsers.

## Documentation

- **API Reference (GitHub Pages)**: https://takuto-na.github.io/numopt-js/
- **Source Repository**: https://github.com/takuto-NA/numopt-js

## Requirements

- Node.js >= 18.0.0
- Modern browsers with ES2020 support (for browser builds)

## Installation

```bash
npm install numopt-js
```

The published package ships `dist/`, this README, and the license. Runnable tutorials under `examples/` are available from a git clone of this repository (not from the npm tarball).

## Start Here

| Goal | Algorithm | Inputs |
|------|-----------|--------|
| Minimize a scalar cost | Gradient Descent, BFGS, L-BFGS | `cost(p) -> number`, `grad(p) -> Float64Array` |
| Black-box scalar cost | CMA-ES | `cost(p) -> number` |
| Nonlinear least squares | Gauss–Newton, Levenberg–Marquardt | `residual(p) -> Float64Array` |
| Equality constraints \(c(p,x)=0\) | Adjoint GD/BFGS/L-BFGS, approximate/full Newton–Adjoint, Constrained GN/LM | cost/residual + `constraint(p,x)` |

Adjoint is a reduced-space solver: the constraint defines \(x(p)\), so the optimizer searches only \(p\). That is the point when there are few design variables and many implicit states. Adjoint BFGS is standard dense BFGS on that reduced cost; the gradient is the adjoint. Constrained GN/LM use the same split and are usually better for small-residual least squares. A penalty method on the concatenated \((p,x)\) does not.

**Why `Float64Array`?** Predictable numeric performance. Convert with `new Float64Array([1, 2, 3])`.

Least-squares solvers minimize \(f(p) = 1/2 \|r(p)\|^2\).

## Result Object

Common fields: `finalParameters`, `converged`, `iterations`, `finalCost`.

- Gradient / BFGS / L-BFGS: `finalGradientNorm`
- CMA-ES: `functionEvaluations`, `finalStepSize`, `stopReason`, optional `profiling`
- GN / LM: `finalResidualNorm` (LM also has `finalLambda`)
- Constrained / Adjoint GD / Adjoint BFGS: `finalStates`, `finalConstraintNorm`

`result.parameters` is a deprecated alias of `result.finalParameters`.

## Quick Start (Node)

### ESM

```js
import { gradientDescent } from 'numopt-js';

const cost = (params) => params[0] * params[0] + params[1] * params[1];
const grad = (params) => new Float64Array([2 * params[0], 2 * params[1]]);

const result = gradientDescent(new Float64Array([5, -3]), cost, grad, {
  maxIterations: 200,
  tolerance: 1e-6,
  useLineSearch: true,
});

console.log(result.finalParameters);
```

### CommonJS

```js
const { gradientDescent } = require('numopt-js');

const cost = (params) => params[0] * params[0] + params[1] * params[1];
const grad = (params) => new Float64Array([2 * params[0], 2 * params[1]]);

const result = gradientDescent(new Float64Array([5, -3]), cost, grad, {
  maxIterations: 200,
  tolerance: 1e-6,
  useLineSearch: true,
});

console.log(result.finalParameters);
```

## BFGS / L-BFGS

```js
import { bfgs, lbfgs } from 'numopt-js';

const cost = (params) => (params[0] - 1) ** 2 + (params[1] + 2) ** 2;
const grad = (params) => new Float64Array([2 * (params[0] - 1), 2 * (params[1] + 2)]);

const bfgsResult = bfgs(new Float64Array([10, 10]), cost, grad, {
  maxIterations: 200,
  tolerance: 1e-8
});

const lbfgsResult = lbfgs(new Float64Array([10, 10]), cost, grad, {
  maxIterations: 200,
  tolerance: 1e-8,
  historySize: 10
});
```

## Adjoint BFGS

Same inputs as adjoint gradient descent: design parameters \(p\), implicit states \(x\), and \(c(p,x)=0\). The search stays in \(p\); Strong Wolfe is on by default.

```js
import { adjointBfgs } from 'numopt-js';

const cost = (p, x) => p[0] * p[0] + x[0] * x[0];
const constraint = (p, x) => new Float64Array([p[0] + x[0] - 1]);

const result = adjointBfgs(new Float64Array([2]), new Float64Array([-1]), cost, constraint, {
  maxIterations: 100,
  tolerance: 1e-6
});

console.log(result.finalParameters, result.finalStates, result.finalConstraintNorm);
```

Convergence requires both the reduced gradient and the constraint residual to pass their tolerances.
`stopReason` distinguishes `gradientTolerance`, `stepStagnation`, `lineSearchFailure`,
`stateRestorationFailure`, `nonFiniteEvaluation`, and `maxIterations`. A small step alone
does not establish convergence. `stepTolerance` controls relative parameter-step stagnation
(default `16 * Number.EPSILON`), separately from gradient `tolerance`.

Both line-search helpers return `0` when their budgets expire without a verified step.
Strong Wolfe requires sufficient decrease and its curvature condition; Armijo backtracking
requires sufficient decrease. Quasi-Newton callers stop unsuccessfully when a search fails.

## Implicit residual derivatives

`computeEffectiveJacobian(p, x, residual, constraint, options)` returns
`J_eff = r_p - r_x C_x^+ C_p`. For `f = 1/2 ||r||²`, use `J_eff.transpose() * r`
as the reduced gradient passed to ordinary BFGS. Omitted partials use finite differences;
options `drdp`, `drdx`, `dcdp`, and `dcdx` accept analytical `ml-matrix` matrices.

```js
import { computeEffectiveJacobian } from 'numopt-js';
import { Matrix } from 'ml-matrix';

const residual = (p, x) => new Float64Array([p[0] - 1, x[0] - 2]);
const constraint = (p, x) => {
  const closure = x[0] - p[0] ** 2;
  return new Float64Array([closure, 2 * closure]); // consistent redundant rows
};
const p = new Float64Array([1.5]);
const x = new Float64Array([p[0] ** 2]); // solve states first at every trial p
const jacobian = computeEffectiveJacobian(p, x, residual, constraint);
const gradient = jacobian.transpose().mmul(Matrix.columnVector(residual(p, x))).to1DArray();
```

The caller must provide feasible states on a consistent branch. A true implicit derivative
requires a nonsingular square state Jacobian, or a full-column-rank rectangular one with
compatible parameter derivatives. Otherwise the result is a least-squares/regularized
linearization and does not establish a feasible state function. Regular square/tall systems
use one LU/QR factorization for all parameter columns. The four-bar example includes state
solves, BFGS integration, and redundant constraints; see [the validation](research/mechanical-character.md).

## Newton–Adjoint with an approximate Hessian

`adjointNewton` implements the approximation in [section 12 of “Newton's method and the Adjoint method”](https://qiita.com/naturemon/items/df0242e8a2e5814503f5):

\[
S=-c_x^{-1}c_p,\qquad H=f_{pp}+S^T f_{xx}S,\qquad H\Delta p=-\nabla\tilde f.
\]

It uses the adjoint for the reduced gradient, diagonal damping when the approximate Hessian is not positive definite, and Armijo backtracking starting at a unit step. Every accepted point has restored feasible states. Small steps with a large gradient stop with `converged: false`.

```js
import { adjointNewton } from 'numopt-js';

const result = adjointNewton(
  new Float64Array([2]), new Float64Array([-1]),
  (p, x) => p[0] ** 2 + x[0] ** 2,
  (p, x) => new Float64Array([p[0] + x[0] - 1]),
  { tolerance: 1e-6 }
);
// p ≈ 0.5, x ≈ 0.5
console.log(result.stopReason, result.finalGradientNorm, result.finalConstraintNorm);
```

Optional `d2fdp2(p,x)` and `d2fdx2(p,x)` return symmetric `ml-matrix` matrices of sizes parameterCount × parameterCount and stateCount × stateCount. They are **partial** Hessians of the scalar cost, holding the other argument fixed. Residual input is supported with cost `f = 1/2 ||r||²`; Hessian callbacks still differentiate that scalar cost.

Without these callbacks, central second differences evaluate `f_pp` and the projected state curvature `S^T f_xx S` directly. This takes O(parameterCount²) objective samples per model, without constructing a dense state Hessian. `hessianStepSizeP` / `hessianStepSizeX` default to relative steps of `1e-4`; choose analytical blocks when finite differences suffer from noise, cancellation, or domain boundaries. Sensitivities use one dense state-Jacobian factorization with multiple right-hand sides.

**This is an incomplete reduced Hessian.** It omits mixed p–x partials and curvature of the state solution x(p). It can outperform BFGS when those terms vanish or are small, but can converge only linearly or be substantially slower when they matter. It does not guarantee quadratic convergence. Dense sensitivity solves also add work relative to BFGS, especially with many parameters. Prefer `adjointBfgs` for a general scalar objective and Constrained GN/LM for small-residual least squares unless a comparison supports this model.

Results additionally include `stopReason`, `hessianEvaluations`, `dampedSteps`, `gradientFallbackSteps`, and `finalHessianDamping`. `regularization` affects the state/adjoint solves; `hessianDamping` is a separate minimum shift of H. Damping does not repair a missing or nonunique x(p).

Run `npm run example:adjoint-newton` and `npm run benchmark:adjoint-newton` from a clone. The benchmark compares analytical and numerical derivatives, includes two omitted-curvature counterexamples, reports warmed median/IQR timing and evaluation counts, and writes gitignored Markdown/JSON under `benchmark-results/`. See [the analysis](research/newton-adjoint.md) for the exact missing terms and convergence limits.

## Adjoint L-BFGS

`adjointLbfgs` applies L-BFGS to the reduced objective `f(p,x(p))`, using adjoint gradients and feasible state restoration. It accepts the same analytical first-partial callbacks as `adjointBfgs`; omitted callbacks use numerical differentiation.

```js
import { adjointLbfgs } from 'numopt-js';

const result = adjointLbfgs(
  new Float64Array([1.4]), new Float64Array([-10]),
  (p, x) => (x[0] - 4) ** 2 + 0.1 * (p[0] - 2) ** 2,
  (p, x) => new Float64Array([x[0] - p[0] ** 2]),
  { historySize: 10, tolerance: 1e-7, constraintTolerance: 1e-12 }
);
console.log(result.finalParameters, result.stopReason); // approximately [2], gradientTolerance
```

With `q=historySize` (default 10), direction computation and correction storage cost O(q × parameterCount). No dense inverse Hessian is allocated. Dense state Jacobians and state/adjoint solves still contribute O(stateCount²) storage and O(stateCount³) factorization work; the constraint parameter Jacobian also uses O(stateCount × parameterCount) storage. Increasing `q` can improve convergence on difficult problems, but does not guarantee fewer iterations or full BFGS convergence rates.

The solver tries a unit step in Strong Wolfe search, checks actual Armijo decrease before acceptance, clears unusable curvature history, and reports `converged=true` only when both gradient and feasibility tolerances pass. `stopReason` distinguishes `gradientTolerance`, `stepStagnation`, `lineSearchFailure`, `stateRestorationFailure`, `nonFiniteEvaluation` and `maxIterations`; `maxHistoryPairs` reports the largest retained history. Accepted gradients and restored states are reused to avoid redundant solves. See [the comparison protocol and measurements](research/adjoint-lbfgs.md).

## Full Newton–Adjoint

`adjointNewtonExact` includes the mixed partials and nonlinear state-curvature terms omitted by `adjointNewton`. With `L(p,x,λ)=f(p,x)-λ^T c(p,x)`, it uses the **full reduced Hessian**

\[
H=Z^T\nabla^2_{(p,x)}L\,Z,\qquad Z=\begin{bmatrix}I\\S\end{bmatrix},\quad S=-c_x^{-1}c_p.
\]

The constraint Hessian in L accounts for the contracted second state sensitivities; no second-sensitivity tensor is stored. This is a dense reduced Newton solver with the same feasibility, damping, and Armijo safeguards as the approximate version.

```js
import { adjointNewtonExact } from 'numopt-js';

const result = adjointNewtonExact(
  new Float64Array([1]), new Float64Array([1]),
  (p, x) => (p[0] - 1) ** 2 / 2 + x[0],
  (p, x) => new Float64Array([x[0] - p[0] ** 2])
);
// Full reduced cost: 3/2 p² - p + 1/2; optimum p=1/3.
console.log(result.finalParameters, result.stopReason);
```

Supply either `lagrangianHessian(p,x,lambda)` (a symmetric joint matrix in **[p,x] order**) or `lagrangianHessianVector(p,x,lambda,direction)` (its action on a joint vector). Both differentiate **f - lambda^T c with lambda fixed**, including all constraint curvature. A Hessian of f alone is insufficient on nonlinear constraints. The vector callback avoids allocating the joint Hessian, while the solver still assembles a dense parameter-space Hessian.

Without a callback, central directional differences of L include all terms, with `hessianStepSize` defaulting to `1e-4`. “Exact” identifies the complete mathematical Hessian model; numerical differentiation and state/adjoint solves still have errors. Accurate analytical derivatives and solves can give local quadratic convergence near a regular minimum, when damping vanishes and unit steps are accepted. This does not guarantee global convergence or lower total cost than BFGS.

Both Newton solvers tighten state restoration as needed relative to the gradient tolerance, subject to a roundoff scale and the user's feasibility upper bound. A repeated unchanged cost with a large gradient reports `costStagnation`, rather than cycling or reporting success. Shared state/adjoint solves now try the unshifted system first when `regularization=0`; singular solves retain regularization retries.

Run `npm run example:adjoint-newton-exact` or `npm run benchmark:adjoint-newton-exact`. The comparison measures BFGS, approximate Newton and full Newton at the same independently evaluated stationarity/parameter target, with analytical/numerical derivatives and dimension scaling. Reports and local convergence traces are saved under `benchmark-results/adjoint-newton-exact.*`. See [the derivation and cost analysis](research/newton-adjoint-exact.md).

## Dimension scaling and peak memory

Run `npm run benchmark:adjoint-scaling` to compare BFGS, approximate Newton, full Newton with a dense joint Hessian, and full Newton with a Hessian-vector callback. The sweep varies design and state counts separately, up to 512 designs and 1024 states on selected slices. It measures time to a common independently verified RMS accuracy and cold-process peak RSS, rather than final heap size. Solvers and measured processes run serially.

`npm run benchmark:adjoint-scaling:followups` adds longer-budget reruns, crossover refinements, numerical second curvature (analytical first derivatives), and heterogeneous initial values. Then run `node scripts/summarize-adjoint-scaling.mjs` for a combined report under gitignored `benchmark-results/`. Optional PNG/SVG plots use `scripts/plot-adjoint-scaling.py` with Python/Matplotlib.

The observed crossover depends on convergence difficulty and the Hessian representation. Peak RSS includes the runtime, cached model inputs, temporary matrices, and GC behavior. The current BFGS update also uses O(parameterCount³) dense matrix products; these measurements do not describe a rank-two-optimized BFGS or a sparse PDE solver. See [the protocol and interpretation](research/adjoint-scaling.md).

## CMA-ES

```js
import { cmaEs } from 'numopt-js';

const sphere = (params) => params.reduce((sum, value) => sum + value * value, 0);

const result = cmaEs(new Float64Array([10, -7, 3, 5]), sphere, {
  maxIterations: 200,
  populationSize: 20,
  initialStepSize: 2.0,
  randomSeed: 123456,
  targetCost: 1e-10,
  restartStrategy: 'none',
  profiling: true,
});
```

Use `restartStrategy: 'ipop'` for multi-modal problems.

## Browser Usage

Prefer a bundler (Vite/Webpack/Rollup) and `import { gradientDescent } from 'numopt-js'`.

Without a bundler, use an import map pointing at `dist/index.browser.js`, or import that file by path. Serve over HTTP (not `file://`). For SSR frameworks, run optimization on the client.

## Examples (git clone)

Clone this repository, then:

```bash
npm install
npm run example:rosenbrock
```

Recommended order:

1. `npm run example:rosenbrock` — scalar cost + line search
2. `npm run example:lm` — residual least squares
3. `npm run example:gauss-newton` — undamped NLS
4. `npm run example:cma-es` — derivative-free
5. `npm run example:constrained` — Constrained LM / GN / Adjoint GD / Adjoint BFGS on one problem
6. `npm run example:adjoint` — basic adjoint (GD and BFGS)
7. `npm run example:adjoint-advanced` — harder adjoint cases (GD and BFGS)
8. `npm run example:adjoint-reduced` — few parameters vs many implicit states
9. `npm run example:layout-toy` — small layout toy
10. `npm run example:mechanical-character` — synthetic four-bar trajectory fitting based on Coros et al. (2013), section 4.2; compares BFGS with reference/public/redundant implicit gradients, full finite differences, Adjoint BFGS, and Constrained GN/LM. Reports strict accuracy failures with exit code 1; see [scope, capabilities, and validation](research/mechanical-character.md).
11. `npm run example:linkage-trajectory` — fit only three moving link lengths, keeping ground pivots, marker attachment and topology fixed. Generates an offline HTML replay with iteration/phase sliders, animation and independently verified reachable/distorted targets. See [the experiment and reproducible checks](research/linkage-trajectory-example.md).

Manual benchmarks (not CI):

- `npm run benchmark:paper` — paper-grade comparison of every public solver class. Warmup 1 run is discarded; deterministic solvers repeat 7 times (median and IQR); CMA-ES uses 5 seeds (warmup seed 1, timed seeds 1–5). Success is parameter error (and constraint norm when constrained), never `result.converged`. GD / BFGS / L-BFGS / Adjoint GD / Adjoint BFGS use analytical derivatives; GN / LM / Constrained / Penalty use numeric Jacobians. Rosenbrock CMA-ES uses IPOP; Rosenbrock GD may hit a 10000-iteration cap and still succeed on parameter error. Primary metrics are evaluation counts and success; wall-clock is secondary and machine-dependent. Writes `benchmark-results/paper-benchmark.md` and `.json` (gitignored). Hypothesis failures set exit code 1; do not commit the generated numbers.
- `npm run benchmark:constrained`, `npm run benchmark:curve-bending` — informal single-run smokes.
- `npm run benchmark:mechanical-character` — reachable and distorted four-bar targets, independent held-out geometry checks, warmed median/IQR timings, and gradient work counts. Writes Markdown/JSON under `benchmark-results/`; returns exit code 1 if a strict accuracy check fails. This is a synthetic validation of section 4.2, not a reproduction of the original characters or the full paper. Optional plots: `python scripts/plot-mechanical-character.py` (Matplotlib).
- `npm run benchmark:mechanical-character:robustness` — three initial designs for four-bars and slider-cranks, near-toggle sensitivities, and expected failures for inconsistent or infeasible states. Writes separate JSON/Markdown reports; accuracy and safety checks determine the exit code.
- `npm run benchmark:mechanical-character:paper` — paper-guided Eqs. (2)-(4) checks, negative gradient controls, redundant constraints, geared five-bars (ratios -2/-1/1/2; six/eight design variables), phase refinement and gradient-only timings. Includes independent checks for tanh saturation; currently returns exit code 1 for two single-start fits that fail the original-design first-order check. All failures and best-of-three initialization results are retained; see [findings and scope](research/mechanical-character-paper-validation.md).
- `npm run benchmark:linkage-trajectory` — three initial designs × two targets × public-IFT BFGS / full-FD BFGS / Constrained LM, with three timed repeats (54 trials). Checks unused phases, closure, assembly branch and every recorded iteration. Optional plot after the Example: `python scripts/plot-linkage-trajectory.py` (Matplotlib).

Full signatures and options: [TypeDoc API reference](https://takuto-na.github.io/numopt-js/).

## Convergence Options (quick map)

- **GD / BFGS / L-BFGS / GN / Constrained GN / Adjoint GD / Adjoint BFGS**: `tolerance`
- **LM / Constrained LM**: `tolGradient`, `tolStep`, `tolResidual` (or `tolerance` as a fallback when those are omitted)
- **CMA-ES**: `functionTolerance`, `parameterTolerance`, `targetCost`, `maxFunctionEvaluations`
- **Adjoint (ill-conditioned ∂c/∂x)**: `regularization`
- **Adjoint Newton**: `tolerance` (reduced gradient), `constraintTolerance`, `stepTolerance` (stagnation); `hessianDamping` for the curvature model
- **Adjoint BFGS**: `tolerance` (reduced gradient), `constraintTolerance`, separate relative `stepTolerance` (stagnation)

Result printing helpers: `printResult` / `formatResult` — see TypeDoc. Typed `print*` / `format*` variants are deprecated.

## Troubleshooting

- **Does not converge**: try better initials, raise `maxIterations`, relax tolerances, enable line search for GD/Adjoint, use `logLevel: 'DEBUG'`.
- **Adjoint rejects the start**: `∂c/∂x` must be square, and \(x(p)\) must exist near the guess. Projectable implicit-state starts only; \(|p| > \sqrt{2}\) on \(p^2+x^2=2\) has no real \(x\).
- **Singular / ill-conditioned Jacobian**: prefer LM / Constrained LM; for Adjoint try `regularization` and feasible initials.
- **Wrong numeric type**: pass `Float64Array`, not plain arrays.

## Out of Scope

- Automatic differentiation
- Inequality constraints
- Global optimality guarantees
- Sparse matrix kernels
- Parallel / multi-threaded solvers

## References

- Moré, J. J., "The Levenberg-Marquardt Algorithm: Implementation and Theory," in *Numerical Analysis*, Lecture Notes in Mathematics 630, 1978. DOI: https://doi.org/10.1007/BFb0067700
- Lourakis, M. I. A., Levenberg-Marquardt overview. PDF: http://users.ics.forth.gr/lourakis/levmar/levmar.pdf
- Nocedal, J. & Wright, S. J., *Numerical Optimization* (2nd ed.), 2006

## License

MIT

## Contributing

Contributions are welcome. Follow `CODING_RULES.md` in this repository when submitting pull requests (contributor guide; not shipped on npm).
