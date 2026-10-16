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
| Equality constraints \(c(p,x)=0\) | Adjoint GD, Adjoint BFGS, Constrained GN/LM | cost/residual + `constraint(p,x)` |

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

## Optional LM linear solvers

LM accepts a prepared synchronous `linearSolver`. Omit it to keep the existing
ml-matrix Cholesky/LU behavior, or pass `createMlMatrixLinearSolver()` explicitly.
Residuals, Jacobians, damping, acceptance, callbacks and stopping tests stay in
the same JavaScript LM loop.

`createKiboLinearSolver` adapts the existing kibo WASM LLT facade. It takes the
already damped matrix and signed right-hand side; it does not apply damping a
second time. A numerical LLT failure uses the existing ml-matrix LU fallback.
`statistics` reports actual WASM calls, successes, fallback attempts/reasons and
conversion/backend-boundary timings. Configuration, ABI, shape, non-finite
input/output and lifetime errors throw `LinearSolverConfigurationError`; changing
lambda cannot repair those errors.

The optional integration is available from a source checkout. Normal package
imports and default LM need no WASM asset, initialization, fetch, or Docker.
To generate the external assets, use Git and Docker, with Node 24.21.0 for the
validated kibo execution environment:

```sh
npm ci
npm run prepare:kibo
npm run build
npm run example:lm:kibo
npm run test:kibo:node
```

Preparation uses the source SHA and image digest fixed in
`integration/kibo/toolchains.json`, verifies a clean source, and records compiler,
source tree and asset hashes in `.cache/kibo/manifest.json`. Generated assets
include kibo's LICENSE/NOTICE. No kibo npm package or public release is required.

```js
import { createKiboLinearSolver, levenbergMarquardt } from './dist/index.js';
import createModule from './.cache/kibo/kibo.mjs';
import { LinearAlgebra } from './.cache/kibo/facade.mjs';

const module = await createModule(); // Initialize once before synchronous LM.
const linearSolver = createKiboLinearSolver(
  new LinearAlgebra(module, initialParameters.length, initialParameters.length)
);
try {
  const result = levenbergMarquardt(initialParameters, residual, { linearSolver });
  console.log(result.finalParameters, linearSolver.statistics);
} finally {
  linearSolver.dispose();
}
```

Give each adapter a dedicated facade; do not resize/share it while in use. The
adapter owns that facade, reuses its WASM workspace, and releases it on dispose.
Returned solution copies survive subsequent calls, memory growth and disposal.
The prepared C++ solve's allocation guarantee does not cover JS conversion/copy
or module setup. `inputPayloadBytes`/`outputPayloadBytes` count logical payload,
not every JS allocation or facade-internal copy. Backend-boundary time includes
those facade copies and any LU fallback; it is not isolated C++ kernel time.

`npm run test:kibo:browser` runs HTTP-served browser-bundle checks with the fixed
Playwright/Chromium versions (use the browser Docker image from the toolchain
file, or install its matching browser). The checks execute line, exponential and
analytic-Jacobian fits in both backends, plus fatal boundary errors, injected
numeric failures through public LM, and resource lifetime checks. Result files
include numopt source/bundle hashes and the external WASM manifest. The
validated browser is Chromium; other browser/Node versions are not covered by
this opt-in integration's evidence.

`npm run benchmark:kibo` manually collects five warmups and 30 samples for both
backends, including a fixed, well-conditioned 128x32 problem. Raw samples,
median/IQR, callback/solve counts and provenance go to ignored
`benchmark-results/kibo/`. Initialization and setup are reported separately from
copy-inclusive LM time. CI validates correctness and lifetime, without a speed
threshold. Packing the default package remains independent of generated assets.

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

Manual benchmarks (not CI):

- `npm run benchmark:paper` — paper-grade comparison of every public solver class. Warmup 1 run is discarded; deterministic solvers repeat 7 times (median and IQR); CMA-ES uses 5 seeds (warmup seed 1, timed seeds 1–5). Success is parameter error (and constraint norm when constrained), never `result.converged`. GD / BFGS / L-BFGS / Adjoint GD / Adjoint BFGS use analytical derivatives; GN / LM / Constrained / Penalty use numeric Jacobians. Rosenbrock CMA-ES uses IPOP; Rosenbrock GD may hit a 10000-iteration cap and still succeed on parameter error. Primary metrics are evaluation counts and success; wall-clock is secondary and machine-dependent. Writes `benchmark-results/paper-benchmark.md` and `.json` (gitignored). Hypothesis failures set exit code 1; do not commit the generated numbers.
- `npm run benchmark:constrained`, `npm run benchmark:curve-bending` — informal single-run smokes.

Full signatures and options: [TypeDoc API reference](https://takuto-na.github.io/numopt-js/).

## Convergence Options (quick map)

- **GD / BFGS / L-BFGS / GN / Constrained GN / Adjoint GD / Adjoint BFGS**: `tolerance`
- **LM / Constrained LM**: `tolGradient`, `tolStep`, `tolResidual` (or `tolerance` as a fallback when those are omitted)
- **CMA-ES**: `functionTolerance`, `parameterTolerance`, `targetCost`, `maxFunctionEvaluations`
- **Adjoint (ill-conditioned ∂c/∂x)**: `regularization`

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
