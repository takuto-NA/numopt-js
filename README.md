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
| Equality constraints \(c(p,x)=0\) | Adjoint, Constrained GN/LM | cost/residual + `constraint(p,x)` |

**Why `Float64Array`?** Predictable numeric performance. Convert with `new Float64Array([1, 2, 3])`.

Least-squares solvers minimize \(f(p) = 1/2 \|r(p)\|^2\).

## Result Object

Common fields: `finalParameters`, `converged`, `iterations`, `finalCost`.

- Gradient / BFGS / L-BFGS: `finalGradientNorm`
- CMA-ES: `functionEvaluations`, `finalStepSize`, `stopReason`, optional `profiling`
- GN / LM: `finalResidualNorm` (LM also has `finalLambda`)
- Constrained / Adjoint: `finalStates`, `finalConstraintNorm`

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
5. `npm run example:constrained` — Constrained LM / GN / Adjoint on one problem
6. `npm run example:adjoint` — basic adjoint
7. `npm run example:adjoint-advanced` — harder adjoint cases
8. `npm run example:layout-toy` — small layout toy

Manual benchmarks (not CI): `npm run benchmark:constrained`, `npm run benchmark:curve-bending`.

Full signatures and options: [TypeDoc API reference](https://takuto-na.github.io/numopt-js/).

## Convergence Options (quick map)

- **GD / BFGS / L-BFGS / GN / Constrained GN / Adjoint**: `tolerance`
- **LM / Constrained LM**: `tolGradient`, `tolStep`, `tolResidual`
- **CMA-ES**: `functionTolerance`, `parameterTolerance`, `targetCost`, `maxFunctionEvaluations`
- **Adjoint (ill-conditioned ∂c/∂x)**: `regularization`

Result printing helpers: `printResult` / `formatResult` (and typed variants) — see TypeDoc.

## Troubleshooting

- **Does not converge**: try better initials, raise `maxIterations`, relax tolerances, enable line search for GD/Adjoint, use `logLevel: 'DEBUG'`.
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
