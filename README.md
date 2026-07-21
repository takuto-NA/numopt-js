# numopt-js

A flexible numerical optimization library for JavaScript/TypeScript that works smoothly in browsers. This library addresses the lack of flexible continuous optimization libraries for JavaScript that work well in browser environments.

## Documentation

- **API Reference (GitHub Pages)**: https://takuto-na.github.io/numopt-js/
- **Source Repository**: https://github.com/takuto-NA/numopt-js

## Features

- **Gradient Descent**: Simple, robust optimization algorithm with line search support
- **Line Search**: Backtracking line search with Armijo condition for optimal step sizes (following Nocedal & Wright, *Numerical Optimization* (2nd ed.), Algorithm 3.1)
- **Strong Wolfe Line Search**: Robust line search satisfying Strong Wolfe conditions (recommended for quasi-Newton methods)
- **BFGS**: Quasi-Newton method that updates a dense inverse Hessian approximation (unconstrained smooth optimization)
- **L-BFGS**: Memory-efficient quasi-Newton method (two-loop recursion) for larger parameter counts
- **CMA-ES**: Derivative-free black-box optimization using a covariance-adapting search distribution (unconstrained)
- **Gauss-Newton Method**: Efficient method for nonlinear least squares problems
- **Levenberg-Marquardt Algorithm**: Robust algorithm combining Gauss-Newton with damping
- **Constrained Gauss-Newton**: Efficient constrained nonlinear least squares using effective Jacobian
- **Constrained Levenberg-Marquardt**: Robust constrained nonlinear least squares with damping
- **Adjoint Method**: Efficient constrained optimization using adjoint variables (solves only one linear system per iteration instead of parameterCount systems)
- **Numerical Differentiation**: Automatic gradient and Jacobian computation via finite differences
- **Browser-Compatible**: Works seamlessly in modern browsers
- **TypeScript-First**: Full TypeScript support with comprehensive type definitions
- **Debug-Friendly**: Progress callbacks, verbose logging, and detailed diagnostics

## Requirements

- Node.js >= 18.0.0
- Modern browsers with ES2020 support (for browser builds)

## Installation

```bash
npm install numopt-js
```

## Start Here (Pick Your Path)

- **Minimize a scalar cost** (smooth unconstrained optimization): use **Gradient Descent** (`cost: (p) => number`, `grad: (p) => Float64Array`). Start at [Gradient Descent](#gradient-descent).
- **Minimize a scalar cost (faster than GD for many problems)**: use **BFGS** or **L-BFGS** (`cost: (p) => number`, `grad: (p) => Float64Array`). Start at [BFGS / L-BFGS](#bfgs--l-bfgs).
- **Minimize a scalar cost (black-box, no gradient)**: use **CMA-ES** (`cost: (p) => number`). Start at [CMA-ES](#cma-es-black-box-optimization).
- **Fit a model with residuals** (nonlinear least squares): use **Levenberg–Marquardt** or **Gauss–Newton** (`residual: (p) => Float64Array`, optional `jacobian: (p) => Matrix`). Start at [Levenberg-Marquardt](#levenberg-marquardt-nonlinear-least-squares).
- **Equality-constrained problems** \(c(p, x) = 0\): use **Adjoint** / **Constrained GN/LM** (`constraint: (p, x) => Float64Array`). Start at [Adjoint Method](#adjoint-method-constrained-optimization).
- **Browser usage**: start at [Browser Usage](#browser-usage).

**Why `Float64Array`?** This library uses `Float64Array` for predictable numeric performance. You can always convert from normal arrays with `new Float64Array([1, 2, 3])` (more details below).

## Cost Function vs Residual Function (Important)

- **Cost function**: `cost(p) -> number` (used by `gradientDescent`)
- **Residual function**: `residual(p) -> Float64Array` (used by `gaussNewton` / `levenbergMarquardt`), where the library minimizes \(f(p) = 1/2 \|r(p)\|^2\)

## Result Object (What to Look At)

Most algorithms return a `result` with these fields:

- **Common (all algorithms)**: `finalParameters`, `converged`, `iterations`, `finalCost`
- **Gradient Descent**: `finalGradientNorm`, `usedLineSearch`
- **BFGS / L-BFGS**: `finalGradientNorm`
- **CMA-ES**: `functionEvaluations`, `finalStepSize`, `finalMaxStdDev`
- **Gauss-Newton / Levenberg–Marquardt**: `finalResidualNorm` (and LM also has `finalLambda`)
- **Constrained algorithms / Adjoint**: `finalStates`, `finalConstraintNorm`

Note: `result.parameters` is a deprecated alias of `result.finalParameters` and will be removed in a future release.

## 60-second Quick Start (Node)

Pick **one** of the following and run it.

### ESM (recommended)

Create `quick.mjs`:

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

## BFGS / L-BFGS

Use these for smooth unconstrained problems when you can provide a gradient.

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

console.log(bfgsResult.finalParameters, lbfgsResult.finalParameters);
```

Run:

```bash
node quick.mjs
```

### CommonJS

Create `quick.cjs`:

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

Run:

```bash
node quick.cjs
```

## CMA-ES (Black-box Optimization)

Use CMA-ES when your cost function is a **black box** and you **don’t have a gradient**.

Two important inputs:
- `initialStepSize` (sigma0): your initial error guess / search radius
- `randomSeed`: set this for reproducible runs (recommended for testing and debugging)
- `restartStrategy`: use `"ipop"` for multi-modal problems (λ doubles per restart)
- `profiling`: enable lightweight timing breakdown in the result

```js
import { cmaEs } from 'numopt-js';

const sphere = (params) => params.reduce((sum, v) => sum + v * v, 0);

const result = cmaEs(new Float64Array([10, -7, 3, 5]), sphere, {
  maxIterations: 200,
  populationSize: 20,
  initialStepSize: 2.0,
  randomSeed: 123456,
  targetCost: 1e-10,
  restartStrategy: "none",
  profiling: true,
});

console.log(result.finalParameters, result.finalCost, result.stopReason, result.profiling);
```

### IPOP (Restart Strategy)

For multi-modal problems (e.g., Rastrigin), use IPOP restarts:

```js
const result = cmaEs(new Float64Array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]), sphere, {
  maxIterations: 1200,
  populationSize: 20,
  initialStepSize: 2.5,
  randomSeed: 123456,
  restartStrategy: "ipop",
});
```

## Node Usage (ESM + CommonJS)

numopt-js supports both ESM (`import`) and CommonJS (`require`) in Node.js.

### ESM

Note:
- If your project is CommonJS (default), use a `.mjs` file to run ESM.
- If you want ESM by default, set `package.json` to `"type": "module"` (then `.js` is treated as ESM).

```typescript
import { gradientDescent } from 'numopt-js';
```

### CommonJS

Note:
- If your project is ESM (`"type": "module"`), use a `.cjs` file to run CommonJS.

```js
const { gradientDescent } = require('numopt-js');
```

## Browser Usage

numopt-js is designed to work seamlessly in browser environments. The library automatically provides a browser-optimized bundle that includes all dependencies.

**Important**:
- **Don’t use `file://`** for the import-maps / direct-import examples. Serve your files via a local static server (for example: `npx serve`, `python -m http.server`, or Vite) so ES modules load correctly.
- **SSR frameworks (Next.js, etc.)**: run numopt-js on the client side. If you hit SSR errors, move the code into a client component (`"use client"`) or dynamically import it with SSR disabled.

### Option 1: Bundler (Recommended)

If you're using a bundler (Vite/Webpack/Rollup), just import from the package and the bundler will resolve the browser build via `package.json` exports.

```typescript
import { gradientDescent } from 'numopt-js';
```

### Option 2: Import Maps

If you're using import maps (no bundler), map `numopt-js` to the browser bundle:

```html
<script type="importmap">
{
  "imports": {
    "numopt-js": "./node_modules/numopt-js/dist/index.browser.js"
  }
}
</script>
<script type="module">
  import { gradientDescent } from 'numopt-js';
  
  // Your code here
</script>
```

### Option 3: Direct file import (no bundler, no import maps)

Import the browser bundle by path. (In this mode, you **cannot** use the bare specifier `numopt-js`.)

```html
<script type="module">
  import { gradientDescent } from "./node_modules/numopt-js/dist/index.browser.js";
  // Your code here
</script>
```

### Troubleshooting

**Problem**: `ReferenceError: exports is not defined` when using in browser

**Solution**: Make sure you're using `dist/index.browser.js` instead of `dist/index.js`. The browser bundle includes all dependencies and is pre-configured for browser environments.

**Problem**: Module not found errors

**Solution**: 
- Ensure you're using a modern bundler that supports `package.json` exports
- For direct browser usage, use import maps or explicitly import from `dist/index.browser.js`
- Check that your build tool supports ES modules

## Examples

After `npm install`, run examples with `npm run <script>`. Recommended order:

1. `npm run example:gradient` — scalar cost + gradient
2. `npm run example:rosenbrock` — line search on a classic non-convex problem
3. `npm run example:lm` — residual-based least squares
4. `npm run example:gauss-newton` — nonlinear least squares without damping
5. `npm run example:cma-es` — derivative-free black-box optimization
6. `npm run example:constrained-gauss-newton` — equality-constrained least squares
7. `npm run example:constrained-lm` — damped constrained least squares
8. `npm run example:adjoint` — constrained optimization with states \(x\) and parameters \(p\)

Also available: `example:line-search`, `example:layout-toy`, `example:adjoint-advanced`.

Optional manual benchmarks (not CI): `npm run benchmark:constrained`, `npm run benchmark:curve-bending`.

### Gradient Descent

Based on standard steepest-descent with backtracking line search (Nocedal & Wright, "Numerical Optimization" 2/e, Ch. 2; Boyd & Vandenberghe, "Convex Optimization", Sec. 9.3).

```typescript
import { gradientDescent } from 'numopt-js';

// Define cost function and gradient
const costFunction = (params: Float64Array) => {
  return params[0] * params[0] + params[1] * params[1];
};

const gradientFunction = (params: Float64Array) => {
  return new Float64Array([2 * params[0], 2 * params[1]]);
};

// Optimize
const initialParams = new Float64Array([5.0, -3.0]);
const result = gradientDescent(initialParams, costFunction, gradientFunction, {
  maxIterations: 1000,
  tolerance: 1e-6,
  useLineSearch: true
});

console.log('Optimized parameters:', result.finalParameters);
console.log('Final cost:', result.finalCost);
console.log('Converged:', result.converged);
```

**Using Result Formatter**: For better formatted output, use the built-in result formatter:

```typescript
import { gradientDescent, printGradientDescentResult } from 'numopt-js';

const result = gradientDescent(initialParams, costFunction, gradientFunction, {
  maxIterations: 1000,
  tolerance: 1e-6,
  useLineSearch: true
});

// Automatically formats and prints the result
printGradientDescentResult(result);
```

### Levenberg-Marquardt (Nonlinear Least Squares)

```typescript
import { levenbergMarquardt } from 'numopt-js';

// Define residual function
const residualFunction = (params: Float64Array) => {
  const [a, b] = params;
  const residuals = new Float64Array(xData.length);
  
  for (let i = 0; i < xData.length; i++) {
    const predicted = a * xData[i] + b;
    residuals[i] = predicted - yData[i];
  }
  
  return residuals;
};

// Optimize (with automatic numerical Jacobian)
const initialParams = new Float64Array([0, 0]);
const result = levenbergMarquardt(initialParams, residualFunction, {
  useNumericJacobian: true,
  maxIterations: 100,
  tolGradient: 1e-6
});

console.log('Optimized parameters:', result.finalParameters);
console.log('Final residual norm:', result.finalResidualNorm);
```

**Using Result Formatter**:

```typescript
import { levenbergMarquardt, printLevenbergMarquardtResult } from 'numopt-js';

const result = levenbergMarquardt(initialParams, residualFunction, {
  useNumericJacobian: true,
  maxIterations: 100,
  tolGradient: 1e-6
});

printLevenbergMarquardtResult(result);
```

### With User-Provided Jacobian

```typescript
import { levenbergMarquardt } from 'numopt-js';
import { Matrix } from 'ml-matrix';

const jacobianFunction = (params: Float64Array) => {
  // Compute analytical Jacobian
  return new Matrix(/* ... */);
};

const result = levenbergMarquardt(initialParams, residualFunction, {
  jacobian: jacobianFunction, // User-provided Jacobian in options
  maxIterations: 100
});
```

### Numerical Differentiation

If you don't have analytical gradients or Jacobians, you can use numerical differentiation:

#### Option 1: Helper Functions (Recommended)

The easiest way to use numerical differentiation is with the helper functions:

```typescript
import { gradientDescent, createFiniteDiffGradient } from 'numopt-js';

const costFn = (params: Float64Array) => {
  return Math.pow(params[0] - 3, 2) + Math.pow(params[1] - 2, 2);
};

// Create a gradient function automatically
const gradientFn = createFiniteDiffGradient(costFn);

const result = gradientDescent(
  new Float64Array([0, 0]),
  costFn,
  gradientFn,  // No parameter order confusion!
  { maxIterations: 100, tolerance: 1e-6 }
);
```

#### Option 2: Direct Usage

You can also use `finiteDiffGradient` directly:

```typescript
import { gradientDescent, finiteDiffGradient } from 'numopt-js';

const costFn = (params: Float64Array) => {
  return Math.pow(params[0] - 3, 2) + Math.pow(params[1] - 2, 2);
};

const result = gradientDescent(
  new Float64Array([0, 0]),
  costFn,
  (params) => finiteDiffGradient(params, costFn),  // ⚠️ Note: params first!
  { maxIterations: 100, tolerance: 1e-6 }
);
```

**Important**: When using `finiteDiffGradient` directly, note the parameter order:
- ✅ Correct: `finiteDiffGradient(params, costFn)`
- ❌ Wrong: `finiteDiffGradient(costFn, params)`

#### Custom Step Size

Both approaches support custom step sizes for the finite difference approximation:

```typescript
// With helper function
const gradientFn = createFiniteDiffGradient(costFn, { stepSize: 1e-8 });

// Direct usage
const gradient = finiteDiffGradient(params, costFn, { stepSize: 1e-8 });
```

**Practical tips (finite differences)**:
- **Scale your parameters** so typical values are around \(O(1)\). If one parameter is \(10^{-6}\) and another is \(10^{6}\), a single global `stepSize` will often fail.
- If you work in physical units, consider **normalizing inputs/parameters** first, then convert back after optimization.

### Adjoint Method (Constrained Optimization)

The adjoint method efficiently solves constrained optimization problems by solving for an adjoint variable λ instead of explicitly inverting matrices. This requires solving only one linear system per iteration, making it much more efficient than naive approaches.

**Mathematical background**: For constraint `c(p, x) = 0`, the method computes `df/dp = ∂f/∂p - λ^T ∂c/∂p` where λ solves `(∂c/∂x)^T λ = (∂f/∂x)^T`.

**Constrained Least Squares**: For residual functions `r(p, x)` with constraints `c(p, x) = 0`, the library provides constrained Gauss-Newton and Levenberg-Marquardt methods. These use the effective Jacobian `J_eff = r_p - r_x C_x^+ C_p` to capture constraint effects, enabling quadratic convergence near the solution while maintaining constraint satisfaction.

```typescript
import { adjointGradientDescent } from 'numopt-js';

// Define cost function: f(p, x) = p² + x²
const costFunction = (p: Float64Array, x: Float64Array) => {
  return p[0] * p[0] + x[0] * x[0];
};

// Define constraint: c(p, x) = p + x - 1 = 0
const constraintFunction = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([p[0] + x[0] - 1.0]);
};

// Initial values (should satisfy constraint: c(p₀, x₀) = 0)
const initialP = new Float64Array([2.0]);
const initialX = new Float64Array([-1.0]); // 2 + (-1) - 1 = 0

// Optimize
const result = adjointGradientDescent(
  initialP,
  initialX,
  costFunction,
  constraintFunction,
  {
    maxIterations: 100,
    tolerance: 1e-6,
    useLineSearch: true,
    logLevel: 'DEBUG' // Enable detailed iteration logging
  }
);

console.log('Optimized parameters:', result.finalParameters);
console.log('Final states:', result.finalStates);
console.log('Final cost:', result.finalCost);
console.log('Constraint norm:', result.finalConstraintNorm);
```

**With Residual Functions**: The method also supports residual functions `r(p, x)` where `f = 1/2 r^T r`:

```typescript
// Residual function: r(p, x) = [p - 0.5, x - 0.5]
const residualFunction = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([p[0] - 0.5, x[0] - 0.5]);
};

const result = adjointGradientDescent(
  initialP,
  initialX,
  residualFunction, // Can use residual function directly
  constraintFunction,
  { maxIterations: 100, tolerance: 1e-6 }
);
```

**With Analytical Derivatives**: For better performance, you can provide analytical partial derivatives:

```typescript
import { Matrix } from 'ml-matrix';

const result = adjointGradientDescent(
  initialP,
  initialX,
  costFunction,
  constraintFunction,
  {
    dfdp: (p: Float64Array, x: Float64Array) => new Float64Array([2 * p[0]]),
    dfdx: (p: Float64Array, x: Float64Array) => new Float64Array([2 * x[0]]),
    dcdp: (p: Float64Array, x: Float64Array) => new Matrix([[1]]),
    dcdx: (p: Float64Array, x: Float64Array) => new Matrix([[1]]),
    maxIterations: 100
  }
);
```

### Constrained Gauss-Newton (Constrained Nonlinear Least Squares)

For constrained nonlinear least squares problems, use the constrained Gauss-Newton method:

```typescript
import { constrainedGaussNewton } from 'numopt-js';

// Define residual function: r(p, x) = [p - 0.5, x - 0.5]
const residualFunction = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([p[0] - 0.5, x[0] - 0.5]);
};

// Define constraint: c(p, x) = p + x - 1 = 0
const constraintFunction = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([p[0] + x[0] - 1.0]);
};

// Initial values (should satisfy constraint: c(p₀, x₀) = 0)
const initialP = new Float64Array([2.0]);
const initialX = new Float64Array([-1.0]); // 2 + (-1) - 1 = 0

// Optimize
const result = constrainedGaussNewton(
  initialP,
  initialX,
  residualFunction,
  constraintFunction,
  {
    maxIterations: 100,
    tolerance: 1e-6
  }
);

console.log('Optimized parameters:', result.finalParameters);
console.log('Final states:', result.finalStates);
console.log('Final cost:', result.finalCost);
console.log('Constraint norm:', result.finalConstraintNorm);
```

### Constrained Levenberg-Marquardt (Robust Constrained Least Squares)

For more robust constrained optimization, use the constrained Levenberg-Marquardt method:

```typescript
import { constrainedLevenbergMarquardt } from 'numopt-js';

const result = constrainedLevenbergMarquardt(
  initialP,
  initialX,
  residualFunction,
  constraintFunction,
  {
    maxIterations: 100,
    tolGradient: 1e-6,
    tolStep: 1e-6,
    tolResidual: 1e-6,
    lambdaInitial: 1e-3,
    lambdaFactor: 10.0
  }
);
```

**With Analytical Derivatives**: For better performance, provide analytical partial derivatives:

```typescript
import { Matrix } from 'ml-matrix';

const result = constrainedGaussNewton(
  initialP,
  initialX,
  residualFunction,
  constraintFunction,
  {
    drdp: (p: Float64Array, x: Float64Array) => new Matrix([[1], [0]]),
    drdx: (p: Float64Array, x: Float64Array) => new Matrix([[0], [1]]),
    dcdp: (p: Float64Array, x: Float64Array) => new Matrix([[1]]),
    dcdx: (p: Float64Array, x: Float64Array) => new Matrix([[1]]),
    maxIterations: 100
  }
);
```

### Gradient Descent

```typescript
function gradientDescent(
  initialParameters: Float64Array,
  costFunction: CostFn,
  gradientFunction: GradientFn,
  options?: GradientDescentOptions
): GradientDescentResult
```

### BFGS / L-BFGS

```typescript
function bfgs(
  initialParameters: Float64Array,
  costFunction: CostFn,
  gradientFunction: GradientFn,
  options?: BfgsOptions
): OptimizationResult

function lbfgs(
  initialParameters: Float64Array,
  costFunction: CostFn,
  gradientFunction: GradientFn,
  options?: LbfgsOptions
): OptimizationResult
```

### CMA-ES

```typescript
function cmaEs(
  initialParameters: Float64Array,
  costFunction: CostFn,
  options?: CmaEsOptions
): CmaEsResult
```

### Gauss-Newton

```typescript
function gaussNewton(
  initialParameters: Float64Array,
  residualFunction: ResidualFn,
  options?: GaussNewtonOptions
): OptimizationResult
```

### Levenberg-Marquardt

```typescript
function levenbergMarquardt(
  initialParameters: Float64Array,
  residualFunction: ResidualFn,
  options?: LevenbergMarquardtOptions
): LevenbergMarquardtResult
```

### Line Search

```typescript
function backtrackingLineSearch(
  costFunction: CostFn,
  gradientFunction: GradientFn,
  currentParameters: Float64Array,
  searchDirection: Float64Array,
  options?: LineSearchOptions
): number

function strongWolfeLineSearch(
  costFunction: CostFn,
  gradientFunction: GradientFn,
  currentParameters: Float64Array,
  searchDirection: Float64Array,
  options?: StrongWolfeLineSearchOptions
): number
```

### Adjoint Gradient Descent

```typescript
function adjointGradientDescent(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options?: AdjointGradientDescentOptions
): AdjointGradientDescentResult
```

### Constrained Gauss-Newton

```typescript
function constrainedGaussNewton(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options?: ConstrainedGaussNewtonOptions
): ConstrainedGaussNewtonResult
```

### Constrained Levenberg-Marquardt

```typescript
function constrainedLevenbergMarquardt(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options?: ConstrainedLevenbergMarquardtOptions
): ConstrainedLevenbergMarquardtResult
```

### Options

All algorithms support common options:

- `maxIterations?: number` - Maximum number of iterations (default: 1000)
- `tolerance?: number` - Convergence tolerance (default: 1e-6)
- `onIteration?: (iteration: number, cost: number, params: Float64Array) => void` - Progress callback
- `verbose?: boolean` - Enable verbose logging (default: false)

#### Gradient Descent Options

- `stepSize?: number` - Fixed step size (learning rate). If not provided, line search is used (default: undefined, uses line search)
- `useLineSearch?: boolean` - Use line search to determine optimal step size (default: true)

#### Levenberg-Marquardt Options

- `jacobian?: JacobianFn` - Analytical Jacobian function (if provided, used instead of numerical differentiation)
- `useNumericJacobian?: boolean` - Use numerical differentiation for Jacobian (default: true)
- `jacobianStep?: number` - Step size for numerical Jacobian computation (default: 1e-6)
- `lambdaInitial?: number` - Initial damping parameter (default: 1e-3)
- `lambdaFactor?: number` - Factor for updating lambda (default: 10.0)
- `tolGradient?: number` - Tolerance for gradient norm convergence (default: 1e-6)
- `tolStep?: number` - Tolerance for step size convergence (default: 1e-6)
- `tolResidual?: number` - Tolerance for residual norm convergence (default: 1e-6)

**Levenberg-Marquardt References**

- Moré, J. J., "The Levenberg-Marquardt Algorithm: Implementation and Theory," in *Numerical Analysis*, Lecture Notes in Mathematics 630, 1978. DOI: https://doi.org/10.1007/BFb0067700
- Lourakis, M. I. A., "A Brief Description of the Levenberg-Marquardt Algorithm," 2005 tutorial. PDF: https://users.ics.forth.gr/lourakis/levmar/levmar.pdf

#### Gauss-Newton Options

- `jacobian?: JacobianFn` - Analytical Jacobian function (if provided, used instead of numerical differentiation)
- `useNumericJacobian?: boolean` - Use numerical differentiation for Jacobian (default: true)
- `jacobianStep?: number` - Step size for numerical Jacobian computation (default: 1e-6)

#### Adjoint Gradient Descent Options

- `dfdp?: (p: Float64Array, x: Float64Array) => Float64Array` - Analytical partial derivative ∂f/∂p (optional)
- `dfdx?: (p: Float64Array, x: Float64Array) => Float64Array` - Analytical partial derivative ∂f/∂x (optional)
- `dcdp?: (p: Float64Array, x: Float64Array) => Matrix` - Analytical partial derivative ∂c/∂p (optional)
- `dcdx?: (p: Float64Array, x: Float64Array) => Matrix` - Analytical partial derivative ∂c/∂x (optional)
- `stepSizeP?: number` - Step size for numerical differentiation w.r.t. parameters (default: 1e-6)
- `stepSizeX?: number` - Step size for numerical differentiation w.r.t. states (default: 1e-6)
- `constraintTolerance?: number` - Tolerance for constraint satisfaction check (default: 1e-6)

#### Constrained Gauss-Newton Options

- `drdp?: (p: Float64Array, x: Float64Array) => Matrix` - Analytical partial derivative ∂r/∂p (optional)
- `drdx?: (p: Float64Array, x: Float64Array) => Matrix` - Analytical partial derivative ∂r/∂x (optional)
- `dcdp?: (p: Float64Array, x: Float64Array) => Matrix` - Analytical partial derivative ∂c/∂p (optional)
- `dcdx?: (p: Float64Array, x: Float64Array) => Matrix` - Analytical partial derivative ∂c/∂x (optional)
- `stepSizeP?: number` - Step size for numerical differentiation w.r.t. parameters (default: 1e-6)
- `stepSizeX?: number` - Step size for numerical differentiation w.r.t. states (default: 1e-6)
- `constraintTolerance?: number` - Tolerance for constraint satisfaction check (default: 1e-6)

#### Constrained Levenberg-Marquardt Options

Extends `ConstrainedGaussNewtonOptions` with:
- `lambdaInitial?: number` - Initial damping parameter (default: 1e-3)
- `lambdaFactor?: number` - Factor for updating lambda (default: 10.0)
- `tolGradient?: number` - Tolerance for gradient norm convergence (default: 1e-6)
- `tolStep?: number` - Tolerance for step size convergence (default: 1e-6)
- `tolResidual?: number` - Tolerance for residual norm convergence (default: 1e-6)

**Note**: The constraint function `c(p, x)` does not need to return a vector with the same length as the state vector `x`. The constrained solvers support both square and non-square constraint Jacobians (overdetermined and underdetermined systems) by solving the relevant linear systems in a least-squares sense (with regularization when needed). If you see instability, try scaling/normalizing your states/constraints.

#### Numerical Differentiation Options

- `stepSize?: number` - Step size for finite difference approximation (default: 1e-6)

## Result Formatting

The library provides helper functions for formatting and displaying optimization results in a consistent, user-friendly manner. These functions replace repetitive `console.log` statements and provide better readability.

### Basic Usage

```typescript
import { gradientDescent, printGradientDescentResult } from 'numopt-js';

const result = gradientDescent(initialParams, costFunction, gradientFunction, {
  maxIterations: 1000,
  tolerance: 1e-6
});

// Print formatted result
printGradientDescentResult(result);
```

### Available Formatters

- `printOptimizationResult()` - For basic `OptimizationResult`
- `printGradientDescentResult()` - For `GradientDescentResult` (includes line search info)
- `printLevenbergMarquardtResult()` - For `LevenbergMarquardtResult` (includes lambda)
- `printCmaEsResult()` - For `CmaEsResult` (includes evaluations / step size)
- `printConstrainedGaussNewtonResult()` - For constrained optimization results
- `printConstrainedLevenbergMarquardtResult()` - For constrained LM results
- `printAdjointGradientDescentResult()` - For adjoint method results
- `printResult()` - Type-safe overloaded function that works with any result type
- Matching `format*` helpers return the same text without printing

### Customization Options

All formatters accept an optional `ResultFormatterOptions` object:

```typescript
import { printOptimizationResult } from 'numopt-js';

const startTime = performance.now();
const result = /* ... optimization ... */;
const elapsedTime = performance.now() - startTime;

printOptimizationResult(result, {
  showSectionHeaders: true,      // Show "=== Optimization Results ===" header
  showExecutionTime: true,        // Include execution time
  elapsedTimeMs: elapsedTime,    // Execution time in milliseconds
  maxParametersToShow: 10,        // Max parameters to display before truncating
  parameterPrecision: 6,         // Decimal places for parameters
  costPrecision: 8,              // Decimal places for cost/norms
  constraintPrecision: 10        // Decimal places for constraint violations
});
```

### Formatting Strings Instead of Printing

If you need the formatted string instead of printing to console:

```typescript
import { formatOptimizationResult } from 'numopt-js';

const formattedString = formatOptimizationResult(result);
// Use formattedString as needed (e.g., save to file, send to API, etc.)
```

### Automatic Parameter Formatting

The formatters automatically handle parameter arrays:
- **Small arrays (≤3 elements)**: Displayed individually with labels (`p = 1.0, x = 2.0`)
- **Medium arrays (4-10 elements)**: Displayed as array (`[1.0, 2.0, 3.0, ...]`)
- **Large arrays (>10 elements)**: Truncated with "... and N more" (`[1.0, 2.0, ..., ... and 15 more]`)

## References

- Moré, J. J., "The Levenberg-Marquardt Algorithm: Implementation and Theory," in *Numerical Analysis*, Lecture Notes in Mathematics 630, 1978. DOI: https://doi.org/10.1007/BFb0067700
- Lourakis, M. I. A., "A Brief Description of the Levenberg-Marquardt Algorithm," 2005 tutorial. PDF: http://www.ics.forth.gr/~lourakis/publ/2005/LM.pdf
- Nocedal, J. & Wright, S. J., "Numerical Optimization" (2nd ed.), Chapter 12 (constrained optimization), 2006

## Out of Scope (Current)

- Automatic differentiation
- Inequality constraints
- Global optimality guarantees
- Sparse matrix kernels
- Parallel / multi-threaded solvers

## Type Definitions

### Why Float64Array?

This library uses `Float64Array` instead of regular JavaScript arrays for:
- **Performance**: Float64Array provides better performance for numerical computations
- **Memory efficiency**: More memory-efficient storage for large parameter vectors
- **Type safety**: Ensures all values are 64-bit floating-point numbers

To convert from regular arrays:
```typescript
const regularArray = [1.0, 2.0, 3.0];
const float64Array = new Float64Array(regularArray);
```

### Why Matrix from ml-matrix?

The library uses `Matrix` from the `ml-matrix` package for Jacobian matrices because:
- **Efficient matrix operations**: Provides optimized matrix multiplication and linear algebra operations
- **Well-tested**: Mature library with comprehensive matrix operations
- **Browser-compatible**: Works seamlessly in browser environments

To create a Matrix from a 2D array:
```typescript
import { Matrix } from 'ml-matrix';
const matrix = new Matrix([[1, 2], [3, 4]]);
```

## Troubleshooting

### Common Errors and Solutions

#### Error: "Jacobian computation is required but not provided"

**Problem**: You're using `levenbergMarquardt` or `gaussNewton` without providing a Jacobian function and numerical Jacobian is disabled.

**Solutions**:
1. Enable numerical Jacobian (default behavior):
   ```typescript
   levenbergMarquardt(params, residualFn, { useNumericJacobian: true })
   ```

2. Provide an analytical Jacobian function:
   ```typescript
   const jacobianFn = (params: Float64Array) => {
     // Your Jacobian computation
     return new Matrix(/* ... */);
   };
   levenbergMarquardt(params, residualFn, { jacobian: jacobianFn, ...options })
   ```

#### Algorithm doesn't converge

**Possible causes**:
- Initial parameters are too far from the solution
- Tolerance is too strict
- Maximum iterations too low
- Step size (for gradient descent) is inappropriate

**Solutions**:
1. Try different initial parameters
2. Increase `maxIterations`
3. Adjust tolerance values (`tolerance`, `tolGradient`, `tolStep`, `tolResidual`)
4. For gradient descent, enable line search (`useLineSearch: true`) or adjust `stepSize`
5. Enable verbose logging (`verbose: true`) to see what's happening

#### Singular matrix error (Gauss-Newton)

**Problem**: The Jacobian matrix is singular or ill-conditioned, making the normal equations unsolvable.

**Solutions**:
1. Use Levenberg-Marquardt instead (handles singular matrices better)
2. Check your residual function for numerical issues
3. Try different initial parameters
4. Increase numerical Jacobian step size (`jacobianStep`)

#### Singular matrix error (Constrained Gauss-Newton)

**Problem**: The effective Jacobian `J_eff^T J_eff` is singular or ill-conditioned.

**Solutions**:
1. Use Constrained Levenberg-Marquardt instead (handles singular matrices better with damping)
2. Check that constraint Jacobian `∂c/∂x` is well-conditioned
3. Verify initial states satisfy constraints approximately
4. Try different initial parameters and states

#### Singular matrix error (Adjoint Method)

**Problem**: The constraint Jacobian `∂c/∂x` is singular or ill-conditioned, making the adjoint equation unsolvable.

**Solutions**:
1. Check that `∂c/∂x` is well-conditioned (if square) or has full rank (if non-square)
3. Verify initial states satisfy the constraint approximately (`c(p₀, x₀) ≈ 0`)
4. Try different initial values that don't make `∂c/∂x` singular
5. For nonlinear constraints, ensure initial values are on the constraint manifold

#### Results don't match expectations

**Check**:
1. Verify your cost/residual function is correct
2. Check that gradient/Jacobian functions are correct (if provided)
3. Try enabling `verbose: true` or `logLevel: 'DEBUG'` to see iteration details
4. Use `onIteration` callback to monitor progress
5. Verify initial parameters are reasonable
6. For adjoint method, ensure initial states satisfy constraints approximately

### Debugging Tips

1. **Enable verbose logging**: Set `verbose: true` to see detailed iteration information
2. **Use progress callbacks**: Use `onIteration` to monitor convergence:
   ```typescript
   const result = gradientDescent(params, costFn, gradFn, {
     onIteration: (iter, cost, params) => {
       console.log(`Iteration ${iter}: cost = ${cost}`);
     }
   });
   ```
3. **Check convergence status**: Always check `result.converged` to see if optimization succeeded
4. **Monitor gradient/residual norms**: Check `finalGradientNorm` or `finalResidualNorm` to understand convergence quality

## License

MIT

## Contributing

Contributions are welcome. Follow `CODING_RULES.md` in this repository when submitting pull requests (it is a contributor guide, not part of the npm package).

