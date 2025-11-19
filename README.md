# numopt-js

A flexible numerical optimization library for JavaScript/TypeScript that works smoothly in browsers. This library addresses the lack of flexible continuous optimization libraries for JavaScript that work well in browser environments.

## Features

- **Gradient Descent**: Simple, robust optimization algorithm with line search support
- **Line Search**: Backtracking line search with Armijo condition for optimal step sizes (following Nocedal & Wright, *Numerical Optimization* (2nd ed.), Algorithm 3.1)
- **Gauss-Newton Method**: Efficient method for nonlinear least squares problems
- **Levenberg-Marquardt Algorithm**: Robust algorithm combining Gauss-Newton with damping
- **Numerical Differentiation**: Automatic gradient and Jacobian computation via finite differences
- **Browser-Compatible**: Works seamlessly in modern browsers
- **TypeScript-First**: Full TypeScript support with comprehensive type definitions
- **Debug-Friendly**: Progress callbacks, verbose logging, and detailed diagnostics

## Installation

```bash
npm install numopt-js
```

## Quick Start

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

console.log('Optimized parameters:', result.parameters);
console.log('Final cost:', result.finalCost);
console.log('Converged:', result.converged);
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

console.log('Optimized parameters:', result.parameters);
console.log('Final residual norm:', result.finalResidualNorm);
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

## API Reference

### Gradient Descent

```typescript
function gradientDescent(
  initialParameters: Float64Array,
  costFunction: CostFn,
  gradientFunction: GradientFn,
  options?: GradientDescentOptions
): GradientDescentResult
```

### Levenberg-Marquardt

```typescript
function levenbergMarquardt(
  initialParameters: Float64Array,
  residualFunction: ResidualFn,
  options?: LevenbergMarquardtOptions
): LevenbergMarquardtResult
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

#### Gauss-Newton Options

- `jacobian?: JacobianFn` - Analytical Jacobian function (if provided, used instead of numerical differentiation)
- `useNumericJacobian?: boolean` - Use numerical differentiation for Jacobian (default: true)
- `jacobianStep?: number` - Step size for numerical Jacobian computation (default: 1e-6)

#### Numerical Differentiation Options

- `stepSize?: number` - Step size for finite difference approximation (default: 1e-6)

## Examples

See the `examples/` directory for complete working examples:

- Gradient descent with Rosenbrock function
- Curve fitting with Levenberg-Marquardt
- Linear and nonlinear regression

To run the examples:

```bash
# Using npm scripts (recommended)
npm run example:gradient
npm run example:rosenbrock
npm run example:lm
npm run example:gauss-newton

# Or directly with tsx
npx tsx examples/gradient-descent-example.ts
npx tsx examples/curve-fitting-lm.ts
npx tsx examples/rosenbrock-optimization.ts
```

## MVP Scope

### Included

- Gradient descent with line search
- Gauss-Newton method
- Levenberg-Marquardt algorithm
- Numerical differentiation (central difference)
- Browser compatibility
- TypeScript support

### Not Included (Future Work)

- Automatic differentiation
- Constraint handling (inequality/equality constraints)
- Global optimization guarantees
- Evolutionary algorithms (CMA-ES, etc.)
- Other optimization algorithms (BFGS, etc.)
- Sparse matrix support
- Parallel computation

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

#### Results don't match expectations

**Check**:
1. Verify your cost/residual function is correct
2. Check that gradient/Jacobian functions are correct (if provided)
3. Try enabling `verbose: true` to see iteration details
4. Use `onIteration` callback to monitor progress
5. Verify initial parameters are reasonable

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

## Requirements

- Node.js >= 14.0.0
- Modern browsers with ES2020 support

## License

MIT

## Contributing

Contributions are welcome! Please read `CODING_RULES.md` before submitting pull requests.

