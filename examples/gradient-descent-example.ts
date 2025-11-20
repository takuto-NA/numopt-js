/**
 * Example: Gradient Descent Optimization
 * 
 * This example demonstrates how to use gradient descent to minimize
 * a simple quadratic function: f(x, y) = x^2 + y^2
 * 
 * Minimum is at (0, 0)
 */

import { gradientDescent, printGradientDescentResult } from '../src/index';
import type { CostFn, GradientFn } from '../src/core/types';

// Define cost function: f(x, y) = x^2 + y^2
const costFunction: CostFn = (params: Float64Array) => {
  return params[0] * params[0] + params[1] * params[1];
};

// Define gradient function: ∇f(x, y) = [2x, 2y]
const gradientFunction: GradientFn = (params: Float64Array) => {
  return new Float64Array([2 * params[0], 2 * params[1]]);
};

// Run optimization
const initialParameters = new Float64Array([5.0, -3.0]);

console.log('Starting gradient descent optimization...');
console.log('Initial parameters:', Array.from(initialParameters));

const result = gradientDescent(initialParameters, costFunction, gradientFunction, {
  maxIterations: 1000,
  tolerance: 1e-6,
  useLineSearch: true,
  verbose: true,
  onIteration: (iteration, cost, params) => {
    if (iteration % 10 === 0) {
      console.log(`Iteration ${iteration}: cost = ${cost.toFixed(6)}`);
    }
  }
});

printGradientDescentResult(result);

