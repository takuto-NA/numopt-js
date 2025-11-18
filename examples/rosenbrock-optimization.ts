/**
 * Example: Rosenbrock Function Optimization
 * 
 * This example demonstrates optimizing the Rosenbrock function,
 * a classic test function for optimization algorithms.
 * 
 * Rosenbrock function: f(x, y) = (a - x)^2 + b(y - x^2)^2
 * where typically a = 1, b = 100
 * 
 * Global minimum is at (1, 1) with f(1, 1) = 0
 * This function has a narrow, curved valley that makes optimization challenging.
 */

import { gradientDescent, finiteDiffGradient } from '../src/index';
import type { CostFn, GradientFn } from '../src/core/types';

// Rosenbrock function parameters
const a = 1.0;
const b = 100.0;

// Define cost function: f(x, y) = (a - x)^2 + b(y - x^2)^2
const rosenbrockFunction: CostFn = (params: Float64Array) => {
  const x = params[0];
  const y = params[1];
  const term1 = (a - x) * (a - x);
  const term2 = b * (y - x * x) * (y - x * x);
  return term1 + term2;
};

// Analytical gradient: ∇f(x, y) = [-2(a-x) - 4bx(y-x^2), 2b(y-x^2)]
const rosenbrockGradient: GradientFn = (params: Float64Array) => {
  const x = params[0];
  const y = params[1];
  const dx = -2 * (a - x) - 4 * b * x * (y - x * x);
  const dy = 2 * b * (y - x * x);
  return new Float64Array([dx, dy]);
};

console.log('=== Rosenbrock Function Optimization ===\n');
console.log('Function: f(x, y) = (1 - x)^2 + 100(y - x^2)^2');
console.log('Global minimum: (1, 1) with f(1, 1) = 0\n');

// Test with analytical gradient
console.log('--- Using Analytical Gradient ---');
const initialParams1 = new Float64Array([-1.2, 1.0]);
console.log('Initial parameters:', Array.from(initialParams1));
console.log('Initial cost:', rosenbrockFunction(initialParams1).toFixed(6));

const startTime1 = performance.now();
const result1 = gradientDescent(initialParams1, rosenbrockFunction, rosenbrockGradient, {
  maxIterations: 10000,
  tolerance: 1e-8,
  useLineSearch: true,
  verbose: false,
  onIteration: (iteration, cost) => {
    if (iteration % 1000 === 0) {
      console.log(`  Iteration ${iteration}: cost = ${cost.toFixed(10)}`);
    }
  }
});

const endTime1 = performance.now();
const elapsedTime1 = endTime1 - startTime1;

console.log('\nResults:');
console.log('  Final parameters:', Array.from(result1.parameters).map(x => x.toFixed(6)));
console.log('  Final cost:', result1.finalCost.toFixed(10));
console.log('  Converged:', result1.converged);
console.log('  Iterations:', result1.iterations);
console.log('  Error from true minimum:', 
  Math.sqrt((result1.parameters[0] - 1)**2 + (result1.parameters[1] - 1)**2).toFixed(10));
console.log(`  Execution time: ${elapsedTime1.toFixed(2)} ms (${(elapsedTime1 / 1000).toFixed(3)} seconds)`);
console.log(`  Time per iteration: ${(elapsedTime1 / result1.iterations).toFixed(3)} ms`);

// Test with numerical gradient
console.log('\n--- Using Numerical Gradient (Finite Differences) ---');
const initialParams2 = new Float64Array([-1.2, 1.0]);
console.log('Initial parameters:', Array.from(initialParams2));

const numericalGradient: GradientFn = (params: Float64Array) => {
  return finiteDiffGradient(params, rosenbrockFunction, { stepSize: 1e-6 });
};

const startTime2 = performance.now();
const result2 = gradientDescent(initialParams2, rosenbrockFunction, numericalGradient, {
  maxIterations: 10000,
  tolerance: 1e-8,
  useLineSearch: true,
  verbose: false,
  onIteration: (iteration, cost) => {
    if (iteration % 1000 === 0) {
      console.log(`  Iteration ${iteration}: cost = ${cost.toFixed(10)}`);
    }
  }
});

const endTime2 = performance.now();
const elapsedTime2 = endTime2 - startTime2;

console.log('\nResults:');
console.log('  Final parameters:', Array.from(result2.parameters).map(x => x.toFixed(6)));
console.log('  Final cost:', result2.finalCost.toFixed(10));
console.log('  Converged:', result2.converged);
console.log('  Iterations:', result2.iterations);
console.log('  Error from true minimum:', 
  Math.sqrt((result2.parameters[0] - 1)**2 + (result2.parameters[1] - 1)**2).toFixed(10));
console.log(`  Execution time: ${elapsedTime2.toFixed(2)} ms (${(elapsedTime2 / 1000).toFixed(3)} seconds)`);
console.log(`  Time per iteration: ${(elapsedTime2 / result2.iterations).toFixed(3)} ms`);

