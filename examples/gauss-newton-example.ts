/**
 * Example: Gauss-Newton Method
 * 
 * This example demonstrates solving a nonlinear least squares problem
 * using the Gauss-Newton method.
 * 
 * Problem: Find x such that x^2 = 4
 * Residual: r(x) = x^2 - 4
 * Solution: x = ±2
 */

import { gaussNewton, printOptimizationResult } from '../src/index';
import type { ResidualFn, JacobianFn } from '../src/core/types';
import { Matrix } from 'ml-matrix';

// Define residual function: r(x) = x^2 - 4
const residualFunction: ResidualFn = (params: Float64Array) => {
  return new Float64Array([params[0] * params[0] - 4]);
};

// Define Jacobian function: J(x) = [2x]
const jacobianFunction: JacobianFn = (params: Float64Array) => {
  return new Matrix([[2 * params[0]]]);
};

// Run optimization
const initialParameters = new Float64Array([3.0]);

console.log('Starting Gauss-Newton optimization...');
console.log('Initial parameter:', initialParameters[0]);
console.log('Target: Find x such that x^2 = 4\n');

const result = gaussNewton(initialParameters, residualFunction, {
  jacobian: jacobianFunction,
  maxIterations: 100,
  tolerance: 1e-6,
  verbose: true,
  onIteration: (iteration, cost, params) => {
    if (iteration % 5 === 0) {
      console.log(`Iteration ${iteration}: x = ${params[0].toFixed(6)}, cost = ${cost.toFixed(6)}`);
    }
  }
});

printOptimizationResult(result);

// Verify solution
const residual = residualFunction(result.finalParameters);
console.log('\nVerification:');
console.log(`Residual: ${residual[0]} (should be close to 0)`);
console.log(`x^2 = ${result.finalParameters[0] * result.finalParameters[0]} (should be close to 4)`);

