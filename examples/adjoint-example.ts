/**
 * Example: Constrained Optimization with Adjoint Method
 * 
 * This example demonstrates solving a constrained optimization problem
 * using the adjoint gradient descent method.
 * 
 * Problem:
 * Minimize: f(p, x) = p² + x²
 * Subject to: c(p, x) = p + x - 1 = 0
 * 
 * Analytical solution: p = 0.5, x = 0.5, f = 0.5
 * 
 * The adjoint method efficiently computes gradients by solving for
 * an adjoint variable λ instead of explicitly inverting matrices.
 */

import { adjointGradientDescent, printAdjointGradientDescentResult } from '../src/index';
import type { ConstrainedCostFn, ConstraintFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

// Define cost function: f(p, x) = p² + x²
const costFunction: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
  return p[0] * p[0] + x[0] * x[0];
};

// Define constraint: c(p, x) = p + x - 1 = 0
const constraintFunction: ConstraintFn = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([p[0] + x[0] - 1.0]);
};

console.log('=== Constrained Optimization: Adjoint Method ===\n');
console.log('Problem:');
console.log('  Minimize: f(p, x) = p² + x²');
console.log('  Subject to: c(p, x) = p + x - 1 = 0\n');
console.log('Analytical solution:');
console.log('  p = 0.5, x = 0.5, f = 0.5\n');

// Initial guess
const initialP = new Float64Array([2.0]);
const initialX = new Float64Array([-1.0]); // Satisfies constraint: 2 + (-1) - 1 = 0

console.log('Initial values:');
console.log(`  p₀ = ${initialP[0]}`);
console.log(`  x₀ = ${initialX[0]}`);
const initialConstraint = constraintFunction(initialP, initialX);
console.log(`  c(p₀, x₀) = ${initialConstraint[0]} (should be ≈ 0)\n`);

console.log('Starting adjoint gradient descent optimization...\n');

console.log('Using DEBUG log level to see detailed iteration information...\n');

const startTime = performance.now();
const result = adjointGradientDescent(
  initialP,
  initialX,
  costFunction,
  constraintFunction,
  {
    maxIterations: 100,
    tolerance: 1e-6,
    useLineSearch: true,
    logLevel: 'DEBUG', // Enable detailed logging for each iteration
    onIteration: (iteration, cost, params) => {
      // onIteration callback can be used for custom visualization
      // Logger already provides detailed output with logLevel: 'DEBUG'
    }
  }
);

const endTime = performance.now();
const elapsedTime = endTime - startTime;

const finalConstraint = constraintFunction(result.parameters, result.finalStates);
printAdjointGradientDescentResult(result, {
  showExecutionTime: true,
  elapsedTimeMs: elapsedTime
});
console.log(`\n  c(p, x) = ${finalConstraint[0].toFixed(8)} (should be ≈ 0)`);

// Verify solution
console.log('\n=== Verification ===');
const errorP = Math.abs(result.parameters[0] - 0.5);
const errorX = Math.abs(result.finalStates[0] - 0.5);
const errorF = Math.abs(result.finalCost - 0.5);
const constraintNorm = vectorNorm(finalConstraint);

console.log(`Parameter error: ${errorP.toFixed(6)}`);
console.log(`State error: ${errorX.toFixed(6)}`);
console.log(`Cost error: ${errorF.toFixed(6)}`);
console.log(`Constraint violation: ${constraintNorm.toFixed(8)}`);

if (errorP < 1e-3 && errorX < 1e-3 && errorF < 1e-3 && constraintNorm < 1e-3) {
  console.log('\n✅ Solution verified: All errors are within tolerance!');
} else {
  console.log('\n⚠️  Solution may need more iterations or different settings.');
}

