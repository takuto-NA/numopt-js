/**
 * Debug script to investigate constrainedLevenbergMarquardt failure on Rosenbrock problem
 */

import { constrainedLevenbergMarquardt } from '../src/core/constrainedLevenbergMarquardt';
import type { ConstrainedResidualFn, ConstraintFn } from '../src/core/types';

// Rosenbrock residual function
const rosenbrockResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
  const a = 1.0 - p[0];
  const b = x[0] - p[0] * p[0];
  return new Float64Array([a, 10.0 * b]);
};

// Circle constraint
const circleConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([p[0] * p[0] + x[0] * x[0] - 2.0]);
};

console.log('=== Debugging constrainedLevenbergMarquardt on Rosenbrock Problem ===\n');

const initialP = new Float64Array([1.0]);
const initialX = new Float64Array([1.0]);

console.log('Initial values:');
console.log(`  p₀ = ${initialP[0]}`);
console.log(`  x₀ = ${initialX[0]}`);

const initialResidual = rosenbrockResidual(initialP, initialX);
const initialCost = 0.5 * (initialResidual[0] ** 2 + initialResidual[1] ** 2);
console.log(`  Initial cost: ${initialCost}`);

const initialConstraint = circleConstraint(initialP, initialX);
console.log(`  Initial constraint: ${initialConstraint[0]}\n`);

console.log('Running optimization with verbose logging...\n');

const result = constrainedLevenbergMarquardt(
  initialP,
  initialX,
  rosenbrockResidual,
  circleConstraint,
  {
    maxIterations: 500,
    tolGradient: 1e-4,
    tolStep: 1e-6,
    constraintTolerance: 1e-3,
    lambdaInitial: 1e-2,
    verbose: true,
    logLevel: 'debug'
  }
);

console.log('\n=== Results ===');
console.log(`Converged: ${result.converged}`);
console.log(`Iterations: ${result.iterations}`);
console.log(`Final cost: ${result.finalCost}`);
console.log(`Final gradient norm: ${result.finalGradientNorm}`);
console.log(`Final residual norm: ${result.finalResidualNorm}`);
console.log(`Final constraint norm: ${result.finalConstraintNorm}`);
console.log(`Final lambda: ${result.finalLambda}`);
console.log(`Final parameters: [${Array.from(result.finalParameters).join(', ')}]`);
console.log(`Final states: [${Array.from(result.finalStates).join(', ')}]`);

// Check for NaN or Infinity
if (!isFinite(result.finalCost)) {
  console.log('\n⚠️ ERROR: Final cost is not finite!');
  console.log('This indicates a numerical instability issue.');
}

if (result.finalParameters.some(p => !isFinite(p))) {
  console.log('\n⚠️ ERROR: Some parameters are not finite!');
  console.log(`Parameters: [${Array.from(result.finalParameters).join(', ')}]`);
}

if (result.finalStates.some(x => !isFinite(x))) {
  console.log('\n⚠️ ERROR: Some states are not finite!');
  console.log(`States: [${Array.from(result.finalStates).join(', ')}]`);
}
