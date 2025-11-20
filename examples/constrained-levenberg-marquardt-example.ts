/**
 * Example: Constrained Levenberg-Marquardt Method
 * 
 * This example demonstrates solving a constrained nonlinear least squares problem
 * using the constrained Levenberg-Marquardt method with effective Jacobian.
 * 
 * Problem:
 * Minimize: f(p, x) = 1/2 ((p - 0.5)² + (x - 0.5)²)
 * Subject to: c(p, x) = p + x - 1 = 0
 * 
 * Residual: r(p, x) = [p - 0.5, x - 0.5]
 * Analytical solution: p = 0.5, x = 0.5, f = 0
 * 
 * The constrained Levenberg-Marquardt method uses effective Jacobian J_eff with
 * damping parameter lambda for robust convergence, even when the Hessian is singular.
 */

import { constrainedLevenbergMarquardt, constrainedGaussNewton, adjointGradientDescent } from '../src/index';
import type { ConstrainedResidualFn, ConstraintFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

// Define residual function: r(p, x) = [p - 0.5, x - 0.5]
const residualFunction: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([p[0] - 0.5, x[0] - 0.5]);
};

// Define constraint: c(p, x) = p + x - 1 = 0
const constraintFunction: ConstraintFn = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([p[0] + x[0] - 1.0]);
};

console.log('=== Constrained Levenberg-Marquardt Method ===\n');
console.log('Problem:');
console.log('  Minimize: f(p, x) = 1/2 ((p - 0.5)² + (x - 0.5)²)');
console.log('  Subject to: c(p, x) = p + x - 1 = 0\n');
console.log('Analytical solution:');
console.log('  p = 0.5, x = 0.5, f = 0\n');

// Initial guess
const initialP = new Float64Array([2.0]);
const initialX = new Float64Array([-1.0]); // Satisfies constraint: 2 + (-1) - 1 = 0

console.log('Initial values:');
console.log(`  p₀ = ${initialP[0]}`);
console.log(`  x₀ = ${initialX[0]}`);
const initialConstraint = constraintFunction(initialP, initialX);
console.log(`  c(p₀, x₀) = ${initialConstraint[0].toFixed(8)} (should be ≈ 0)\n`);

console.log('Starting constrained Levenberg-Marquardt optimization...\n');

const startTime = performance.now();
let iterationCount = 0;
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
    lambdaFactor: 10.0,
    logLevel: 'INFO',
    onIteration: (iteration, cost, params) => {
      iterationCount = iteration;
      if (iteration === 0 || iteration % 5 === 0) {
        // Note: We can't access currentStates here, so we'll log constraint after optimization
        console.log(`Iteration ${iteration}: p = ${params[0].toFixed(6)}, cost = ${cost.toFixed(8)}`);
      }
    }
  }
);
const endTime = performance.now();
const elapsedTime = endTime - startTime;

console.log('\n=== Optimization Results ===');
console.log('Optimized parameters:');
console.log(`  p = ${result.parameters[0].toFixed(8)} (analytical: 0.5)`);
console.log(`  x = ${result.finalStates[0].toFixed(8)} (analytical: 0.5)`);
console.log('\nCost:');
console.log(`  f(p, x) = ${result.finalCost.toFixed(8)} (analytical: 0.0)`);
console.log('\nConstraint satisfaction:');
const finalConstraint = constraintFunction(result.parameters, result.finalStates);
console.log(`  c(p, x) = ${finalConstraint[0].toFixed(10)} (should be ≈ 0)`);
console.log(`  ||c(p, x)|| = ${result.finalConstraintNorm?.toFixed(10) ?? 'N/A'}`);

console.log('\nConvergence:');
console.log(`  Converged: ${result.converged}`);
console.log(`  Iterations: ${result.iterations}`);
console.log(`  Final gradient norm: ${result.finalGradientNorm?.toFixed(8) ?? 'N/A'}`);
console.log(`  Final residual norm: ${result.finalResidualNorm?.toFixed(8) ?? 'N/A'}`);
console.log(`  Final lambda: ${result.finalLambda.toFixed(6)}`);

console.log(`\nExecution time: ${elapsedTime.toFixed(2)} ms`);
if (result.iterations > 0) {
  console.log(`Time per iteration: ${(elapsedTime / result.iterations).toFixed(3)} ms`);
}

// Compare with other methods
console.log('\n=== Comparison with Other Methods ===');

// Constrained Gauss-Newton
const startTimeGN = performance.now();
const resultGN = constrainedGaussNewton(
  initialP,
  initialX,
  residualFunction,
  constraintFunction,
  {
    maxIterations: 100,
    tolerance: 1e-6,
    logLevel: 'WARN'
  }
);
const endTimeGN = performance.now();
const elapsedTimeGN = endTimeGN - startTimeGN;

console.log('Constrained Gauss-Newton:');
console.log(`  Iterations: ${resultGN.iterations}`);
console.log(`  Final cost: ${resultGN.finalCost.toFixed(8)}`);
console.log(`  Execution time: ${elapsedTimeGN.toFixed(2)} ms`);
console.log(`  Constraint norm: ${resultGN.finalConstraintNorm?.toFixed(10) ?? 'N/A'}`);

// Adjoint Gradient Descent
const costFunction = (p: Float64Array, x: Float64Array) => {
  const residual = residualFunction(p, x);
  return 0.5 * (residual[0] * residual[0] + residual[1] * residual[1]);
};

const startTimeAGD = performance.now();
const resultAGD = adjointGradientDescent(
  initialP,
  initialX,
  costFunction,
  constraintFunction,
  {
    maxIterations: 100,
    tolerance: 1e-6,
    useLineSearch: true,
    logLevel: 'WARN'
  }
);
const endTimeAGD = performance.now();
const elapsedTimeAGD = endTimeAGD - startTimeAGD;

console.log('Adjoint Gradient Descent:');
console.log(`  Iterations: ${resultAGD.iterations}`);
console.log(`  Final cost: ${resultAGD.finalCost.toFixed(8)}`);
console.log(`  Execution time: ${elapsedTimeAGD.toFixed(2)} ms`);
console.log(`  Constraint norm: ${resultAGD.finalConstraintNorm?.toFixed(10) ?? 'N/A'}`);

console.log('\nConstrained Levenberg-Marquardt:');
console.log(`  Iterations: ${result.iterations}`);
console.log(`  Final cost: ${result.finalCost.toFixed(8)}`);
console.log(`  Execution time: ${elapsedTime.toFixed(2)} ms`);
console.log(`  Constraint norm: ${result.finalConstraintNorm?.toFixed(10) ?? 'N/A'}`);

// Find fastest method
const methods = [
  { name: 'Constrained Levenberg-Marquardt', iterations: result.iterations },
  { name: 'Constrained Gauss-Newton', iterations: resultGN.iterations },
  { name: 'Adjoint Gradient Descent', iterations: resultAGD.iterations }
];
methods.sort((a, b) => a.iterations - b.iterations);
console.log(`\nFastest convergence: ${methods[0].name} (${methods[0].iterations} iterations)`);

// Verify solution
console.log('\n=== Verification ===');
const errorP = Math.abs(result.parameters[0] - 0.5);
const errorX = Math.abs(result.finalStates[0] - 0.5);
const errorF = Math.abs(result.finalCost - 0.0);
const constraintNorm = vectorNorm(finalConstraint);

console.log(`Parameter error: ${errorP.toFixed(8)}`);
console.log(`State error: ${errorX.toFixed(8)}`);
console.log(`Cost error: ${errorF.toFixed(8)}`);
console.log(`Constraint violation: ${constraintNorm.toFixed(10)}`);

const tolerance = 1e-3;
if (errorP < tolerance && errorX < tolerance && errorF < tolerance && constraintNorm < tolerance) {
  console.log('\n✅ Solution verified: All errors are within tolerance!');
  console.log(`✅ Constraint satisfied: ||c(p, x)|| = ${constraintNorm.toFixed(10)} < ${tolerance}`);
  console.log(`✅ Fast convergence: ${result.iterations} iterations`);
  if (result.iterations <= resultGN.iterations && result.iterations <= resultAGD.iterations) {
    console.log(`✅ Best performance: Converged faster than other methods`);
  }
} else {
  console.log('\n⚠️  Solution may need more iterations or different settings.');
  if (constraintNorm >= tolerance) {
    console.log(`   Constraint violation too large: ${constraintNorm.toFixed(10)} >= ${tolerance}`);
  }
}

