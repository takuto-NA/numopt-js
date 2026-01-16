/**
 * Example: Constrained Gauss-Newton Method
 * 
 * This example demonstrates solving a constrained nonlinear least squares problem
 * using the constrained Gauss-Newton method with effective Jacobian.
 * 
 * Problem:
 * Minimize: f(p, x) = 1/2 ((p - 0.5)² + (x - 0.5)²)
 * Subject to: c(p, x) = p + x - 1 = 0
 * 
 * Residual: r(p, x) = [p - 0.5, x - 0.5]
 * Analytical solution: p = 0.5, x = 0.5, f = 0
 * 
 * The constrained Gauss-Newton method uses effective Jacobian J_eff to capture
 * constraint effects, enabling quadratic convergence near the solution.
 */

import { constrainedGaussNewton, adjointGradientDescent, printConstrainedGaussNewtonResult, printAdjointGradientDescentResult } from '../src/index';
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

console.log('=== Constrained Gauss-Newton Method ===\n');
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

console.log('Starting constrained Gauss-Newton optimization...\n');

const startTime = performance.now();
const result = constrainedGaussNewton(
  initialP,
  initialX,
  residualFunction,
  constraintFunction,
  {
    maxIterations: 100,
    tolerance: 1e-6,
    logLevel: 'INFO',
    onIteration: (iteration, cost, params) => {
      if (iteration === 0 || iteration % 5 === 0) {
        // Note: We can't access currentStates here, so we'll log constraint after optimization
        console.log(`Iteration ${iteration}: p = ${params[0].toFixed(6)}, cost = ${cost.toFixed(8)}`);
      }
    }
  }
);
const endTime = performance.now();
const elapsedTime = endTime - startTime;

const finalConstraint = constraintFunction(result.finalParameters, result.finalStates);
printConstrainedGaussNewtonResult(result, {
  showExecutionTime: true,
  elapsedTimeMs: elapsedTime
});
console.log(`\n  c(p, x) = ${finalConstraint[0].toFixed(10)} (should be ≈ 0)`);

// Compare with adjoint gradient descent
console.log('\n=== Comparison with Adjoint Gradient Descent ===');
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
printAdjointGradientDescentResult(resultAGD, {
  showSectionHeaders: false,
  showExecutionTime: true,
  elapsedTimeMs: elapsedTimeAGD
});

console.log('\nConstrained Gauss-Newton:');
printConstrainedGaussNewtonResult(result, {
  showSectionHeaders: false,
  showExecutionTime: true,
  elapsedTimeMs: elapsedTime
});

if (result.iterations < resultAGD.iterations) {
  console.log(`\n✅ Constrained Gauss-Newton converged faster (${result.iterations} vs ${resultAGD.iterations} iterations)`);
} else {
  console.log(`\n⚠️  Adjoint Gradient Descent converged faster (${resultAGD.iterations} vs ${result.iterations} iterations)`);
}

// Verify solution
console.log('\n=== Verification ===');
const errorP = Math.abs(result.finalParameters[0] - 0.5);
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
} else {
  console.log('\n⚠️  Solution may need more iterations or different settings.');
  if (constraintNorm >= tolerance) {
    console.log(`   Constraint violation too large: ${constraintNorm.toFixed(10)} >= ${tolerance}`);
  }
}

