/**
 * Compare equality-constrained least-squares solvers on one shared toy problem.
 *
 * Problem:
 *   Minimize  f(p, x) = 1/2 ((p - 0.5)² + (x - 0.5)²)
 *   Subject to  c(p, x) = p + x - 1 = 0
 * Analytical solution: p = 0.5, x = 0.5, f = 0
 *
 * Runs Constrained LM, Constrained GN, and Adjoint GD side by side.
 */

import {
  constrainedLevenbergMarquardt,
  constrainedGaussNewton,
  adjointGradientDescent,
  printConstrainedLevenbergMarquardtResult,
  printConstrainedGaussNewtonResult,
  printAdjointGradientDescentResult
} from '../src/index';
import type { ConstrainedResidualFn, ConstraintFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

const TARGET_PARAMETER = 0.5;
const TARGET_STATE = 0.5;
const TARGET_COST = 0.0;
const SOLUTION_TOLERANCE = 1e-3;
const MAX_ITERATIONS = 100;
const CONVERGENCE_TOLERANCE = 1e-6;

const residualFunction: ConstrainedResidualFn = (parameters, states) => {
  return new Float64Array([parameters[0] - TARGET_PARAMETER, states[0] - TARGET_STATE]);
};

const constraintFunction: ConstraintFn = (parameters, states) => {
  return new Float64Array([parameters[0] + states[0] - 1.0]);
};

const costFunction = (parameters: Float64Array, states: Float64Array) => {
  const residual = residualFunction(parameters, states);
  return 0.5 * (residual[0] * residual[0] + residual[1] * residual[1]);
};

const initialParameters = new Float64Array([2.0]);
const initialStates = new Float64Array([-1.0]);

console.log('=== Constrained solver comparison ===\n');
console.log('Problem: min 1/2 ||r||²  s.t. p + x = 1, r = [p-0.5, x-0.5]');
console.log(`Analytical solution: p = ${TARGET_PARAMETER}, x = ${TARGET_STATE}, f = ${TARGET_COST}\n`);

const lmStart = performance.now();
const lmResult = constrainedLevenbergMarquardt(
  initialParameters,
  initialStates,
  residualFunction,
  constraintFunction,
  {
    maxIterations: MAX_ITERATIONS,
    tolGradient: CONVERGENCE_TOLERANCE,
    tolStep: CONVERGENCE_TOLERANCE,
    tolResidual: CONVERGENCE_TOLERANCE,
    logLevel: 'WARN'
  }
);
const lmElapsedMs = performance.now() - lmStart;

const gnStart = performance.now();
const gnResult = constrainedGaussNewton(
  initialParameters,
  initialStates,
  residualFunction,
  constraintFunction,
  {
    maxIterations: MAX_ITERATIONS,
    tolerance: CONVERGENCE_TOLERANCE,
    logLevel: 'WARN'
  }
);
const gnElapsedMs = performance.now() - gnStart;

const adjointStart = performance.now();
const adjointResult = adjointGradientDescent(
  initialParameters,
  initialStates,
  costFunction,
  constraintFunction,
  {
    maxIterations: MAX_ITERATIONS,
    tolerance: CONVERGENCE_TOLERANCE,
    useLineSearch: true,
    logLevel: 'WARN'
  }
);
const adjointElapsedMs = performance.now() - adjointStart;

console.log('Constrained Levenberg-Marquardt:');
printConstrainedLevenbergMarquardtResult(lmResult, {
  showSectionHeaders: false,
  showExecutionTime: true,
  elapsedTimeMs: lmElapsedMs
});

console.log('\nConstrained Gauss-Newton:');
printConstrainedGaussNewtonResult(gnResult, {
  showSectionHeaders: false,
  showExecutionTime: true,
  elapsedTimeMs: gnElapsedMs
});

console.log('\nAdjoint Gradient Descent:');
printAdjointGradientDescentResult(adjointResult, {
  showSectionHeaders: false,
  showExecutionTime: true,
  elapsedTimeMs: adjointElapsedMs
});

const rankedByIterations = [
  { name: 'Constrained LM', iterations: lmResult.iterations },
  { name: 'Constrained GN', iterations: gnResult.iterations },
  { name: 'Adjoint GD', iterations: adjointResult.iterations }
].sort((left, right) => left.iterations - right.iterations);

console.log(
  `\nFastest by iteration count: ${rankedByIterations[0].name} (${rankedByIterations[0].iterations})`
);

function verifySolver(
  label: string,
  finalParameters: Float64Array,
  finalStates: Float64Array,
  finalCost: number
): boolean {
  const constraintNorm = vectorNorm(constraintFunction(finalParameters, finalStates));
  const parameterError = Math.abs(finalParameters[0] - TARGET_PARAMETER);
  const stateError = Math.abs(finalStates[0] - TARGET_STATE);
  const costError = Math.abs(finalCost - TARGET_COST);
  const looksCorrect =
    parameterError < SOLUTION_TOLERANCE &&
    stateError < SOLUTION_TOLERANCE &&
    costError < SOLUTION_TOLERANCE &&
    constraintNorm < SOLUTION_TOLERANCE;

  console.log(`\n=== ${label} verification ===`);
  console.log(`Parameter error: ${parameterError.toFixed(8)}`);
  console.log(`State error: ${stateError.toFixed(8)}`);
  console.log(`Cost error: ${costError.toFixed(8)}`);
  console.log(`Constraint violation: ${constraintNorm.toFixed(10)}`);
  console.log(looksCorrect ? 'Solution verified within tolerance.' : 'Outside tolerance.');
  return looksCorrect;
}

const allVerified =
  verifySolver('Constrained LM', lmResult.finalParameters, lmResult.finalStates, lmResult.finalCost) &&
  verifySolver('Constrained GN', gnResult.finalParameters, gnResult.finalStates, gnResult.finalCost) &&
  verifySolver(
    'Adjoint GD',
    adjointResult.finalParameters,
    adjointResult.finalStates,
    adjointResult.finalCost
  );

if (!allVerified) {
  console.log('\nOne or more solvers need more iterations or different settings.');
}
