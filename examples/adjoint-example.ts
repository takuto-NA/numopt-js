/**
 * Equality-constrained optimization with the adjoint method.
 *
 * Problem:
 *   Minimize  f(p, x) = p² + x²
 *   Subject to  c(p, x) = p + x - 1 = 0
 * Analytical solution: p = 0.5, x = 0.5, f = 0.5
 */

import {
  adjointBfgs,
  adjointGradientDescent,
  printAdjointBfgsResult,
  printAdjointGradientDescentResult
} from '../src/index';
import type { ConstrainedCostFn, ConstraintFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

const TARGET_PARAMETER = 0.5;
const TARGET_STATE = 0.5;
const TARGET_COST = 0.5;
const SOLUTION_TOLERANCE = 1e-3;
const MAX_ITERATIONS = 100;
const CONVERGENCE_TOLERANCE = 1e-6;

const costFunction: ConstrainedCostFn = (parameters, states) => {
  return parameters[0] * parameters[0] + states[0] * states[0];
};

const constraintFunction: ConstraintFn = (parameters, states) => {
  return new Float64Array([parameters[0] + states[0] - 1.0]);
};

console.log('=== Constrained Optimization: Adjoint Method ===\n');
console.log('Problem: min p² + x²  s.t. p + x = 1');
console.log(`Analytical solution: p = ${TARGET_PARAMETER}, x = ${TARGET_STATE}, f = ${TARGET_COST}\n`);

const initialParameters = new Float64Array([2.0]);
const initialStates = new Float64Array([-1.0]);

const startTime = performance.now();
const result = adjointGradientDescent(
  initialParameters,
  initialStates,
  costFunction,
  constraintFunction,
  {
    maxIterations: MAX_ITERATIONS,
    tolerance: CONVERGENCE_TOLERANCE,
    useLineSearch: true,
    logLevel: 'INFO'
  }
);
const elapsedTimeMs = performance.now() - startTime;

const finalConstraint = constraintFunction(result.finalParameters, result.finalStates);
printAdjointGradientDescentResult(result, {
  showExecutionTime: true,
  elapsedTimeMs
});
console.log(`\n  c(p, x) = ${finalConstraint[0].toFixed(8)} (should be ≈ 0)`);

const bfgsStartTime = performance.now();
const bfgsResult = adjointBfgs(
  initialParameters,
  initialStates,
  costFunction,
  constraintFunction,
  {
    maxIterations: MAX_ITERATIONS,
    tolerance: CONVERGENCE_TOLERANCE,
    useLineSearch: true,
    logLevel: 'INFO'
  }
);
const bfgsElapsedTimeMs = performance.now() - bfgsStartTime;

console.log('\n=== Adjoint BFGS ===\n');
printAdjointBfgsResult(bfgsResult, {
  showExecutionTime: true,
  elapsedTimeMs: bfgsElapsedTimeMs
});

function verifyResult(
  label: string,
  finalParameters: Float64Array,
  finalStates: Float64Array,
  finalCost: number
): boolean {
  const constraint = constraintFunction(finalParameters, finalStates);
  const parameterError = Math.abs(finalParameters[0] - TARGET_PARAMETER);
  const stateError = Math.abs(finalStates[0] - TARGET_STATE);
  const costError = Math.abs(finalCost - TARGET_COST);
  const constraintNorm = vectorNorm(constraint);

  console.log(`\n=== ${label} verification ===`);
  console.log(`Parameter error: ${parameterError.toFixed(6)}`);
  console.log(`State error: ${stateError.toFixed(6)}`);
  console.log(`Cost error: ${costError.toFixed(6)}`);
  console.log(`Constraint violation: ${constraintNorm.toFixed(8)}`);
  return (
    parameterError < SOLUTION_TOLERANCE &&
    stateError < SOLUTION_TOLERANCE &&
    costError < SOLUTION_TOLERANCE &&
    constraintNorm < SOLUTION_TOLERANCE
  );
}

const verified =
  verifyResult('Adjoint GD', result.finalParameters, result.finalStates, result.finalCost) &&
  verifyResult('Adjoint BFGS', bfgsResult.finalParameters, bfgsResult.finalStates, bfgsResult.finalCost);

if (verified) {
  console.log('\nSolution verified within tolerance.');
} else {
  console.error('\nSolution may need more iterations or different settings.');
  process.exit(1);
}
