/**
 * Harder adjoint cases: multi-variable affine constraints and a nonlinear constraint.
 * Starts from a valid implicit-state point so x(p) exists locally.
 */

import { adjointGradientDescent, printAdjointGradientDescentResult } from '../src/index';
import type { ConstrainedCostFn, ConstraintFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

const SOLUTION_TOLERANCE = 1e-2;
const CONSTRAINT_TOLERANCE = 1e-6;

function requireSolution(
  label: string,
  parameters: Float64Array,
  expectedParameters: number[],
  states: Float64Array,
  expectedStates: number[],
  constraintNorm: number
): boolean {
  const parameterOk = expectedParameters.every(
    (value, index) => Math.abs(parameters[index] - value) < SOLUTION_TOLERANCE
  );
  const stateOk = expectedStates.every(
    (value, index) => Math.abs(states[index] - value) < SOLUTION_TOLERANCE
  );
  const feasible = constraintNorm < CONSTRAINT_TOLERANCE;
  if (parameterOk && stateOk && feasible) {
    return true;
  }
  console.error(`\n${label} did not reach the feasible analytical solution.`);
  return false;
}

console.log('=== Advanced Constrained Optimization: Adjoint Method ===\n');

console.log('Problem 1: 2D Constrained Optimization');
console.log('  Minimize: f(p, x) = (p₁ - 1)² + (p₂ - 2)² + x₁² + x₂²');
console.log('  Subject to: c₁ = p₁ + x₁ - 1 = 0, c₂ = p₂ + x₂ - 2 = 0');
console.log('  Analytical solution: p = [1, 2], x = [0, 0], f = 0\n');

const cost2D: ConstrainedCostFn = (parameters, states) => {
  return (
    Math.pow(parameters[0] - 1, 2) +
    Math.pow(parameters[1] - 2, 2) +
    states[0] * states[0] +
    states[1] * states[1]
  );
};

const constraint2D: ConstraintFn = (parameters, states) => {
  return new Float64Array([
    parameters[0] + states[0] - 1.0,
    parameters[1] + states[1] - 2.0
  ]);
};

const result2D = adjointGradientDescent(
  new Float64Array([3.0, 4.0]),
  new Float64Array([-2.0, -2.0]),
  cost2D,
  constraint2D,
  {
    maxIterations: 200,
    tolerance: 1e-6,
    useLineSearch: true,
    logLevel: 'WARN'
  }
);

const finalConstraint2D = constraint2D(result2D.finalParameters, result2D.finalStates);
printAdjointGradientDescentResult(result2D, { showSectionHeaders: false });
console.log(`  ||c(p, x)|| = ${vectorNorm(finalConstraint2D).toFixed(8)}`);
console.log('  Analytical: p = [1, 2], x = [0, 0], f = 0\n');

console.log('\n' + '='.repeat(70) + '\n');
console.log('Problem 2: Nonlinear Constraint');
console.log('  Minimize: f(p, x) = (p - 1)² + (x - 1)²');
console.log('  Subject to: c(p, x) = p² + x² - 2 = 0');
console.log('  Analytical solution: p = 1, x = 1, f = 0');
console.log('  Start from a feasible implicit-state point on the circle.\n');

const costCircle: ConstrainedCostFn = (parameters, states) => {
  return Math.pow(parameters[0] - 1, 2) + Math.pow(states[0] - 1, 2);
};

const constraintCircle: ConstraintFn = (parameters, states) => {
  return new Float64Array([parameters[0] * parameters[0] + states[0] * states[0] - 2.0]);
};

const initialCircleParameter = 0.2;
const initialCircleState = Math.sqrt(2 - initialCircleParameter * initialCircleParameter);

const resultCircle = adjointGradientDescent(
  new Float64Array([initialCircleParameter]),
  new Float64Array([initialCircleState]),
  costCircle,
  constraintCircle,
  {
    maxIterations: 300,
    tolerance: 1e-6,
    useLineSearch: true,
    constraintTolerance: CONSTRAINT_TOLERANCE,
    logLevel: 'WARN'
  }
);

const finalConstraintCircle = constraintCircle(
  resultCircle.finalParameters,
  resultCircle.finalStates
);
printAdjointGradientDescentResult(resultCircle, { showSectionHeaders: false });
console.log(`  ||c(p, x)|| = ${vectorNorm(finalConstraintCircle).toFixed(8)}`);
console.log('  Analytical: p = 1.0, x = 1.0, f = 0.0\n');

console.log('\n' + '='.repeat(70) + '\n');
console.log('Summary:');
console.log(
  `  Problem 1 (2D): ${result2D.converged ? 'converged' : 'did not converge'} in ${result2D.iterations} iterations`
);
console.log(
  `  Problem 2 (Nonlinear): ${resultCircle.converged ? 'converged' : 'did not converge'} in ${resultCircle.iterations} iterations`
);

const verified =
  requireSolution(
    'Problem 1',
    result2D.finalParameters,
    [1, 2],
    result2D.finalStates,
    [0, 0],
    vectorNorm(finalConstraint2D)
  ) &&
  requireSolution(
    'Problem 2',
    resultCircle.finalParameters,
    [1],
    resultCircle.finalStates,
    [1],
    vectorNorm(finalConstraintCircle)
  );

if (!verified) {
  process.exit(1);
}
