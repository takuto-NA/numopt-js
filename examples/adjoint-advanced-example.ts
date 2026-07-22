/**
 * Harder adjoint cases: multi-variable affine constraints and a nonlinear constraint.
 * For the basic residual toy problem, see examples/constrained-compare.ts.
 */

import { adjointGradientDescent, printAdjointGradientDescentResult } from '../src/index';
import type { ConstrainedCostFn, ConstraintFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

console.log('=== Advanced Constrained Optimization: Adjoint Method ===\n');

// Problem 1: 2D affine constraints
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

const initialParameters2D = new Float64Array([3.0, 4.0]);
const initialStates2D = new Float64Array([-2.0, -2.0]);

const result2D = adjointGradientDescent(
  initialParameters2D,
  initialStates2D,
  cost2D,
  constraint2D,
  {
    maxIterations: 200,
    tolerance: 1e-6,
    useLineSearch: true,
    logLevel: 'INFO'
  }
);

const finalConstraint2D = constraint2D(result2D.finalParameters, result2D.finalStates);
printAdjointGradientDescentResult(result2D, { showSectionHeaders: false });
console.log(`  ||c(p, x)|| = ${vectorNorm(finalConstraint2D).toFixed(8)}`);
console.log('  Analytical: p = [1, 2], x = [0, 0], f = 0\n');

// Problem 2: Nonlinear constraint
console.log('\n' + '='.repeat(70) + '\n');
console.log('Problem 2: Nonlinear Constraint');
console.log('  Minimize: f(p, x) = (p - 1)² + (x - 1)²');
console.log('  Subject to: c(p, x) = p² + x² - 2 = 0');
console.log('  Analytical solution: p = 1, x = 1, f = 0\n');

const costCircle: ConstrainedCostFn = (parameters, states) => {
  return Math.pow(parameters[0] - 1, 2) + Math.pow(states[0] - 1, 2);
};

const constraintCircle: ConstraintFn = (parameters, states) => {
  return new Float64Array([parameters[0] * parameters[0] + states[0] * states[0] - 2.0]);
};

const initialParametersCircle = new Float64Array([1.5]);
const initialStatesCircle = new Float64Array([0.5]);

const resultCircle = adjointGradientDescent(
  initialParametersCircle,
  initialStatesCircle,
  costCircle,
  constraintCircle,
  {
    maxIterations: 500,
    tolerance: 1e-4,
    useLineSearch: true,
    constraintTolerance: 1e-3,
    logLevel: 'INFO'
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
