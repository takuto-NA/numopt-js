/**
 * Compare all three solvers on an objective with a persistent nonlinear state-curvature contribution.
 */

import { adjointBfgs, adjointNewton, adjointNewtonExact, printResult } from '../src/index';
import { createQuadraticConvergenceProblem } from './adjoint-newton-problems';

const problem = createQuadraticConvergenceProblem();
const options = { ...problem.derivatives, ...problem.exactDerivatives, tolerance: 1e-9, constraintTolerance: 1e-12 };
for (const [name, solver] of [['BFGS', adjointBfgs], ['Approximate Newton', adjointNewton], ['Full Newton', adjointNewtonExact]] as const) {
  console.log(`\n${name}: expected p = ${problem.expected[0]}`);
  const result = solver(problem.parameters, problem.states, problem.cost, problem.constraint, options);
  printResult(result);
  if (Math.abs(result.finalParameters[0] - problem.expected[0]) > 1e-6) process.exitCode = 1;
}
