/**
 * Newton-adjoint on an implicit chain: exact partial curvature and numerical directional curvature.
 */

import { adjointNewton, printResult } from '../src/index';
import { createChainProblem } from './adjoint-newton-problems';

const problem = createChainProblem();
for (const mode of ['analytical', 'numerical'] as const) {
  const result = adjointNewton(problem.parameters, problem.states, problem.cost, problem.constraint,
    mode === 'analytical' ? problem.derivatives : {});
  console.log(`\n${mode}: expected p = ${problem.expected[0]}`);
  printResult(result);
  if (!result.converged || Math.abs(result.finalParameters[0] - problem.expected[0]) > 1e-5) process.exitCode = 1;
}
