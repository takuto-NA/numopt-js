/** Runs the section 4.2 mechanism-fitting example with existing numopt-js solvers. */

import { createFourBarProblem, decodeLengths } from './mechanical-character/four-bar';
import { METHODS, runMechanismSolver, verifyMechanismOutcome } from './mechanical-character/solvers';

const problem = createFourBarProblem();
console.log('Coros et al. (2013), section 4.2: synthetic four-bar validation (not original character data).');
const rows = METHODS.map(method => {
  const outcome = runMechanismSolver(problem, method);
  const verification = verifyMechanismOutcome(problem, outcome, false);
  console.log(method, 'lengths [crank, coupler, rocker, ground]:', decodeLengths(outcome.result.finalParameters));
  return { method, ...verification };
});
console.table(rows);
if (rows.some(row => !row.passed)) process.exitCode = 1;
