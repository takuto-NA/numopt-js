/** Validate frozen design variables, three-dimensional implicit gradients, fitting, and honest replay histories. */

import { describe, expect, it } from 'vitest';
import { finiteDiffConstraintPartialP, finiteDiffResidualPartialP, vectorNorm } from '../src/index';
import { assembleStates, createPhases, implicitGradient, oracleStates } from '../examples/mechanical-character/four-bar';
import { createTrajectoryProblem, DESIGN_DIMENSION, INITIAL_DESIGNS, TARGET_CASES } from '../examples/linkage-trajectory/model';
import { createTrajectoryEvaluator, fitTrajectory, FIT_METHODS } from '../examples/linkage-trajectory/fit';
import { HELD_OUT_OFFSET, HELD_OUT_PHASE_COUNT, verifyTrajectoryFit } from '../examples/linkage-trajectory/verify';

const DERIVATIVE_TOLERANCE = 1e-8;
const COST_CORRUPTION = 0.1;
const WRONG_BRANCH_ANGLE = Math.PI;
const PHASE_SEPARATION_TOLERANCE = 1e-12;

describe('fixed-topology trajectory fitting example', () => {
  it('leaves ground pivots, marker, phases and target unchanged for all tested designs', () => {
    const problem = createTrajectoryProblem();
    const targetBefore = JSON.stringify(problem.source.target);
    const fixed = Array.from(problem.source.targetParameters.slice(DESIGN_DIMENSION));
    for (const initial of INITIAL_DESIGNS) {
      expect(Array.from(problem.fullParameters(new Float64Array(initial)).slice(DESIGN_DIMENSION))).toEqual(fixed);
    }
    const heldOut = createPhases(HELD_OUT_PHASE_COUNT, HELD_OUT_OFFSET);
    expect(heldOut.every(phase => problem.source.phases.every(training =>
      Math.abs(phase - training) > PHASE_SEPARATION_TOLERANCE))).toBe(true);
    fitTrajectory(problem);
    expect(JSON.stringify(problem.source.target)).toBe(targetBefore);
  });

  it('matches three-column partials and public IFT to fully reassembled finite differences', () => {
    const problem = createTrajectoryProblem('distorted');
    const parameters = problem.initialParameters;
    const states = assembleStates(problem.fullParameters(parameters), problem.source.phases);
    for (const [analytical, numerical] of [
      [problem.derivatives.dcdp(parameters, states), finiteDiffConstraintPartialP(parameters, states, problem.constraint)],
      [problem.derivatives.drdp(parameters, states), finiteDiffResidualPartialP(parameters, states, problem.residual)]
    ]) expect(Math.max(...analytical.sub(numerical).to1DArray().map(Math.abs))).toBeLessThan(DERIVATIVE_TOLERANCE);
    const reference = implicitGradient(problem.source, problem.fullParameters(parameters),
      oracleStates(problem.fullParameters(parameters), problem.source.phases), 'analytic').slice(0, DESIGN_DIMENSION);
    for (const difference of [false, true]) {
      const gradient = createTrajectoryEvaluator(problem, difference).gradient(parameters);
      expect(vectorNorm(gradient.map((value, index) => value - reference[index]))).toBeLessThan(DERIVATIVE_TOLERANCE);
    }
  });

  it.each(TARGET_CASES)('passes all three initial designs and methods on %s targets', targetCase => {
    for (const initial of INITIAL_DESIGNS) for (const method of FIT_METHODS) {
      const problem = createTrajectoryProblem(targetCase, initial);
      const outcome = fitTrajectory(problem, method);
      const verification = verifyTrajectoryFit(problem, outcome);
      expect(verification.passed, `${targetCase}: ${initial}: ${method}: ${JSON.stringify(verification)}`).toBe(true);
      expect(outcome.result.converged).toBe(true);
      expect(outcome.snapshots[0].parameters).toEqual(Array.from(initial));
      expect(outcome.snapshots[outcome.snapshots.length - 1].parameters).toEqual(Array.from(outcome.result.finalParameters));
      for (const snapshot of outcome.snapshots) {
        expect(Array.from(problem.fullParameters(new Float64Array(snapshot.parameters)).slice(DESIGN_DIMENSION)))
          .toEqual(Array.from(problem.source.targetParameters.slice(DESIGN_DIMENSION)));
      }
    }
  });

  it('rejects a false history or wrong returned assembly even if the solver flag says converged', () => {
    const problem = createTrajectoryProblem();
    const outcome = fitTrajectory(problem);
    outcome.snapshots[0].cost += COST_CORRUPTION;
    expect(verifyTrajectoryFit(problem, outcome).passed).toBe(false);
    outcome.snapshots[0].cost -= COST_CORRUPTION;
    outcome.states[0] += WRONG_BRANCH_ANGLE;
    expect(verifyTrajectoryFit(problem, outcome).passed).toBe(false);
  });
});
