/** Fixed four-bar topology, fixed ground and marker, with only three moving link lengths adjustable. */

import { Matrix } from 'ml-matrix';
import {
  createFourBarProblem, DEFAULT_SAMPLE_COUNT, markerParameterJacobian, markerPosition, markerStateJacobian,
  phaseConstraint, phaseConstraintStateJacobian
} from '../mechanical-character/four-bar';

export const DESIGN_DIMENSION = 3;
export const INITIAL_DESIGNS = [[-0.9, 0.8, -0.7], [0, 0, 0], [0.8, -0.7, 0.9]] as const;
export const TARGET_CASES = ['reachable', 'distorted'] as const;
export type TargetCase = typeof TARGET_CASES[number];
export type TrajectoryProblem = ReturnType<typeof createTrajectoryProblem>;

function designColumns(matrix: Matrix): Matrix {
  return matrix.subMatrix(0, matrix.rows - 1, 0, DESIGN_DIMENSION - 1);
}

export function createTrajectoryProblem(
  targetCase: TargetCase = 'reachable', initialDesign: readonly number[] = INITIAL_DESIGNS[0],
  sampleCount: number = DEFAULT_SAMPLE_COUNT
) {
  if (initialDesign.length !== DESIGN_DIMENSION || !initialDesign.every(Number.isFinite)) {
    throw new Error('Exactly three finite design parameters are required');
  }
  const source = createFourBarProblem(sampleCount, targetCase === 'distorted');
  // The ground and marker coordinates have no columns in the design space.
  const fixedParameters = source.targetParameters.slice();
  const fullParameters = (design: Float64Array): Float64Array => {
    if (design.length !== DESIGN_DIMENSION || !design.every(Number.isFinite)) {
      throw new Error('Exactly three finite design parameters are required');
    }
    const full = fixedParameters.slice();
    full.set(design);
    return full;
  };
  const residual = (design: Float64Array, states: Float64Array) => source.residual(fullParameters(design), states);
  const constraint = (design: Float64Array, states: Float64Array) => source.constraint(fullParameters(design), states);
  const cost = (design: Float64Array, states: Float64Array) => source.cost(fullParameters(design), states);
  const derivatives = {
    drdp: (design: Float64Array, states: Float64Array) => designColumns(source.derivatives.drdp(fullParameters(design), states)),
    drdx: (design: Float64Array, states: Float64Array) => source.derivatives.drdx(fullParameters(design), states),
    dcdp: (design: Float64Array, states: Float64Array) => designColumns(source.derivatives.dcdp(fullParameters(design), states)),
    dcdx: (design: Float64Array, states: Float64Array) => source.derivatives.dcdx(fullParameters(design), states)
  };
  return { targetCase, source, fullParameters, residual, constraint, cost, derivatives,
    initialParameters: new Float64Array(initialDesign) };
}

/** Phase blocks avoid coupling independent poses into one large sensitivity solve. */
export function createPhaseFunctions(problem: TrajectoryProblem, phaseIndex: number) {
  const phase = problem.source.phases[phaseIndex];
  const scale = 1 / Math.sqrt(problem.source.phases.length);
  return {
    residual: (design: Float64Array, states: Float64Array) => markerPosition(problem.fullParameters(design), states, phase)
      .map((value, coordinate) => scale * (value - problem.source.target[phaseIndex][coordinate])),
    constraint: (design: Float64Array, states: Float64Array) => phaseConstraint(problem.fullParameters(design), states, phase),
    drdp: (design: Float64Array, states: Float64Array) =>
      designColumns(markerParameterJacobian(problem.fullParameters(design), states, phase)).mul(scale),
    drdx: (design: Float64Array, states: Float64Array) => markerStateJacobian(problem.fullParameters(design), states).mul(scale),
    dcdx: (design: Float64Array, states: Float64Array) => phaseConstraintStateJacobian(problem.fullParameters(design), states)
  };
}
