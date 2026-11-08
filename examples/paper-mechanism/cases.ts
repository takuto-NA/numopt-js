/** Shared phase-level contract for paper-equation checks on six-parameter four-bars and six/eight-parameter five-bars. */

import { Matrix } from 'ml-matrix';
import {
  createFourBarProblem, createPhases, markerParameterJacobian, markerPosition, markerStateJacobian,
  oracleStates, phaseConstraint, phaseConstraintParameterJacobian, phaseConstraintStateJacobian
} from '../mechanical-character/four-bar';
import { FIVE_BAR_TARGET, fiveBarConstraint, fiveBarMarker, fiveBarMarkerParameterJacobian,
  fiveBarMarkerStateJacobian, fiveBarOracle, fiveBarParameterJacobian, fiveBarStateJacobian, GEAR_RATIOS } from './geared-five-bar';

export const PAPER_PHASE_COUNTS = [16, 32, 128] as const;
const DEFAULT_PHASE_COUNT = 32;
const FIVE_BAR_MINIMAL_COLUMNS = [0, 1, 2, 4, 6, 7];
const DISTORTION_AMPLITUDE = 0.02;
const DISTORTION_FREQUENCY = 3;
const COORDINATE_COUNT = 2;
export type PaperMechanism = {
  name: string;
  initialParameters: Float64Array;
  targetParameters: Float64Array;
  boundedParameterIndices: number[];
  phases: number[];
  targetAt: (phases: number[]) => number[][];
  constraint: (parameters: Float64Array, states: Float64Array, phase: number) => Float64Array;
  marker: (parameters: Float64Array, states: Float64Array, phase: number) => Float64Array;
  dcdp: (parameters: Float64Array, states: Float64Array, phase: number) => Matrix;
  dcdx: (parameters: Float64Array, states: Float64Array) => Matrix;
  dmdp: (parameters: Float64Array, states: Float64Array, phase: number) => Matrix;
  dmdx: (parameters: Float64Array, states: Float64Array) => Matrix;
  oracle: (parameters: Float64Array, phases: number[]) => Float64Array;
};

export function createPaperFourBar(sampleCount: number = DEFAULT_PHASE_COUNT, distorted: boolean = false): PaperMechanism {
  const source = createFourBarProblem(sampleCount, distorted);
  return { name: 'four-bar-6', initialParameters: source.initialParameters, targetParameters: source.targetParameters,
    boundedParameterIndices: [0, 1, 2, 3],
    phases: source.phases, targetAt: source.targetAt, constraint: phaseConstraint, marker: markerPosition,
    dcdp: phaseConstraintParameterJacobian, dcdx: phaseConstraintStateJacobian,
    dmdp: markerParameterJacobian, dmdx: markerStateJacobian, oracle: oracleStates };
}

export function createPaperFiveBar(
  ratio: number, parameterCount: 6 | 8 = 8, sampleCount: number = DEFAULT_PHASE_COUNT, distorted: boolean = false
): PaperMechanism {
  if (!GEAR_RATIOS.some(value => value === ratio)) throw new Error('Gear ratio must be -2, -1, 1, or 2');
  const columns = parameterCount === 6 ? FIVE_BAR_MINIMAL_COLUMNS : FIVE_BAR_TARGET.map((_, index) => index);
  const expand = (parameters: Float64Array) => {
    if (parameters.length !== columns.length) throw new Error('Wrong five-bar parameter dimension');
    const full = new Float64Array(FIVE_BAR_TARGET);
    columns.forEach((column, index) => { full[column] = parameters[index]; });
    return full;
  };
  const selectColumns = (matrix: Matrix) => matrix.selection([0, 1], columns);
  const targetParameters = Float64Array.from(columns, column => FIVE_BAR_TARGET[column]);
  const oracle = (parameters: Float64Array, phases: number[]) => fiveBarOracle(expand(parameters), phases, ratio);
  const marker = (parameters: Float64Array, states: Float64Array, phase: number) => fiveBarMarker(expand(parameters), states, phase);
  const targetAt = (phases: number[]) => {
    const states = oracle(targetParameters, phases);
    return phases.map((phase, index) => {
      const point = Array.from(marker(targetParameters, states.subarray(COORDINATE_COUNT * index, COORDINATE_COUNT * (index + 1)), phase));
      if (distorted) { point[0] += DISTORTION_AMPLITUDE * Math.cos(DISTORTION_FREQUENCY * phase);
        point[1] += DISTORTION_AMPLITUDE * Math.sin(DISTORTION_FREQUENCY * phase); }
      return point;
    });
  };
  return { name: `geared-five-bar-${parameterCount}:ratio=${ratio}`, initialParameters: new Float64Array(parameterCount), targetParameters,
    boundedParameterIndices: columns.map((_, index) => index),
    phases: createPhases(sampleCount), targetAt, oracle, marker,
    constraint: (parameters, states, phase) => fiveBarConstraint(expand(parameters), states, phase, ratio),
    dcdp: (parameters, states, phase) => selectColumns(fiveBarParameterJacobian(expand(parameters), states, phase, ratio)),
    dcdx: (parameters, states) => fiveBarStateJacobian(expand(parameters), states),
    dmdp: (parameters, states, phase) => selectColumns(fiveBarMarkerParameterJacobian(expand(parameters), states, phase)),
    dmdx: (parameters, states) => fiveBarMarkerStateJacobian(expand(parameters), states) };
}

export function createPaperCases(sampleCount: number = DEFAULT_PHASE_COUNT, distorted: boolean = false): PaperMechanism[] {
  return [createPaperFourBar(sampleCount, distorted), ...GEAR_RATIOS.flatMap(ratio =>
    ([6, 8] as const).map(count => createPaperFiveBar(ratio, count, sampleCount, distorted)))];
}
