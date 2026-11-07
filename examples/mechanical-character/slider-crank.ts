/** Slider-crank fixture with a prismatic joint, numerical assembly, and independent geometric verification. */

import { Matrix } from 'ml-matrix';
import { bfgs, computeEffectiveJacobian, constrainedLevenbergMarquardt, gaussNewton, vectorNorm } from '../../src/index';
import { createPhases } from './four-bar';

const STATE_COUNT_PER_PHASE = 2;
const PARAMETER_COUNT = 2;
const SLIDER_INDEX = 1;
const LENGTH_CENTERS = [0.55, 1.6];
const LENGTH_RANGES = [0.2, 0.3];
const TARGET_PARAMETERS = [0.4, -0.35];
const ASSEMBLY_SOLVER_TOLERANCE = 1e-13;
const ASSEMBLY_TOLERANCE = 1e-10;
const MAXIMUM_ASSEMBLY_ITERATIONS = 40;
const MAXIMUM_DESIGN_ITERATIONS = 300;
const GRADIENT_TOLERANCE = 1e-9;
const STEP_TOLERANCE = 1e-12;
const TRAJECTORY_TOLERANCE = 1e-5;
const HELD_OUT_SAMPLE_COUNT = 257;
const HELD_OUT_PHASE_OFFSET = 0.5;
const LEAST_SQUARES_COST_FACTOR = 0.5;

export const SLIDER_METHODS = ['BFGS + public IFT', 'Constrained LM'] as const;
export type SliderMethod = typeof SLIDER_METHODS[number];
export type SliderGeometry = { crank: number; rod: number };
export type SliderProblem = ReturnType<typeof createSliderCrankProblem>;

function decodeGeometry(parameters: Float64Array): SliderGeometry {
  const lengths = LENGTH_CENTERS.map((center, index) => center + LENGTH_RANGES[index] * Math.tanh(parameters[index]));
  return { crank: lengths[0], rod: lengths[1] };
}

export function sliderConstraint(geometry: SliderGeometry, states: Float64Array, phase: number): Float64Array {
  return new Float64Array([
    geometry.crank * Math.cos(phase) + geometry.rod * Math.cos(states[0]) - states[SLIDER_INDEX],
    geometry.crank * Math.sin(phase) + geometry.rod * Math.sin(states[0])
  ]);
}

export function sliderStateJacobian(geometry: SliderGeometry, states: Float64Array): Matrix {
  return new Matrix([[-geometry.rod * Math.sin(states[0]), -1], [geometry.rod * Math.cos(states[0]), 0]]);
}

/** Partials with respect to physical [crank, rod] lengths, used independently in the near-toggle probe. */
export function sliderPhysicalParameterJacobian(states: Float64Array, phase: number): Matrix {
  return new Matrix([[Math.cos(phase), Math.cos(states[0])], [Math.sin(phase), Math.sin(states[0])]]);
}

export function sliderOracle(geometry: SliderGeometry, phase: number): Float64Array {
  const vertical = geometry.crank * Math.sin(phase);
  const discriminant = geometry.rod ** 2 - vertical ** 2;
  if (!(geometry.crank > 0) || !(geometry.rod > 0) || !(discriminant > 0)) {
    throw new Error('Singular or infeasible slider-crank configuration');
  }
  return new Float64Array([-Math.asin(vertical / geometry.rod),
    geometry.crank * Math.cos(phase) + Math.sqrt(discriminant)]);
}

function oracleCycle(parameters: Float64Array, phases: number[]): Float64Array {
  const geometry = decodeGeometry(parameters);
  return Float64Array.from(phases.flatMap(phase => Array.from(sliderOracle(geometry, phase))));
}

function assembleCycle(parameters: Float64Array, phases: number[]): Float64Array {
  const geometry = decodeGeometry(parameters);
  const states = new Float64Array(STATE_COUNT_PER_PHASE * phases.length);
  let previous: Float64Array = new Float64Array([0, geometry.crank + geometry.rod]);
  phases.forEach((phase, index) => {
    const result = gaussNewton(previous, angles => sliderConstraint(geometry, angles, phase), {
      jacobian: angles => sliderStateJacobian(geometry, angles), tolerance: ASSEMBLY_SOLVER_TOLERANCE,
      maxIterations: MAXIMUM_ASSEMBLY_ITERATIONS
    });
    previous = result.finalParameters;
    const norm = vectorNorm(sliderConstraint(geometry, previous, phase));
    if (!Number.isFinite(norm) || norm > ASSEMBLY_TOLERANCE || Math.cos(previous[0]) <= 0) {
      throw new Error(`Slider-crank assembly failed at phase ${phase}`);
    }
    states.set(previous, STATE_COUNT_PER_PHASE * index);
  });
  return states;
}

export function createSliderCrankProblem(sampleCount: number, initialParameters = new Float64Array(PARAMETER_COUNT)) {
  const phases = createPhases(sampleCount);
  const targetParameters = new Float64Array(TARGET_PARAMETERS);
  const targetStates = oracleCycle(targetParameters, phases);
  const residualScale = 1 / Math.sqrt(sampleCount);
  const residual = (_parameters: Float64Array, states: Float64Array): Float64Array => Float64Array.from(phases,
    (_phase, index) => residualScale * (states[STATE_COUNT_PER_PHASE * index + SLIDER_INDEX]
      - targetStates[STATE_COUNT_PER_PHASE * index + SLIDER_INDEX]));
  const constraint = (parameters: Float64Array, states: Float64Array): Float64Array => {
    const geometry = decodeGeometry(parameters);
    const values = new Float64Array(states.length);
    phases.forEach((phase, index) => values.set(sliderConstraint(geometry,
      states.subarray(STATE_COUNT_PER_PHASE * index, STATE_COUNT_PER_PHASE * (index + 1)), phase), STATE_COUNT_PER_PHASE * index));
    return values;
  };
  const derivatives = {
    drdp: () => Matrix.zeros(sampleCount, PARAMETER_COUNT),
    drdx: () => {
      const matrix = Matrix.zeros(sampleCount, STATE_COUNT_PER_PHASE * sampleCount);
      phases.forEach((_phase, index) => matrix.set(index, STATE_COUNT_PER_PHASE * index + SLIDER_INDEX, residualScale));
      return matrix;
    },
    dcdx: (parameters: Float64Array, states: Float64Array) => {
      const matrix = Matrix.zeros(states.length, states.length);
      phases.forEach((_phase, index) => matrix.setSubMatrix(sliderStateJacobian(decodeGeometry(parameters),
        states.subarray(STATE_COUNT_PER_PHASE * index, STATE_COUNT_PER_PHASE * (index + 1))),
      STATE_COUNT_PER_PHASE * index, STATE_COUNT_PER_PHASE * index));
      return matrix;
    },
    dcdp: (parameters: Float64Array, states: Float64Array) => {
      const matrix = Matrix.zeros(states.length, PARAMETER_COUNT);
      phases.forEach((phase, index) => {
        const block = sliderPhysicalParameterJacobian(
          states.subarray(STATE_COUNT_PER_PHASE * index, STATE_COUNT_PER_PHASE * (index + 1)), phase);
        for (let column = 0; column < PARAMETER_COUNT; column++) {
          block.mulColumn(column, LENGTH_RANGES[column] * (1 - Math.tanh(parameters[column]) ** 2));
        }
        matrix.setSubMatrix(block, STATE_COUNT_PER_PHASE * index, 0);
      });
      return matrix;
    }
  };
  const cost = (parameters: Float64Array, states: Float64Array): number =>
    LEAST_SQUARES_COST_FACTOR * residual(parameters, states).reduce((sum, value) => sum + value * value, 0);
  return { phases, initialParameters, targetParameters, residual, constraint, cost, derivatives };
}

export function runSliderCrank(problem: SliderProblem, method: SliderMethod) {
  const initialStates = assembleCycle(problem.initialParameters, problem.phases);
  if (method === 'Constrained LM') {
    const result = constrainedLevenbergMarquardt(problem.initialParameters, initialStates, problem.residual, problem.constraint,
      { ...problem.derivatives, constraintTolerance: ASSEMBLY_TOLERANCE, tolGradient: GRADIENT_TOLERANCE,
        tolStep: STEP_TOLERANCE, tolResidual: STEP_TOLERANCE, maxIterations: MAXIMUM_DESIGN_ITERATIONS });
    return { result, states: result.finalStates };
  }
  let cachedParameters = problem.initialParameters.slice();
  let cachedStates = initialStates;
  const statesAt = (parameters: Float64Array): Float64Array => {
    if (!parameters.every((value, index) => value === cachedParameters[index])) {
      cachedStates = assembleCycle(parameters, problem.phases);
      cachedParameters = parameters.slice();
    }
    return cachedStates;
  };
  const cost = (parameters: Float64Array): number => problem.cost(parameters, statesAt(parameters));
  const gradient = (parameters: Float64Array): Float64Array => {
    const states = statesAt(parameters);
    const matrix = computeEffectiveJacobian(parameters, states, problem.residual, problem.constraint, problem.derivatives);
    return Float64Array.from(matrix.transpose().mmul(Matrix.columnVector(problem.residual(parameters, states))).to1DArray());
  };
  const result = bfgs(problem.initialParameters, cost, gradient,
    { maxIterations: MAXIMUM_DESIGN_ITERATIONS, tolerance: GRADIENT_TOLERANCE });
  return { result, states: statesAt(result.finalParameters) };
}

export function verifySliderCrank(problem: SliderProblem, outcome: ReturnType<typeof runSliderCrank>) {
  const parameters = outcome.result.finalParameters;
  const phases = createPhases(HELD_OUT_SAMPLE_COUNT, HELD_OUT_PHASE_OFFSET);
  const exact = oracleCycle(parameters, phases);
  const target = oracleCycle(problem.targetParameters, phases);
  const heldOutError = Math.sqrt(phases.reduce((sum, _phase, index) => sum +
    (exact[STATE_COUNT_PER_PHASE * index + SLIDER_INDEX] - target[STATE_COUNT_PER_PHASE * index + SLIDER_INDEX]) ** 2, 0) / phases.length);
  const constraintNorm = vectorNorm(problem.constraint(parameters, outcome.states));
  const trainingStates = oracleCycle(parameters, problem.phases);
  const stateError = vectorNorm(outcome.states.map((value, index) => value - trainingStates[index]));
  const finalCost = problem.cost(parameters, trainingStates);
  const passed = [heldOutError, constraintNorm, stateError, finalCost].every(Number.isFinite)
    && heldOutError <= TRAJECTORY_TOLERANCE && constraintNorm <= ASSEMBLY_TOLERANCE && stateError <= TRAJECTORY_TOLERANCE;
  return { passed, solverConverged: outcome.result.converged, iterations: outcome.result.iterations,
    finalCost, heldOutError, constraintNorm, stateError };
}
