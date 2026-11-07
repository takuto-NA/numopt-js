/** Initial-value sweeps and failure probes for four-bars and slider-cranks, verified independently of solver flags. */

import { Matrix } from 'ml-matrix';
import { adjointBfgs, computeEffectiveJacobian, constrainedGaussNewton, constrainedLevenbergMarquardt, vectorNorm } from '../../src/index';
import { createFourBarProblem } from './four-bar';
import { runMechanismSolver, verifyMechanismOutcome, type Method } from './solvers';
import { createSliderCrankProblem, runSliderCrank, SLIDER_METHODS, sliderConstraint, sliderOracle,
  sliderPhysicalParameterJacobian, sliderStateJacobian, verifySliderCrank } from './slider-crank';

const SAMPLE_COUNT = 24;
const FOUR_BAR_INITIALIZATIONS = [
  [0, 0, 0, 0, 2.35, 0.25], [-0.8, 0.7, -0.7, 0.8, 2.0, 0.1], [0.6, -0.5, 0.8, -0.7, 2.8, 0.65]
];
const FOUR_BAR_METHODS: Method[] = ['BFGS + public IFT', 'Adjoint BFGS', 'Constrained LM'];
const SLIDER_INITIALIZATIONS = [[0, 0], [-0.7, 0.6], [0.8, -0.5]];
const TOGGLE_GAPS = [1e-2, 1e-4, 1e-6];
const PROBE_CRANK = 0.5;
const PROBE_PHASE = Math.PI / 2;
const RELATIVE_PERTURBATION = 1e-3;
const ANALYTICAL_RELATIVE_TOLERANCE = 1e-8;
const NUMERICAL_RELATIVE_TOLERANCE = 1e-5;
const CONSTRAINT_CONTRADICTION = 0.1;
const CONSTRAINT_TOLERANCE = 1e-9;

export function evaluateFourBarInitializations() {
  return FOUR_BAR_INITIALIZATIONS.flatMap((initial, initialIndex) => [false, true].flatMap(distorted =>
    FOUR_BAR_METHODS.map(method => {
      const problem = { ...createFourBarProblem(SAMPLE_COUNT, distorted), initialParameters: new Float64Array(initial) };
      const outcome = runMechanismSolver(problem, method);
      return { mechanism: 'four-bar', case: `${distorted ? 'distorted' : 'reachable'}/start-${initialIndex}`,
        method, initial, ...verifyMechanismOutcome(problem, outcome, distorted) };
    })));
}

export function evaluateSliderInitializations() {
  return SLIDER_INITIALIZATIONS.flatMap((initial, initialIndex) => SLIDER_METHODS.map(method => {
    const problem = createSliderCrankProblem(SAMPLE_COUNT, new Float64Array(initial));
    const outcome = runSliderCrank(problem, method);
    return { mechanism: 'slider-crank', case: `reachable/start-${initialIndex}`, method, initial,
      ...verifySliderCrank(problem, outcome) };
  }));
}

function probeNearToggle(gap: number) {
  const parameters = new Float64Array([PROBE_CRANK, PROBE_CRANK + gap]);
  const geometry = { crank: parameters[0], rod: parameters[1] };
  const states = sliderOracle(geometry, PROBE_PHASE);
  const matrix = computeEffectiveJacobian(parameters, states, (_parameters, angles) => new Float64Array([angles[1]]),
    (trial, angles) => sliderConstraint({ crank: trial[0], rod: trial[1] }, angles, PROBE_PHASE), {
      drdp: () => Matrix.zeros(1, 2), drdx: () => new Matrix([[0, 1]]),
      dcdp: (_trial, angles) => sliderPhysicalParameterJacobian(angles, PROBE_PHASE),
      dcdx: (trial, angles) => sliderStateJacobian({ crank: trial[0], rod: trial[1] }, angles)
    });
  const height = Math.sqrt(geometry.rod ** 2 - geometry.crank ** 2);
  const expected = [-geometry.crank / height, geometry.rod / height];
  const perturbation = gap * RELATIVE_PERTURBATION;
  const errors = expected.map((value, column) => {
    const plus = parameters.slice();
    const minus = parameters.slice();
    plus[column] += perturbation;
    minus[column] -= perturbation;
    const numerical = (sliderOracle({ crank: plus[0], rod: plus[1] }, PROBE_PHASE)[1]
      - sliderOracle({ crank: minus[0], rod: minus[1] }, PROBE_PHASE)[1]) / (2 * perturbation);
    return { analytical: Math.abs(matrix.get(0, column) - value) / Math.max(1, Math.abs(value)),
      numerical: Math.abs(numerical - value) / Math.max(1, Math.abs(value)) };
  });
  const analyticalError = Math.max(...errors.map(error => error.analytical));
  const numericalError = Math.max(...errors.map(error => error.numerical));
  return { case: `near-toggle/gap-${gap}`, passed: analyticalError < ANALYTICAL_RELATIVE_TOLERANCE
      && numericalError < NUMERICAL_RELATIVE_TOLERANCE,
    detail: `IFT relative error=${analyticalError}, reassembled FD relative error=${numericalError}` };
}

function probeBoundary(gap: number) {
  let message = '';
  try {
    sliderOracle({ crank: PROBE_CRANK, rod: PROBE_CRANK + gap }, PROBE_PHASE);
  } catch (error) {
    message = error instanceof Error ? error.message : String(error);
  }
  return { case: gap === 0 ? 'exact-toggle/domain-rejection' : 'infeasible-slider/domain-rejection',
    passed: message.includes('Singular or infeasible'), detail: message || 'Expected a domain rejection' };
}

function probeInconsistentConstraints(method: 'GN' | 'LM') {
  const initial = new Float64Array([0.2]);
  const residual = (parameters: Float64Array, states: Float64Array): Float64Array =>
    new Float64Array([parameters[0] - 1, states[0] - 1]);
  const constraint = (parameters: Float64Array, states: Float64Array): Float64Array =>
    new Float64Array([states[0] - parameters[0], states[0] - parameters[0] - CONSTRAINT_CONTRADICTION]);
  const options = {
    maxIterations: 50, constraintTolerance: CONSTRAINT_TOLERANCE,
    drdp: () => new Matrix([[1], [0]]), drdx: () => new Matrix([[0], [1]]),
    dcdp: () => new Matrix([[-1], [-1]]), dcdx: () => new Matrix([[1], [1]])
  };
  const result = method === 'GN'
    ? constrainedGaussNewton(initial, initial, residual, constraint, options)
    : constrainedLevenbergMarquardt(initial, initial, residual, constraint, options);
  const norm = vectorNorm(constraint(result.finalParameters, result.finalStates));
  const minimumNorm = CONSTRAINT_CONTRADICTION / Math.SQRT2;
  return { case: `inconsistent-constraints/${method}`, passed: !result.converged && Number.isFinite(norm)
      && norm >= minimumNorm - CONSTRAINT_TOLERANCE,
    detail: `converged=${result.converged}, constraint norm=${norm}, unavoidable minimum=${minimumNorm}` };
}

function probeUnprojectableState() {
  let message = '';
  try {
    adjointBfgs(new Float64Array([1.5]), new Float64Array([0.5]),
      (parameters, states) => (parameters[0] - 1) ** 2 + (states[0] - 1) ** 2,
      (parameters, states) => new Float64Array([parameters[0] ** 2 + states[0] ** 2 - 2]));
  } catch (error) {
    message = error instanceof Error ? error.message : String(error);
  }
  return { case: 'unprojectable-state/adjoint', passed: message.includes('Failed to restore feasible states'), detail: message };
}

export function evaluateMechanismSafety() {
  return [...TOGGLE_GAPS.map(probeNearToggle), probeBoundary(0), probeBoundary(-1e-4),
    probeInconsistentConstraints('GN'), probeInconsistentConstraints('LM'), probeUnprojectableState()];
}
