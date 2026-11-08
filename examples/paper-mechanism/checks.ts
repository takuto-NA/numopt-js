/** Equation-level Taylor, redundancy, direct-marker, phase-quadrature, state-accuracy and independent fitting checks. */

import { SingularValueDecomposition, solve } from 'ml-matrix';
import { vectorNorm } from '../../src/index';
import { createPhases } from '../mechanical-character/four-bar';
import { type PaperMechanism } from './cases';
import { checkBoundaryStationarity } from './boundary-check';
import { assemblePaperStates, createPaperEvaluator, fitPaperMechanism, paperCost, paperCurve,
  paperGradientAt, STATE_DIMENSION, type PaperGradient } from './evaluate';

export const HELD_OUT_COUNT = 257;
export const HELD_OUT_OFFSET = 0.37;
export const POSITION_TOLERANCE = 1e-5;
export const GRADIENT_TOLERANCE = 1e-8;
const CLOSURE_TOLERANCE = 1e-9;
const COST_TOLERANCE = 1e-10;
const BRANCH_TOLERANCE = 1e-9;
const MINIMUM_TAYLOR_ORDER = 1.8;
const MAXIMUM_CONTROL_ORDER = 1.2;
const TAYLOR_STEPS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7];
const MARKER_PERTURBATION = 0.1;
const MINIMUM_MARKER_MOTION = 1e-3;
const REFERENCE_PHASE_COUNT = 4096;
const QUADRATURE_COUNTS = [8, 16, 32, 64, 128];
const STATE_TOLERANCES = [1e-2, 1e-4, 1e-8, 1e-13];
const SATURATION_MARGIN = 1e-6;
const INITIAL_DISPLACEMENT = 0.6;
const ALIGNMENT_PHASE_OFFSET = Math.PI / 4;

export function curveDistance(left: number[][], right: number[][]) {
  const distances = left.map((point, index) => Math.hypot(...point.map((value, coordinate) => value - right[index][coordinate])));
  return { rms: Math.hypot(...distances) / Math.sqrt(distances.length), maximum: Math.max(...distances) };
}

function gradientDistance(left: Float64Array, right: Float64Array) {
  return vectorNorm(left.map((value, index) => value - right[index]));
}

function maximumClosure(model: PaperMechanism, parameters: Float64Array, states: Float64Array, phases = model.phases) {
  return Math.max(...phases.map((phase, index) => vectorNorm(model.constraint(parameters,
    states.subarray(STATE_DIMENSION * index, STATE_DIMENSION * (index + 1)), phase))));
}

function exactCost(model: PaperMechanism, parameters: Float64Array, phases = model.phases) {
  return paperCost(model, parameters, model.oracle(parameters, phases), phases);
}

export function checkPaperFit(model: PaperMechanism, outcome: ReturnType<typeof fitPaperMechanism>, initial: Float64Array,
  distorted: boolean) {
  const parameters = outcome.result.finalParameters;
  const phases = createPhases(HELD_OUT_COUNT, HELD_OUT_OFFSET);
  const exactStates = model.oracle(parameters, model.phases);
  const heldOutStates = model.oracle(parameters, phases);
  const heldOutNumerical = assemblePaperStates(model, parameters, phases);
  const trainingCurve = paperCurve(model, parameters, exactStates);
  const heldOutCurve = paperCurve(model, parameters, heldOutStates, phases);
  const heldOut = curveDistance(heldOutCurve, model.targetAt(phases));
  const branchError = Math.max(curveDistance(trainingCurve, paperCurve(model, parameters, outcome.states)).maximum,
    curveDistance(heldOutCurve, paperCurve(model, parameters, heldOutNumerical, phases)).maximum);
  const closure = Math.max(maximumClosure(model, parameters, outcome.states), maximumClosure(model, parameters, heldOutNumerical, phases));
  const initialCost = exactCost(model, initial);
  const finalCost = exactCost(model, parameters);
  const gradientNorm = vectorNorm(paperGradientAt(model, parameters, exactStates, 'analytic-ift'));
  const historyCosts = outcome.history.map(snapshot => exactCost(model, new Float64Array(snapshot.parameters)));
  const historyValid = outcome.history.every((snapshot, index) => Math.abs(snapshot.cost - historyCosts[index]) <= COST_TOLERANCE
    && (index === 0 || historyCosts[index] <= historyCosts[index - 1] + COST_TOLERANCE));
  const periodic = model.oracle(parameters, [0, 2 * Math.PI]);
  const periodError = vectorNorm(model.marker(parameters, periodic.subarray(0, STATE_DIMENSION), 0)
    .map((value, coordinate) => value - model.marker(parameters, periodic.subarray(STATE_DIMENSION), 2 * Math.PI)[coordinate]));
  const accuracy = distorted ? finalCost < initialCost && gradientNorm <= GRADIENT_TOLERANCE
    : Math.sqrt(2 * finalCost) <= POSITION_TOLERANCE && heldOut.maximum <= POSITION_TOLERANCE;
  const boundaryStationarity = checkBoundaryStationarity(model, parameters, trial => exactCost(model, trial));
  const passed = accuracy && boundaryStationarity.passed && historyValid && closure <= CLOSURE_TOLERANCE && branchError <= BRANCH_TOLERANCE
    && periodError <= POSITION_TOLERANCE && [heldOut.rms, heldOut.maximum, finalCost, gradientNorm].every(Number.isFinite);
  const saturatedParameters = model.boundedParameterIndices.filter(index => Math.abs(Math.tanh(parameters[index])) >= 1 - SATURATION_MARGIN);
  return { passed, solverConverged: outcome.result.converged, iterations: outcome.result.iterations, initialCost, finalCost,
    initialError: Math.sqrt(2 * initialCost), trainingError: Math.sqrt(2 * finalCost), heldOutError: heldOut.rms,
    maximumHeldOutError: heldOut.maximum, costReduction: 1 - finalCost / initialCost, gradientNorm, closure, branchError,
    periodError, historyValid, saturatedParameters, boundaryStationarity,
    interiorStationary: saturatedParameters.length === 0 && gradientNorm <= GRADIENT_TOLERANCE,
    initialCurve: paperCurve(model, initial, model.oracle(initial, phases), phases), heldOutCurve, target: model.targetAt(phases) };
}

function taylorRemainders(model: PaperMechanism, parameters: Float64Array, direction: Float64Array,
  gradient: Float64Array, initialCost: number) {
  const slope = gradient.reduce((sum, value, index) => sum + value * direction[index], 0);
  const rows = TAYLOR_STEPS.map(step => {
    const trial = parameters.map((value, index) => value + step * direction[index]);
    return { step, remainder: Math.abs(exactCost(model, trial) - initialCost - step * slope) };
  });
  const orders = rows.slice(0, -1).map((row, index) => Math.log(row.remainder / rows[index + 1].remainder)
    / Math.log(row.step / rows[index + 1].step));
  return { rows, orders };
}

/** Two intentionally incomplete gradients are negative controls, not candidate optimization methods. */
export function checkPaperEquations(model: PaperMechanism) {
  const parameters = model.initialParameters;
  const states = model.oracle(parameters, model.phases);
  const reference = paperGradientAt(model, parameters, states, 'analytic-ift');
  const direction = reference.map(value => value / vectorNorm(reference));
  const gradients = (['paper-ift', 'full-fd', 'frozen-state', 'omit-direct'] as const).map(method => {
    const gradient = createPaperEvaluator(model, method).gradient(parameters);
    return { method, error: gradientDistance(gradient, reference), relativeError: gradientDistance(gradient, reference) / vectorNorm(reference),
      taylor: taylorRemainders(model, parameters, direction, gradient, exactCost(model, parameters)) };
  });
  const redundant = paperGradientAt(model, parameters, states, 'paper-ift', undefined, true);
  let sensitivityResidual = 0;
  let minimumStateSingularValue = Number.POSITIVE_INFINITY;
  model.phases.forEach((phase, index) => {
    const angles = states.subarray(STATE_DIMENSION * index, STATE_DIMENSION * (index + 1));
    const stateJacobian = model.dcdx(parameters, angles);
    const parameterJacobian = model.dcdp(parameters, angles, phase);
    const sensitivity = solve(stateJacobian, parameterJacobian).mul(-1);
    sensitivityResidual = Math.max(sensitivityResidual, vectorNorm(Float64Array.from(stateJacobian.mmul(sensitivity).add(parameterJacobian).to1DArray())));
    minimumStateSingularValue = Math.min(minimumStateSingularValue, ...new SingularValueDecomposition(stateJacobian).diagonal);
  });
  const markerOnly = parameters.slice();
  markerOnly[markerOnly.length - 1] += MARKER_PERTURBATION;
  const changedStates = model.oracle(markerOnly, model.phases);
  const markerStateChange = gradientDistance(states, changedStates);
  const markerMotion = curveDistance(paperCurve(model, parameters, states), paperCurve(model, markerOnly, changedStates)).rms;
  const targetStates = model.oracle(model.targetParameters, model.phases);
  const unalignedError = curveDistance(paperCurve(model, model.targetParameters, targetStates),
    model.targetAt(model.phases.map(phase => phase + ALIGNMENT_PHASE_OFFSET))).rms;
  const positive = gradients.slice(0, 2).every(row => row.relativeError <= POSITION_TOLERANCE && row.error <= GRADIENT_TOLERANCE
    && row.taylor.orders.slice(0, 2).every(order => order >= MINIMUM_TAYLOR_ORDER));
  const controls = gradients.slice(2).every(row => row.taylor.orders.slice(0, 2).every(order => order <= MAXIMUM_CONTROL_ORDER));
  const redundantError = gradientDistance(redundant, reference);
  const passed = positive && controls && redundantError <= GRADIENT_TOLERANCE && sensitivityResidual <= CLOSURE_TOLERANCE
    && minimumStateSingularValue > 0 && markerStateChange <= BRANCH_TOLERANCE && markerMotion >= MINIMUM_MARKER_MOTION
    && unalignedError > POSITION_TOLERANCE;
  return { name: model.name, passed, gradients, redundantError, sensitivityResidual, minimumStateSingularValue,
    markerStateChange, markerMotion, unalignedError };
}

export function checkStateAccuracy(model: PaperMechanism) {
  const exact = model.oracle(model.initialParameters, model.phases);
  const reference = paperGradientAt(model, model.initialParameters, exact, 'analytic-ift');
  return STATE_TOLERANCES.map(tolerance => {
    // Deliberately retain coarse, possibly infeasible states to measure the premise of Eq. (4).
    const states = assemblePaperStates(model, model.initialParameters, model.phases, undefined, tolerance, false);
    const gradient = paperGradientAt(model, model.initialParameters, states, 'paper-ift');
    const closure = maximumClosure(model, model.initialParameters, states);
    return { name: model.name, tolerance, closureMeetsTolerance: closure <= tolerance, maximumClosure: closure,
      gradientError: gradientDistance(gradient, reference) };
  });
}

/** Uses a dense independent geometric quadrature as reference, without adding any solver features. */
export function checkQuadrature(model: PaperMechanism) {
  const phases = createPhases(REFERENCE_PHASE_COUNT, HELD_OUT_OFFSET);
  const referenceModel = { ...model, phases };
  const referenceCost = exactCost(referenceModel, model.initialParameters);
  const referenceGradient = paperGradientAt(referenceModel, model.initialParameters, model.oracle(model.initialParameters, phases), 'analytic-ift');
  return QUADRATURE_COUNTS.map(sampleCount => {
    const active = { ...model, phases: createPhases(sampleCount) };
    const states = model.oracle(model.initialParameters, active.phases);
    return { name: model.name, sampleCount, referencePhaseCount: REFERENCE_PHASE_COUNT,
      costError: Math.abs(exactCost(active, model.initialParameters) - referenceCost),
      gradientError: gradientDistance(paperGradientAt(active, model.initialParameters, states, 'analytic-ift'), referenceGradient) };
  });
}

export function initialPaperDesigns(model: PaperMechanism): Float64Array[] {
  const displacement = (sign: number) => model.initialParameters.map((value, index) => value + sign * INITIAL_DISPLACEMENT * Math.sin(index + 1));
  return [model.initialParameters.slice(), displacement(1), displacement(-1)];
}

export function runPaperFit(model: PaperMechanism, method: PaperGradient, initial: Float64Array, distorted: boolean) {
  const outcome = fitPaperMechanism(model, method, initial);
  return { verification: checkPaperFit(model, outcome, initial, distorted), history: outcome.history,
    counters: outcome.counters, finalParameters: Array.from(outcome.result.finalParameters) };
}
