/** Public BFGS/IFT, full finite differences, and constrained LM with parameter snapshots for replay. */

import { Matrix } from 'ml-matrix';
import { bfgs, computeEffectiveJacobian, constrainedLevenbergMarquardt, finiteDiffGradient,
  type OptimizationResult } from '../../src/index';
import { assembleStates, COORDINATE_COUNT, createWorkCounters, DIFFERENCE_STEP,
  type WorkCounters } from '../mechanical-character/four-bar';
import { createPhaseFunctions, type TrajectoryProblem } from './model';

export const FIT_METHODS = ['BFGS + public IFT', 'BFGS + full FD', 'Constrained LM'] as const;
export type FitMethod = typeof FIT_METHODS[number];
export const FIT_GRADIENT_TOLERANCE = 1e-10;
const MAXIMUM_ITERATIONS = 400;
const CONSTRAINT_TOLERANCE = 1e-10;
const STEP_TOLERANCE = 1e-12;
const RESIDUAL_TOLERANCE = 1e-12;
const HALF_SQUARED_RESIDUAL = 0.5;
export type FitSnapshot = { iteration: number; cost: number; parameters: number[] };
export type FitOutcome = { result: OptimizationResult; states: Float64Array; snapshots: FitSnapshot[]; counters: WorkCounters };

export function createTrajectoryEvaluator(problem: TrajectoryProblem, finiteDifference: boolean = false) {
  const counters = createWorkCounters();
  const phaseFunctions = problem.source.phases.map((_, index) => createPhaseFunctions(problem, index));
  let cachedParameters: Float64Array | undefined;
  let cachedStates: Float64Array;
  const statesAt = (parameters: Float64Array): Float64Array => {
    if (!cachedParameters || !parameters.every((value, index) => value === cachedParameters![index])) {
      cachedStates = assembleStates(problem.fullParameters(parameters), problem.source.phases, counters);
      cachedParameters = parameters.slice();
    }
    return cachedStates;
  };
  const cost = (parameters: Float64Array) => problem.cost(parameters, statesAt(parameters));
  const implicitGradient = (parameters: Float64Array): Float64Array => {
    const states = statesAt(parameters);
    const gradient = new Float64Array(parameters.length);
    phaseFunctions.forEach((functions, index) => {
      const angles = states.subarray(COORDINATE_COUNT * index, COORDINATE_COUNT * (index + 1));
      const constraint = (trial: Float64Array, fixedStates: Float64Array) => {
        counters.constraintEvaluations++;
        return functions.constraint(trial, fixedStates);
      };
      // Like section 4.2, only C_p is differenced, while the state stays fixed.
      const jacobian = computeEffectiveJacobian(parameters, angles, functions.residual, constraint,
        { drdp: functions.drdp, drdx: functions.drdx, dcdx: functions.dcdx, stepSizeP: DIFFERENCE_STEP });
      const partial = jacobian.transpose().mmul(Matrix.columnVector(functions.residual(parameters, angles))).to1DArray();
      partial.forEach((value, column) => { gradient[column] += value; });
    });
    return gradient;
  };
  const gradient = finiteDifference
    ? (parameters: Float64Array) => finiteDiffGradient(parameters, cost, { stepSize: DIFFERENCE_STEP })
    : implicitGradient;
  return { statesAt, cost, gradient, counters };
}

export function fitTrajectory(problem: TrajectoryProblem, method: FitMethod = FIT_METHODS[0]): FitOutcome {
  const snapshots: FitSnapshot[] = [];
  // Native LM reports ||r||^2; the example compares all histories as 1/2 ||r||^2.
  const reportedCostScale = method === 'Constrained LM' ? HALF_SQUARED_RESIDUAL : 1;
  const onIteration = (iteration: number, cost: number, parameters: Float64Array): void => {
    snapshots.push({ iteration, cost: cost * reportedCostScale, parameters: Array.from(parameters) });
  };
  const options = { maxIterations: MAXIMUM_ITERATIONS, tolerance: FIT_GRADIENT_TOLERANCE, onIteration };
  if (method !== 'Constrained LM') {
    const evaluator = createTrajectoryEvaluator(problem, method === 'BFGS + full FD');
    const result = bfgs(problem.initialParameters, evaluator.cost, evaluator.gradient, options);
    return finish(result, evaluator.statesAt(result.finalParameters), snapshots, evaluator.counters);
  }
  const counters = createWorkCounters();
  const initialStates = assembleStates(problem.fullParameters(problem.initialParameters), problem.source.phases, counters);
  const constraint = (parameters: Float64Array, states: Float64Array) => {
    counters.constraintEvaluations += problem.source.phases.length;
    return problem.constraint(parameters, states);
  };
  const result = constrainedLevenbergMarquardt(problem.initialParameters, initialStates, problem.residual, constraint,
    { ...options, ...problem.derivatives, tolGradient: FIT_GRADIENT_TOLERANCE, tolStep: STEP_TOLERANCE,
      tolResidual: RESIDUAL_TOLERANCE, constraintTolerance: CONSTRAINT_TOLERANCE });
  return finish(result, result.finalStates, snapshots, counters, reportedCostScale);
}

function finish(result: OptimizationResult, states: Float64Array, snapshots: FitSnapshot[], counters: WorkCounters,
  reportedCostScale: number = 1): FitOutcome {
  const last = snapshots[snapshots.length - 1];
  if (!last || !result.finalParameters.every((value, index) => value === last.parameters[index])) {
    snapshots.push({ iteration: result.iterations, cost: result.finalCost * reportedCostScale, parameters: Array.from(result.finalParameters) });
  }
  return { result, states, snapshots, counters };
}
