/** Public-solver adapters and independent accuracy checks for the four-bar experiment. */

import {
  adjointBfgs, bfgs, constrainedGaussNewton, constrainedLevenbergMarquardt, vectorNorm,
  type OptimizationResult
} from '../../src/index';
import {
  assembleStates, createPhases, createReducedEvaluator, createWorkCounters, implicitGradient,
  oracleStates, traceCurve, type FourBarProblem, type WorkCounters
} from './four-bar';

export const MAXIMUM_DESIGN_ITERATIONS = 600;
export const GRADIENT_TOLERANCE = 1e-9;
export const CONSTRAINT_TOLERANCE = 1e-9;
export const TRAJECTORY_TOLERANCE = 1e-5;
const MINIMUM_COST_REDUCTION = 0.95;
const HELD_OUT_SAMPLE_COUNT = 257;
const HELD_OUT_PHASE_OFFSET = 0.5;
const STEP_TOLERANCE = 1e-12;
const RESIDUAL_TOLERANCE = 1e-12;

export const METHODS = ['BFGS + IFT (paper)', 'BFGS + public IFT', 'BFGS + redundant IFT',
  'BFGS + full FD', 'Adjoint BFGS', 'Constrained GN', 'Constrained LM'] as const;
export type Method = typeof METHODS[number];
export type SolveOutcome = {
  result: OptimizationResult;
  states: Float64Array;
  counters: WorkCounters;
  history: number[];
};

export function runMechanismSolver(problem: FourBarProblem, method: Method): SolveOutcome {
  const counters = createWorkCounters();
  const history: number[] = [];
  const onIteration = (_iteration: number, cost: number): void => { history.push(cost); };
  const options = { maxIterations: MAXIMUM_DESIGN_ITERATIONS, tolerance: GRADIENT_TOLERANCE, onIteration };
  if (method.startsWith('BFGS +')) {
    const mode = method === 'BFGS + full FD' ? 'finite-difference'
      : method === 'BFGS + public IFT' ? 'public-ift'
      : method === 'BFGS + redundant IFT' ? 'redundant-ift' : 'paper';
    const evaluator = createReducedEvaluator(problem, mode, counters);
    const result = bfgs(problem.initialParameters, evaluator.cost, evaluator.gradient, options);
    return { result, states: evaluator.statesAt(result.finalParameters), counters, history };
  }
  const initialStates = assembleStates(problem.initialParameters, problem.phases, counters);
  const constraint = (parameters: Float64Array, states: Float64Array): Float64Array => {
    counters.constraintEvaluations += problem.phases.length;
    return problem.constraint(parameters, states);
  };
  const constrainedOptions = { ...options, ...problem.derivatives, constraintTolerance: CONSTRAINT_TOLERANCE };
  const result = method === 'Adjoint BFGS'
    ? adjointBfgs(problem.initialParameters, initialStates, problem.cost, constraint,
      { ...constrainedOptions, ...problem.costPartials })
    : method === 'Constrained GN'
      ? constrainedGaussNewton(problem.initialParameters, initialStates, problem.residual, constraint, constrainedOptions)
      : constrainedLevenbergMarquardt(problem.initialParameters, initialStates, problem.residual, constraint,
        { ...constrainedOptions, tolGradient: GRADIENT_TOLERANCE, tolStep: STEP_TOLERANCE, tolResidual: RESIDUAL_TOLERANCE });
  return { result, states: result.finalStates, counters, history };
}

function rootMeanSquaredDistance(curve: number[][], target: number[][]): number {
  return Math.sqrt(curve.reduce((sum, point, index) => sum + point.reduce((subtotal, value, coordinate) =>
    subtotal + (value - target[index][coordinate]) ** 2, 0), 0) / curve.length);
}

/** Verification uses circle geometry and a shifted dense phase grid, outside timed work. */
export function verifyMechanismOutcome(problem: FourBarProblem, outcome: SolveOutcome, distorted: boolean) {
  const { result, states } = outcome;
  const parameters = result.finalParameters;
  const exactStates = oracleStates(parameters, problem.phases);
  const initialStates = oracleStates(problem.initialParameters, problem.phases);
  const initialCost = problem.cost(problem.initialParameters, initialStates);
  const finalCost = problem.cost(parameters, exactStates);
  const constraintNorm = vectorNorm(problem.constraint(parameters, states));
  const gradientNorm = vectorNorm(implicitGradient(problem, parameters, exactStates, 'analytic'));
  const trainingError = Math.sqrt(2 * finalCost);
  const heldOutPhases = createPhases(HELD_OUT_SAMPLE_COUNT, HELD_OUT_PHASE_OFFSET);
  const heldOutError = rootMeanSquaredDistance(traceCurve(parameters, oracleStates(parameters, heldOutPhases), heldOutPhases),
    problem.targetAt(heldOutPhases));
  const stateCurveError = rootMeanSquaredDistance(traceCurve(parameters, states, problem.phases),
    traceCurve(parameters, exactStates, problem.phases));
  const costReduction = 1 - finalCost / initialCost;
  const accuracyPassed = distorted
    ? costReduction >= MINIMUM_COST_REDUCTION && gradientNorm <= GRADIENT_TOLERANCE * 10
    : trainingError <= TRAJECTORY_TOLERANCE && heldOutError <= TRAJECTORY_TOLERANCE;
  const passed = [finalCost, constraintNorm, gradientNorm, trainingError, heldOutError, stateCurveError]
    .every(Number.isFinite) && constraintNorm <= CONSTRAINT_TOLERANCE && stateCurveError <= TRAJECTORY_TOLERANCE && accuracyPassed;
  return { passed, solverConverged: result.converged, iterations: result.iterations, initialCost, finalCost,
    trainingError, heldOutError, constraintNorm, stateCurveError, gradientNorm, costReduction };
}
