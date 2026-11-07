/** Independent circle-geometry checks on training and unused phases, including every recorded design. */

import { vectorNorm } from '../../src/index';
import { assembleStates, createPhases, decodeLengths, implicitGradient, oracleStates, traceCurve } from '../mechanical-character/four-bar';
import { DESIGN_DIMENSION, type TrajectoryProblem } from './model';
import { type FitOutcome, type FitSnapshot } from './fit';

export const HELD_OUT_PHASE_COUNT = 257;
export const POSITION_TOLERANCE = 1e-5;
// A half-step with 257 samples would still share phase pi with the 32-sample design grid.
export const HELD_OUT_OFFSET = 0.37;
const CLOSURE_TOLERANCE = 1e-9;
const GRADIENT_TOLERANCE = 1e-8;
const MONOTONIC_COST_TOLERANCE = 1e-12;
const MINIMUM_DISTORTED_COST_REDUCTION = 0.9;
const MINIMUM_DISTORTED_ERROR = 0.01;

function distanceMetrics(curve: number[][], target: number[][]) {
  const distances = curve.map((point, index) => Math.hypot(...point.map((value, coordinate) => value - target[index][coordinate])));
  return { rms: Math.hypot(...distances) / Math.sqrt(distances.length), maximum: Math.max(...distances) };
}

export function verifyTrajectoryFit(problem: TrajectoryProblem, outcome: FitOutcome) {
  const full = problem.fullParameters(outcome.result.finalParameters);
  const exactStates = oracleStates(full, problem.source.phases);
  const heldOutPhases = createPhases(HELD_OUT_PHASE_COUNT, HELD_OUT_OFFSET);
  const heldOutStates = assembleStates(full, heldOutPhases);
  const heldOutOracle = oracleStates(full, heldOutPhases);
  const heldOut = distanceMetrics(traceCurve(full, heldOutOracle, heldOutPhases), problem.source.targetAt(heldOutPhases));
  const branchError = distanceMetrics(traceCurve(full, heldOutStates, heldOutPhases), traceCurve(full, heldOutOracle, heldOutPhases)).rms;
  const initialFull = problem.fullParameters(problem.initialParameters);
  const initialCost = problem.source.cost(initialFull, oracleStates(initialFull, problem.source.phases));
  const finalCost = problem.source.cost(full, exactStates);
  const gradientNorm = vectorNorm(implicitGradient(problem.source, full, exactStates, 'analytic').slice(0, DESIGN_DIMENSION));
  const constraintNorm = vectorNorm(problem.constraint(outcome.result.finalParameters, outcome.states));
  const trainingBranchError = distanceMetrics(traceCurve(full, outcome.states, problem.source.phases),
    traceCurve(full, exactStates, problem.source.phases)).rms;
  const costReduction = 1 - finalCost / initialCost;
  const history = verifySnapshotHistory(problem, outcome.snapshots);
  const trainingError = Math.sqrt(2 * finalCost);
  const accuracyPassed = problem.targetCase === 'reachable'
    ? trainingError <= POSITION_TOLERANCE && heldOut.maximum <= POSITION_TOLERANCE
    : costReduction >= MINIMUM_DISTORTED_COST_REDUCTION && heldOut.rms >= MINIMUM_DISTORTED_ERROR && gradientNorm <= GRADIENT_TOLERANCE;
  const finite = [initialCost, finalCost, gradientNorm, constraintNorm, trainingBranchError, branchError, heldOut.rms, heldOut.maximum].every(Number.isFinite);
  const passed = finite && accuracyPassed && constraintNorm <= CLOSURE_TOLERANCE
    && Math.max(trainingBranchError, branchError) <= CLOSURE_TOLERANCE && history.passed;
  return { passed, solverConverged: outcome.result.converged, iterations: outcome.result.iterations,
    initialError: Math.sqrt(2 * initialCost), trainingError, heldOutError: heldOut.rms, maximumHeldOutError: heldOut.maximum,
    initialCost, finalCost, costReduction, constraintNorm, gradientNorm, branchError, trainingBranchError,
    initialLengths: decodeLengths(initialFull), finalLengths: decodeLengths(full), history };
}

/** Recompute costs, closure and assembly branch: recorded solver flags alone do not constitute success. */
function verifySnapshotHistory(problem: TrajectoryProblem, snapshots: FitSnapshot[]) {
  let previousCost = Number.POSITIVE_INFINITY;
  let maximumCostDisagreement = 0;
  let maximumConstraintNorm = 0;
  let maximumBranchError = 0;
  let monotonic = true;
  for (const snapshot of snapshots) {
    const full = problem.fullParameters(new Float64Array(snapshot.parameters));
    const numericalStates = assembleStates(full, problem.source.phases);
    const exactStates = oracleStates(full, problem.source.phases);
    const cost = problem.source.cost(full, exactStates);
    maximumCostDisagreement = Math.max(maximumCostDisagreement, Math.abs(cost - snapshot.cost));
    maximumConstraintNorm = Math.max(maximumConstraintNorm, vectorNorm(problem.source.constraint(full, numericalStates)));
    maximumBranchError = Math.max(maximumBranchError, distanceMetrics(traceCurve(full, numericalStates, problem.source.phases),
      traceCurve(full, exactStates, problem.source.phases)).rms);
    if (cost > previousCost + MONOTONIC_COST_TOLERANCE) monotonic = false;
    previousCost = cost;
  }
  const passed = snapshots.length > 0 && monotonic && maximumCostDisagreement <= MONOTONIC_COST_TOLERANCE
    && maximumConstraintNorm <= CLOSURE_TOLERANCE && maximumBranchError <= CLOSURE_TOLERANCE;
  return { passed, monotonic, snapshotCount: snapshots.length, maximumCostDisagreement, maximumConstraintNorm, maximumBranchError };
}
