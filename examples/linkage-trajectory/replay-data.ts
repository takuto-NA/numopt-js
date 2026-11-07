/** Reassemble recorded designs for trajectory and linkage replay; this work is outside optimizer timing. */

import { assembleStates, COORDINATE_COUNT, createPhases, decodeLengths, traceCurve } from '../mechanical-character/four-bar';
import { type TrajectoryProblem } from './model';
import { type FitOutcome } from './fit';
import { verifyTrajectoryFit } from './verify';

const REPLAY_PHASE_COUNT = 128;

export function createTrajectoryReplay(problem: TrajectoryProblem, outcome: FitOutcome) {
  const phases = createPhases(REPLAY_PHASE_COUNT);
  const snapshots = outcome.snapshots.map(snapshot => {
    const parameters = problem.fullParameters(new Float64Array(snapshot.parameters));
    const states = assembleStates(parameters, phases);
    return { ...snapshot, lengths: decodeLengths(parameters), curve: traceCurve(parameters, states, phases),
      couplerAngles: phases.map((_, index) => states[COORDINATE_COUNT * index]) };
  });
  const fixed = problem.fullParameters(problem.initialParameters);
  return { targetCase: problem.targetCase, phases, target: problem.source.targetAt(phases),
    marker: Array.from(fixed.slice(-COORDINATE_COUNT)), snapshots, verification: verifyTrajectoryFit(problem, outcome) };
}

export type TrajectoryReplay = ReturnType<typeof createTrajectoryReplay>;
