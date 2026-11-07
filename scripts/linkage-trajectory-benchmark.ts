/** Repeated three-length fitting trials with independent held-out and iteration-history verification. */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { collectPaperEnvironment, runWarmedRepeats, summarizeNumeric } from './benchmark-harness';
import { fitTrajectory, FIT_METHODS, type FitMethod } from '../examples/linkage-trajectory/fit';
import { createTrajectoryProblem, INITIAL_DESIGNS, TARGET_CASES, type TrajectoryProblem } from '../examples/linkage-trajectory/model';
import { HELD_OUT_OFFSET, verifyTrajectoryFit } from '../examples/linkage-trajectory/verify';

const WARMUP_COUNT = 1;
const REPEAT_COUNT = 3;
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT_DIRECTORY = path.join(ROOT, 'benchmark-results');

function runTrial(problem: TrajectoryProblem, method: FitMethod) {
  const start = performance.now();
  try {
    const outcome = fitTrajectory(problem, method);
    const elapsedMs = performance.now() - start;
    const verification = verifyTrajectoryFit(problem, outcome);
    return { elapsedMs, verification, snapshots: outcome.snapshots, finalParameters: Array.from(outcome.result.finalParameters),
      counters: outcome.counters, error: null };
  } catch (error) {
    return { elapsedMs: performance.now() - start, verification: null, snapshots: [], finalParameters: [], counters: null,
      error: error instanceof Error ? error.message : String(error) };
  }
}

function main(): void {
  const rows = TARGET_CASES.flatMap(targetCase => INITIAL_DESIGNS.flatMap((initial, initialIndex) => FIT_METHODS.map(method => {
    const problem = createTrajectoryProblem(targetCase, initial);
    const trials = runWarmedRepeats({ warmupCount: WARMUP_COUNT, repeatCount: REPEAT_COUNT, run: () => runTrial(problem, method) });
    return { targetCase, initialIndex, method, initialParameters: Array.from(initial), trials,
      successCount: trials.filter(trial => trial.verification?.passed).length,
      timeMs: summarizeNumeric(trials.map(trial => trial.elapsedMs)) };
  })));
  const sourceFiles = ['examples/linkage-trajectory/model.ts', 'examples/linkage-trajectory/fit.ts',
    'examples/linkage-trajectory/verify.ts', 'examples/mechanical-character/four-bar.ts', 'src/core/bfgs.ts',
    'src/core/bfgsUpdate.ts', 'src/core/curvature.ts', 'src/core/lineSearch.ts', 'src/core/effectiveJacobian.ts',
    'src/core/implicitResidualJacobian.ts', 'src/core/leastSquaresMultiple.ts', 'src/core/constrainedLevenbergMarquardt.ts',
    'src/core/constrainedNormalEquations.ts', 'scripts/linkage-trajectory-benchmark.ts'];
  const environment = { ...collectPaperEnvironment(), gitDirty: execFileSync('git', ['status', '--porcelain']).length > 0,
    sourceHashes: Object.fromEntries(sourceFiles.map(file => [file, createHash('sha256').update(fs.readFileSync(path.join(ROOT, file))).digest('hex')])) };
  const protocol = { warmupCount: WARMUP_COUNT, repeatCount: REPEAT_COUNT, trainingPhases: 32, heldOutPhases: 257, heldOutOffset: HELD_OUT_OFFSET,
    adjustableLengths: ['crank', 'coupler', 'rocker'], fixedGroundAndMarker: true, objective: '0.5 * phase-mean squared distance',
    reachableTolerance: 1e-5, distortedMinimumCostReduction: 0.9, distortedGradientTolerance: 1e-8 };
  const format = (value: number | undefined) => value === undefined ? 'error' : value.toExponential(3);
  const lines = ['# Fixed-topology three-length trajectory fitting', '',
    'Four-bar connectivity, ground pivots, marker attachment, input phases, target and assembly branch are fixed.',
    'Only crank, coupler and rocker lengths vary. No target parameters are passed to the optimizer; targets and checks use circle geometry.',
    'BFGS uses numerical assembly and public IFT with finite-difference C_p; the FD baseline reassembles at every perturbation.',
    'Native LM uses analytical first partials and a dense cycle matrix. Its raw SSE history is normalized to 0.5*SSE by the example adapter.',
    'Three initial designs, two targets, three methods, one discarded warmup and three measured trials per condition.',
    'Timing includes fitting and snapshot capture. Verification, replay reconstruction and output are excluded.',
    'Accuracy: reachable held-out maximum error <=1e-5; distorted cost reduction >=90% and independent gradient norm <=1e-8.',
    'Both require closure and assembly-branch errors <=1e-9, finite metrics, and independently checked monotonically decreasing histories.',
    'A distorted-target pass means a tested local stationary fit with nonzero residual; it is not proof of a global optimum or impossibility.', '',
    '| Target | Initial | Method | Pass | Flag | Iterations | Median ms | IQR ms | Initial RMS | Held-out RMS | Max error | Closure |',
    '|---|---|---|---|---|---|---|---|---|---|---|---|'];
  for (const row of rows) {
    const verification = row.trials[0].verification;
    lines.push(`| ${row.targetCase} | ${row.initialIndex} | ${row.method} | ${row.successCount}/${REPEAT_COUNT} | ${verification?.solverConverged ?? 'error'} | ${verification?.iterations ?? 'error'} | ${format(row.timeMs.median)} | ${format(row.timeMs.interquartileRange)} | ${format(verification?.initialError)} | ${format(verification?.heldOutError)} | ${format(verification?.maximumHeldOutError)} | ${format(verification?.constraintNorm)} |`);
  }
  const successCount = rows.reduce((total, row) => total + row.successCount, 0);
  const trialCount = rows.length * REPEAT_COUNT;
  lines.push('', `Independent accuracy checks: ${successCount}/${trialCount} measured trials passed.`, '',
    'Units are dimensionless; phase correspondence is fixed. This synthetic example validates the continuous fitting stage, not original paper characters.',
    'There is no timing ranking pass condition. Small mechanism timings on one machine do not establish general speedups.', '',
    '## Environment', '', '```json', JSON.stringify(environment, null, 2), '```', '');
  fs.mkdirSync(OUTPUT_DIRECTORY, { recursive: true });
  for (const [extension, content] of [['json', JSON.stringify({ environment, protocol, successCount, trialCount, rows }, null, 2)], ['md', lines.join('\n')]]) {
    const output = path.join(OUTPUT_DIRECTORY, `linkage-trajectory-benchmark.${extension}`);
    fs.writeFileSync(output, content); console.log(`Wrote ${output}`);
  }
  console.log(`Independent accuracy checks: ${successCount}/${trialCount}`);
  if (successCount !== trialCount) process.exitCode = 1;
}

main();
