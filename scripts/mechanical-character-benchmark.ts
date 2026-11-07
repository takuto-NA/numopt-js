/** Benchmarks section 4.2 on synthetic four-bars; reports accuracy failures without asserting solver rankings. */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { collectPaperEnvironment, runWarmedRepeats, summarizeNumeric } from './benchmark-harness';
import {
  createFourBarProblem, createReducedEvaluator, createWorkCounters, implicitGradient, oracleStates, traceCurve,
  type GradientMode
} from '../examples/mechanical-character/four-bar';
import { METHODS, runMechanismSolver, verifyMechanismOutcome } from '../examples/mechanical-character/solvers';

const WARMUP_COUNT = 1;
const REPEAT_COUNT = 3;
const GRADIENT_REPEAT_COUNT = 7;
const GRADIENT_SAMPLE_COUNTS = [8, 32, 128];
const GRADIENT_MODES: GradientMode[] = ['analytic', 'paper', 'public-ift', 'redundant-ift', 'finite-difference'];
const OUTPUT_DIRECTORY = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../benchmark-results');
const GRADIENT_ERROR_TOLERANCE = 1e-7;

function compareGradients() {
  return GRADIENT_SAMPLE_COUNTS.flatMap(sampleCount => {
    const problem = createFourBarProblem(sampleCount);
    const reference = implicitGradient(problem, problem.initialParameters,
      oracleStates(problem.initialParameters, problem.phases), 'analytic');
    return GRADIENT_MODES.map(mode => {
      const trials = runWarmedRepeats({ warmupCount: WARMUP_COUNT, repeatCount: GRADIENT_REPEAT_COUNT, run: () => {
        const counters = createWorkCounters();
        const evaluator = createReducedEvaluator(problem, mode, counters);
        // Include assembly of the current state. No state or gradient cache survives between trials.
        const start = performance.now();
        const gradient = evaluator.gradient(problem.initialParameters);
        const elapsedMs = performance.now() - start;
        const error = Math.hypot(...gradient.map((value, index) => value - reference[index]));
        return { elapsedMs, error, ...counters };
      } });
      return { sampleCount, mode, timeMs: summarizeNumeric(trials.map(trial => trial.elapsedMs)),
        maximumGradientError: Math.max(...trials.map(trial => trial.error)),
        phaseAssemblies: trials[0].assemblies, phaseConstraintEvaluations: trials[0].constraintEvaluations, trials };
    });
  });
}

function compareSolvers() {
  return [false, true].flatMap(distorted => METHODS.map(method => {
    const problem = createFourBarProblem();
    const activeProblem = distorted ? createFourBarProblem(problem.phases.length, true) : problem;
    const trials = runWarmedRepeats({ warmupCount: WARMUP_COUNT, repeatCount: REPEAT_COUNT, run: () => {
      const start = performance.now();
      try {
        const outcome = runMechanismSolver(activeProblem, method);
        const elapsedMs = performance.now() - start;
        return { elapsedMs, verification: verifyMechanismOutcome(activeProblem, outcome, distorted),
          counters: outcome.counters, history: outcome.history, parameters: Array.from(outcome.result.finalParameters),
          curve: traceCurve(outcome.result.finalParameters, outcome.states, activeProblem.phases) };
      } catch (error) {
        return { elapsedMs: performance.now() - start, error: error instanceof Error ? error.message : String(error) };
      }
    } });
    const completed = trials.filter(trial => trial.verification !== undefined);
    const strictSuccessCount = completed.filter(trial => trial.verification!.passed).length;
    console.log(`${distorted ? 'distorted' : 'reachable'} / ${method}: ${strictSuccessCount}/${REPEAT_COUNT} strict accuracy checks`);
    return { problem: distorted ? 'distorted' : 'reachable', method, strictSuccessCount, trialCount: REPEAT_COUNT,
      timeMs: summarizeNumeric(trials.map(trial => trial.elapsedMs)), trials,
      target: activeProblem.target,
      initialCurve: traceCurve(activeProblem.initialParameters, oracleStates(activeProblem.initialParameters, activeProblem.phases), activeProblem.phases) };
  }));
}

function main(): void {
  const sourceRoot = path.resolve(OUTPUT_DIRECTORY, '..');
  const sourceFiles = ['src/core/bfgsUpdate.ts', 'src/core/curvature.ts', 'src/core/effectiveJacobian.ts',
    'src/core/leastSquaresMultiple.ts', 'examples/mechanical-character/four-bar.ts', 'examples/mechanical-character/solvers.ts'];
  const sourceHashes = Object.fromEntries(sourceFiles.map(file =>
    [file, createHash('sha256').update(fs.readFileSync(path.join(sourceRoot, file))).digest('hex')]));
  const environment = { ...collectPaperEnvironment(), gitDirty: execFileSync('git', ['status', '--porcelain']).length > 0, sourceHashes };
  const gradients = compareGradients();
  const rows = compareSolvers();
  const format = (value: number | undefined): string => value === undefined ? 'n/a' : value.toExponential(3);
  const lines = [
    '# Mechanical character: synthetic section 4.2 validation', '',
    'Source: [Coros et al. (2013)](https://publications.ri.cmu.edu/storage/publications/pub_files/2013/0/CDMC_final.pdf).', '',
    'This is a new planar four-bar fixture, not reproduction of the original character data or full design system.',
    'Every trajectory uses 32 uniform phases; accuracy is checked independently with circle geometry at 257 shifted phases.',
    `Timing: ${WARMUP_COUNT} discarded warmup, ${REPEAT_COUNT} timed optimizer trials; gradient timing uses ${GRADIENT_REPEAT_COUNT} trials.`,
    'Raw trials, environment, parameters, histories, and curves are retained in the JSON. Verification and output are untimed.',
    'Paper/public/redundant IFT use analytical state/marker partials and finite-difference C_p at fixed states. Redundant IFT doubles the constraint rows.',
    'Other constrained methods use analytical first partials.',
    'All methods use 600 design iterations and 1e-9 gradient tolerance. GN/LM use a different curvature model from paper BFGS.', '',
    'Strict reachable accuracy: training and held-out positional RMS <= 1e-5, constraint norm <= 1e-9, correct assembly branch.',
    'Strict distorted accuracy: >=95% cost reduction, independently computed gradient norm <= 1e-8, same feasibility and branch checks.',
    'Solver flags are shown separately. Failed checks and exceptions remain in the report; no speed ranking is a pass condition.', '',
    '## Optimizer results', '',
    '| Case | Method | Strict pass | Flag | Iterations | Median ms | IQR ms | Held-out RMS | Constraint norm | Gradient norm |',
    '|---|---|---|---|---|---|---|---|---|---|'
  ];
  for (const row of rows) {
    const verification = row.trials[0].verification;
    lines.push(`| ${row.problem} | ${row.method} | ${row.strictSuccessCount}/${row.trialCount} | ${verification?.solverConverged ?? 'error'} | ${verification?.iterations ?? 'n/a'} | ${format(row.timeMs.median)} | ${format(row.timeMs.interquartileRange)} | ${format(verification?.heldOutError)} | ${format(verification?.constraintNorm)} | ${format(verification?.gradientNorm)} |`);
  }
  lines.push('', '## Gradient results', '',
    'Each gradient call includes its own current-state assembly. Full FD reassembles at each of 2p perturbations.',
    'Phase constraint counts include assembly iterations and finite differences of C_p. They count two-scalar phase blocks.', '',
    '| Phases | Mode | Median ms | IQR ms | Max absolute gradient error | Phase assemblies | Phase constraint evaluations |',
    '|---|---|---|---|---|---|---|');
  for (const row of gradients) lines.push(`| ${row.sampleCount} | ${row.mode} | ${format(row.timeMs.median)} | ${format(row.timeMs.interquartileRange)} | ${format(row.maximumGradientError)} | ${row.phaseAssemblies} | ${row.phaseConstraintEvaluations} |`);
  lines.push('', 'The paper reports 20-30x gradient speedups for its mechanisms. No matching speedup is presumed for this fixture.',
    'Native constrained methods currently assemble dense matrices across phases; the paper-style reference uses independent 2x2 blocks.',
    'The constraint evaluation counts include different projection strategies; forward-assembly counts are only comparable in the gradient table.',
    '', '## Environment', '', '```json', JSON.stringify(environment, null, 2), '```', '');
  fs.mkdirSync(OUTPUT_DIRECTORY, { recursive: true });
  const jsonPath = path.join(OUTPUT_DIRECTORY, 'mechanical-character.json');
  const markdownPath = path.join(OUTPUT_DIRECTORY, 'mechanical-character.md');
  fs.writeFileSync(jsonPath, JSON.stringify({ environment, protocol: { warmupCount: WARMUP_COUNT, repeatCount: REPEAT_COUNT,
    gradientRepeatCount: GRADIENT_REPEAT_COUNT }, gradients, rows }, null, 2));
  fs.writeFileSync(markdownPath, lines.join('\n'));
  console.log(`Wrote ${jsonPath}\nWrote ${markdownPath}`);
  if (gradients.some(row => row.maximumGradientError > GRADIENT_ERROR_TOLERANCE)
    || rows.some(row => row.strictSuccessCount !== row.trialCount)) process.exitCode = 1;
}

main();
