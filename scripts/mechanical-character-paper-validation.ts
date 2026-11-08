/** Validate specific claims of Coros et al. (2013) with equation checks, geared linkages and gradient-only timings. */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { vectorNorm } from '../src/index';
import { collectPaperEnvironment, runWarmedRepeats, summarizeNumeric } from './benchmark-harness';
import { createPaperCases, createPaperFiveBar, createPaperFourBar, PAPER_PHASE_COUNTS, type PaperMechanism } from '../examples/paper-mechanism/cases';
import { checkPaperEquations, checkQuadrature, checkStateAccuracy, initialPaperDesigns, runPaperFit } from '../examples/paper-mechanism/checks';
import { createPaperEvaluator, paperGradientAt, type PaperGradient } from '../examples/paper-mechanism/evaluate';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT_DIRECTORY = path.join(ROOT, 'benchmark-results');
const WARMUP_COUNT = 2;
const REPEAT_COUNT = 7;
const DEFAULT_PHASE_COUNT = 32;
const REFINEMENT_PHASE_COUNTS = [8, 16, 32, 64];
const ACCURACY_TOLERANCE = 1e-8;
const PAPER_URL = 'https://publications.ri.cmu.edu/storage/publications/pub_files/2013/0/CDMC_final.pdf';

function fitRow(model: PaperMechanism, method: PaperGradient, initial: Float64Array, initialIndex: number, distorted: boolean) {
  try {
    return { name: model.name, sampleCount: model.phases.length, method, initialIndex, distorted,
      initialParameters: Array.from(initial), ...runPaperFit(model, method, initial, distorted), error: null };
  } catch (error) {
    return { name: model.name, sampleCount: model.phases.length, method, initialIndex, distorted,
      initialParameters: Array.from(initial), verification: null, history: [], counters: null, finalParameters: [],
      error: error instanceof Error ? error.message : String(error) };
  }
}

function fitModels() {
  const primary = [false, true].flatMap(distorted => createPaperCases(DEFAULT_PHASE_COUNT, distorted).flatMap(model =>
    initialPaperDesigns(model).map((initial, index) => fitRow(model, 'paper-ift', initial, index, distorted))));
  // Full-FD optimizer checks are separate from gradient agreement and use the same initial point and tolerance.
  const finiteDifference = createPaperCases().map(model => fitRow(model, 'full-fd', model.initialParameters, 0, false));
  return [...primary, ...finiteDifference];
}

function fitRefinement() {
  return REFINEMENT_PHASE_COUNTS.flatMap(sampleCount =>
    [createPaperFourBar(sampleCount, true), createPaperFiveBar(2, 8, sampleCount, true)].map(model =>
      fitRow(model, 'paper-ift', model.initialParameters, 0, true)));
}

function selectBestInitializations(fits: ReturnType<typeof fitModels>) {
  return [false, true].flatMap(distorted => createPaperCases().map(model => {
    const candidates = fits.filter(row => row.name === model.name && row.distorted === distorted && row.method === 'paper-ift');
    // Select by the independently reconstructed objective, without selecting on the pass flag.
    const best = candidates.reduce((selected, row) => (row.verification?.finalCost ?? Number.POSITIVE_INFINITY)
      < (selected.verification?.finalCost ?? Number.POSITIVE_INFINITY) ? row : selected);
    return { name: model.name, distorted, initialIndex: best.initialIndex,
      passed: best.verification?.passed ?? false, heldOutError: best.verification?.heldOutError,
      projectedGradientNorm: best.verification?.boundaryStationarity.projectedGradientNorm };
  }));
}

function timeGradients() {
  return PAPER_PHASE_COUNTS.flatMap(sampleCount => createPaperCases(sampleCount).flatMap(model =>
    (['included', 'cached'] as const).map(statePreparation => {
    const reference = paperGradientAt(model, model.initialParameters, model.oracle(model.initialParameters, model.phases), 'analytic-ift');
    const methods = (['paper-ift', 'full-fd'] as const).map(method => {
      const trials = runWarmedRepeats({ warmupCount: WARMUP_COUNT, repeatCount: REPEAT_COUNT, run: () => {
        // No assembly or gradient cache survives a trial. Fixed target construction is untimed for both methods.
        const evaluator = createPaperEvaluator(model, method);
        if (statePreparation === 'cached') evaluator.cost(model.initialParameters);
        const before = { ...evaluator.counters };
        const start = performance.now();
        const gradient = evaluator.gradient(model.initialParameters);
        const elapsedMs = performance.now() - start;
        const gradientError = vectorNorm(gradient.map((value, index) => value - reference[index]));
        const assemblies = evaluator.counters.assemblies - before.assemblies;
        const expectedAssemblies = sampleCount * (method === 'paper-ift'
          ? statePreparation === 'cached' ? 0 : 1 : 2 * model.initialParameters.length);
        return { elapsedMs, gradientError, relativeGradientError: gradientError / vectorNorm(reference),
          assemblies, constraints: evaluator.counters.constraints - before.constraints, initialAssemblies: before.assemblies,
          passed: gradientError <= ACCURACY_TOLERANCE && assemblies === expectedAssemblies };
      } });
      return { method, timeMs: summarizeNumeric(trials.map(trial => trial.elapsedMs)), trials };
    });
    return { name: model.name, sampleCount, statePreparation, parameterCount: model.initialParameters.length, methods,
      speedup: methods[1].timeMs.median / methods[0].timeMs.median };
  })));
}

function collectEnvironment() {
  const trackedSources = execFileSync('git', ['ls-files', 'src'], { cwd: ROOT }).toString().trim().split(/\r?\n/);
  const exampleFiles = fs.readdirSync(path.join(ROOT, 'examples/paper-mechanism')).filter(file => file.endsWith('.ts'))
    .map(file => `examples/paper-mechanism/${file}`);
  const sourceFiles = [...trackedSources, ...exampleFiles, 'examples/mechanical-character/four-bar.ts', 'scripts/mechanical-character-paper-validation.ts'];
  const hash = (file: string) => createHash('sha256').update(fs.readFileSync(path.join(ROOT, file))).digest('hex');
  const paperPath = 'benchmark-results/papers/CDMC_final.pdf';
  return { ...collectPaperEnvironment(), gitDirty: execFileSync('git', ['status', '--porcelain'], { cwd: ROOT }).length > 0,
    paperSha256: fs.existsSync(path.join(ROOT, paperPath)) ? hash(paperPath) : null,
    sourceHashes: Object.fromEntries(sourceFiles.map(file => [file, hash(file)])) };
}

function createReport() {
  const models = createPaperCases();
  const equations = models.map(checkPaperEquations);
  const quadrature = models.flatMap(checkQuadrature);
  const stateAccuracy = models.flatMap(checkStateAccuracy);
  const fits = fitModels();
  const bestInitializations = selectBestInitializations(fits);
  const refinement = fitRefinement();
  console.log(`Equation checks ${equations.filter(row => row.passed).length}/${equations.length}; fits ${fits.filter(row => row.verification?.passed).length}/${fits.length}`);
  const gradients = timeGradients();
  const checks = { equations: equations.every(row => row.passed), singleStartFits: fits.every(row => row.verification?.passed),
    bestOfThree: bestInitializations.every(row => row.passed),
    quadrature: quadrature.filter(row => row.sampleCount === PAPER_PHASE_COUNTS[PAPER_PHASE_COUNTS.length - 1])
      .every(row => row.costError <= ACCURACY_TOLERANCE && row.gradientError <= ACCURACY_TOLERANCE),
    strictStateAccuracy: stateAccuracy.filter(row => row.tolerance === 1e-13).every(row =>
      row.maximumClosure <= ACCURACY_TOLERANCE && row.gradientError <= ACCURACY_TOLERANCE),
    refinement: refinement.filter(row => row.sampleCount === REFINEMENT_PHASE_COUNTS[REFINEMENT_PHASE_COUNTS.length - 1])
      .every(row => row.verification?.passed), gradientTrials: gradients.every(row => row.methods.every(method => method.trials.every(trial => trial.passed))) };
  return { environment: collectEnvironment(), protocol: { source: PAPER_URL, sections: ['3.2', '4.2', '7.1'],
    equations: [1, 2, 3, 4], gearRatios: [-2, -1, 1, 2], warmupCount: WARMUP_COUNT, repeatCount: REPEAT_COUNT,
    gradientStatePreparation: ['included', 'cached'], timingExcludesTargetConstructionAndVerification: true,
    paperReportedGradientSpeedup: [20, 30], learningAndSqpImplemented: false },
    checks, equations, quadrature, stateAccuracy, fits, bestInitializations, refinement, gradients };
}

type PaperReport = ReturnType<typeof createReport>;
const format = (value: number | undefined) => value === undefined ? 'error' : value.toExponential(3);

function renderReport(report: PaperReport): string {
  const lines = ['# Coros et al. (2013): paper-guided continuous mechanism validation', '',
    `Source: [Computational Design of Mechanical Characters](${PAPER_URL}), pp. 4-5 and 9.`, '',
    'Synthetic fixtures: one six-parameter four-bar and geared five-bars with six/eight parameters, ratios -2/-1/1/2.',
    'These follow the mechanism class and equations, not the original dimensions, character data or complete pipeline.',
    'Eq. (1) is solved numerically with public Gauss-Newton, rather than claiming an identical Newton-Raphson implementation.',
    'Eq. (2) uses normalized input phase t=theta/(2*pi); uniform samples approximate its integral on [0,1].',
    'Eq. (3)-(4) use analytical state/marker partials, finite-difference C_p with fixed states, and public BFGS.',
    'Timing is one gradient, not a complete optimizer. Both fresh and already-assembled current-state scenarios are measured.',
    'Cached-state setup computes the current cost before timing for both methods. Only additional assemblies count in its timed work.',
    'The paper does not specify this timing boundary precisely enough to equate either fixture measurement with its original factor.',
    'State constraints are tested independently; the raw GN stopping flag or tolerance is not a feasibility certificate.',
    'Distorted fits use gradient norm <=1e-8 in the stated tanh design coordinates, not a global optimality certificate.',
    'Saturated design coordinates are explicitly counted; small transformed gradients there do not certify stationarity in physical lengths.', '',
    'Every fit additionally requires an independent finite-difference projected gradient <=1e-6 in the closed normalized physical design box.',
    `Checks: ${JSON.stringify(report.checks)}`, '',
    '## Eqs. (3)-(4), negative controls and redundant constraints', '',
    '| Model | Pass | Public IFT abs error | Full FD abs error | IFT Taylor order | Frozen-state order | Omit-direct order | Redundant error | Min state singular value |',
    '|---|---|---|---|---|---|---|---|---|'];
  for (const row of report.equations) lines.push(`| ${row.name} | ${row.passed} | ${format(row.gradients[0].error)} | ${format(row.gradients[1].error)} | ${row.gradients[0].taylor.orders[0].toFixed(3)} | ${row.gradients[2].taylor.orders[0].toFixed(3)} | ${row.gradients[3].taylor.orders[0].toFixed(3)} | ${format(row.redundantError)} | ${format(row.minimumStateSingularValue)} |`);
  lines.push('', 'Correct derivatives leave a quadratic Taylor remainder; deliberately dropping either chain-rule contribution leaves first-order error.',
    'Changing the marker-only parameter leaves states unchanged but moves the marker. A phase-shifted target gives nonzero Eq. (2) error even for the same curve shape.', '',
    '## End-to-end BFGS fitting (no timing ranking)', '',
    'Reachable: held-out maximum error <=1e-5. Distorted: objective decreases and independently recomputed design-gradient norm <=1e-8.',
    'Both: closure and branch errors <=1e-9, monotonic independently reconstructed histories. Held-out grid: 257 samples with offset 0.37.', '',
    '| Model | Distorted | Method | Initial | Pass | Solver flag | Iterations | Held-out RMS | Design gradient | Box projected gradient | Saturated coordinates |',
    '|---|---|---|---|---|---|---|---|---|---|---|');
  for (const row of report.fits) lines.push(`| ${row.name} | ${row.distorted} | ${row.method} | ${row.initialIndex} | ${row.verification?.passed ?? false} | ${row.verification?.solverConverged ?? false} | ${row.verification?.iterations ?? 'error'} | ${format(row.verification?.heldOutError)} | ${format(row.verification?.gradientNorm)} | ${format(row.verification?.boundaryStationarity.projectedGradientNorm)} | ${row.verification?.saturatedParameters.length ?? 'error'} |`);
  lines.push('', '## Best objective among the three predefined initial designs', '',
    'This tests initialization sensitivity; it does not reproduce the paper database or its retrieval metric.', '',
    '| Model | Distorted | Selected initial | Pass | Held-out RMS | Box projected gradient |', '|---|---|---|---|---|---|');
  for (const row of report.bestInitializations) lines.push(`| ${row.name} | ${row.distorted} | ${row.initialIndex} | ${row.passed} | ${format(row.heldOutError)} | ${format(row.projectedGradientNorm)} |`);
  lines.push('', '## Section 7.1: gradient computation, not optimizer wall time', '',
    `Two discarded warmups and ${REPEAT_COUNT} trials per method/condition; median and IQR. Failed accuracy checks remain failures.`,
    'Included state: IFT performs N assemblies and central full FD 2*p*N. Cached state: IFT performs 0 additional assemblies and FD 2*p*N.',
    'Every trial uses a new evaluator; cached-state preparation does not survive between trials.',
    'The paper reports 20-30x on its mechanisms; matching that factor is not a pass condition for these synthetic fixtures.', '',
    '| Model | Phases | Parameters | Current state | IFT median ms | IFT IQR | FD median ms | FD IQR | FD/IFT | IFT additional assemblies | FD additional assemblies |',
    '|---|---|---|---|---|---|---|---|---|---|---|---|');
  for (const row of report.gradients) lines.push(`| ${row.name} | ${row.sampleCount} | ${row.parameterCount} | ${row.statePreparation} | ${format(row.methods[0].timeMs.median)} | ${format(row.methods[0].timeMs.interquartileRange)} | ${format(row.methods[1].timeMs.median)} | ${format(row.methods[1].timeMs.interquartileRange)} | ${row.speedup.toFixed(2)} | ${row.methods[0].trials[0].assemblies} | ${row.methods[1].trials[0].assemblies} |`);
  lines.push('', '## Eq. (2): phase quadrature against 4096 independent geometric samples', '',
    '| Model | Phases | Cost error | Gradient error |', '|---|---|---|---|');
  for (const row of report.quadrature) lines.push(`| ${row.name} | ${row.sampleCount} | ${format(row.costError)} | ${format(row.gradientError)} |`);
  lines.push('', '## State accuracy diagnostic (coarse states are intentionally retained)', '',
    '| Model | Requested GN tolerance | Maximum closure | Design-gradient error |', '|---|---|---|---|');
  for (const row of report.stateAccuracy) lines.push(`| ${row.name} | ${format(row.tolerance)} | ${format(row.maximumClosure)} | ${format(row.gradientError)} |`);
  lines.push('', '## Sample-refinement fits on distorted targets', '',
    '| Model | Phases | Training RMS | Held-out RMS | Gradient norm |', '|---|---|---|---|---|');
  for (const row of report.refinement) lines.push(`| ${row.name} | ${row.sampleCount} | ${format(row.verification?.trainingError)} | ${format(row.verification?.heldOutError)} | ${format(row.verification?.gradientNorm)} |`);
  lines.push('', '## Scope', '',
    'No original Fig. 4 dimensions, Fig. 5 character attachment, database/metric training, non-circular gears, gear geometry, 3D mechanisms or fabrication are reproduced.',
    'No claims about global convergence, arbitrary target reachability, physical accuracy or the original 20-30x measurement follow from passing these checks.', '',
    '## Environment', '', '```json', JSON.stringify(report.environment, null, 2), '```', '');
  return lines.join('\n');
}

function main(): void {
  const report = createReport();
  fs.mkdirSync(OUTPUT_DIRECTORY, { recursive: true });
  const output = path.join(OUTPUT_DIRECTORY, 'mechanical-character-paper-validation');
  fs.writeFileSync(`${output}.json`, JSON.stringify(report, null, 2));
  fs.writeFileSync(`${output}.md`, renderReport(report));
  console.log(`Wrote ${output}.{json,md}`);
  console.log(report.checks);
  if (!Object.values(report.checks).every(Boolean)) process.exitCode = 1;
}

main();
