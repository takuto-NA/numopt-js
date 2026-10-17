/** Measure public-LM phase costs, timer perturbation and an exact-Jacobian control. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Matrix } from 'ml-matrix';
import * as originalApi from '../dist/index.js';
import { collectPaperEnvironment, runWarmedRepeats, summarizeNumeric } from './benchmark-harness';
import { assetDirectory, projectRoot, verifyKiboAssets, collectNumoptProvenance, sha256 } from './kibo-assets.mjs';
import { prepareProfileModules, PROBE_NAME } from './instrument-lm-profile.mjs';
import { PhaseRecorder } from './lm-phase-recorder.mjs';
import { collectCpuProfile } from './lm-cpu-profile.mjs';
import { smallProblems, mediumProblem, runProblem } from '../integration/kibo/scenarios.mjs';

const WARMUP_COUNT = 50;
const SAMPLE_COUNT = 100;
const ACCOUNTING_TOLERANCE_MILLISECONDS = 1e-6;
const SHARE_TOLERANCE = 1e-9;
const BACKENDS = ['ml-matrix', 'kibo'];
const MODES = ['unmodified', 'solver-only', 'phases'];
const tag = process.argv[2] ?? 'current';
if (!/^[a-z0-9-]+$/.test(tag)) throw new Error('Optional profile tag must use lowercase letters, numbers and hyphens');
const outputDirectory = path.join(projectRoot, 'benchmark-results/kibo');
const cpuDirectory = path.join(outputDirectory, `cpu-${tag}`);
fs.mkdirSync(cpuDirectory, { recursive: true });
const manifest = verifyKiboAssets();
const generated = prepareProfileModules();
const recorder = new PhaseRecorder();
globalThis[PROBE_NAME] = recorder;
const detailedApi = await import(pathToFileURL(path.join(generated.directory, 'dist/index.js')).href);
const { default: createModule } = await import(pathToFileURL(path.join(assetDirectory, 'kibo.mjs')).href);
const { LinearAlgebra } = await import(pathToFileURL(path.join(assetDirectory, 'facade.mjs')).href);
const { LinearAlgebra: DetailedLinearAlgebra } = await import(pathToFileURL(path.join(generated.directory, 'facade.mjs')).href);
const initializationStarted = performance.now();
const wasmModule = await createModule();
const initializationMilliseconds = performance.now() - initializationStarted;

function wrapSolver(factory) {
  return (...arguments_) => {
    const solver = factory(...arguments_);
    const solve = solver.solve.bind(solver);
    solver.solve = (...inputs) => recorder.measure('linearSolver', () => solve(...inputs));
    return solver;
  };
}

const solverOnlyApi = {
  ...originalApi,
  createMlMatrixLinearSolver: wrapSolver(originalApi.createMlMatrixLinearSolver),
  createKiboLinearSolver: wrapSolver(originalApi.createKiboLinearSolver),
  levenbergMarquardt: (...arguments_) => recorder.measure('lm', () => originalApi.levenbergMarquardt(...arguments_))
};
const phasesApi = {
  ...detailedApi,
  levenbergMarquardt(initial, residual, options) {
    return detailedApi.levenbergMarquardt(initial, parameters => recorder.measure(
      recorder.contains('jacobian') ? 'residualInJacobian' : 'residualOutsideJacobian', () => residual(parameters)), options);
  }
};
const variants = {
  unmodified: { api: originalApi, LinearAlgebra },
  'solver-only': { api: solverOnlyApi, LinearAlgebra },
  phases: { api: phasesApi, LinearAlgebra: DetailedLinearAlgebra }
};
const medium = mediumProblem();
const analytical = { ...medium, name: `${medium.name}-analytic`, jacobian: () =>
  Matrix.from1DArray(medium.fixedJacobian.rows, medium.fixedJacobian.columns, medium.fixedJacobian.values) };
const problems = [...smallProblems(originalApi), medium, analytical];

function checkAccounting(phases, result, mode) {
  if (mode === 'unmodified') { assert.deepEqual(phases, {}); return; }
  assert.equal(phases.lm.calls, 1);
  assert.equal(phases.linearSolver.calls, result.statistics.solveAttempts);
  const sum = Object.values(phases).reduce((total, phase) => total + phase.exclusiveMilliseconds, 0);
  assert.ok(Math.abs(sum - phases.lm.inclusiveMilliseconds) <= ACCOUNTING_TOLERANCE_MILLISECONDS);
  if (mode !== 'phases') return;
  assert.ok(phases.jacobian.calls > 0 && phases.normalEquations.calls > 0);
  if (result.backend === 'kibo') {
    assert.equal(phases.wasmExport.calls, result.statistics.wasmCalls);
    assert.equal(phases.copyIn.calls, result.statistics.wasmCalls);
    assert.equal(phases.copyOut.calls, result.statistics.solveSuccesses);
  }
}

function checkNumericalEquivalence(rows) {
  const fields = ['parameters', 'cost', 'gradientNorm', 'residualNorm', 'converged', 'iterations',
    'residualEvaluations', 'jacobianEvaluations'];
  const counterFields = ['solveAttempts', 'solveSuccesses', 'fallbackAttempts', 'wasmCalls',
    'inputPayloadBytes', 'outputPayloadBytes', 'failureReasons'];
  const deterministicResult = row => ({
    result: Object.fromEntries(fields.map(field => [field, row[field]])),
    statistics: Object.fromEntries(counterFields.map(field => [field, row.statistics[field]]))
  });
  for (const backend of BACKENDS) {
    const matching = rows.filter(row => row.backend === backend);
    const reference = matching.find(row => row.mode === 'unmodified');
    for (const row of matching)
      assert.deepEqual(deterministicResult(row), deterministicResult(reference), `${backend}: probe changed LM`);
  }
}

const configurations = MODES.flatMap(mode => BACKENDS.map(backend => ({ mode, backend })));
const trials = problems.flatMap(problem => runWarmedRepeats({
  warmupCount: WARMUP_COUNT, repeatCount: SAMPLE_COUNT,
  run(repeatIndex) {
    const offset = (repeatIndex % configurations.length + configurations.length) % configurations.length;
    const ordered = [...configurations.slice(offset), ...configurations.slice(0, offset)];
    const rows = ordered.map(({ mode, backend }, orderIndex) => {
      recorder.reset();
      const variant = variants[mode];
      const result = runProblem(variant.api, wasmModule, variant.LinearAlgebra, problem, backend);
      const phases = recorder.snapshot();
      checkAccounting(phases, result, mode);
      return { mode, repeatIndex, orderIndex, ...result, phases };
    });
    checkNumericalEquivalence(rows);
    return rows;
  }
}).flat());

function summarizePhases(rows) {
  const totals = {};
  for (const row of rows) for (const [name, phase] of Object.entries(row.phases)) {
    const total = totals[name] ?? { calls: 0, inclusiveMilliseconds: 0, exclusiveMilliseconds: 0 };
    for (const field of Object.keys(total)) total[field] += phase[field];
    totals[name] = total;
  }
  if (!totals.lm) return {};
  return Object.fromEntries(Object.entries(totals).map(([name, phase]) => [name, {
    callsPerRun: phase.calls / rows.length,
    meanInclusiveMilliseconds: phase.inclusiveMilliseconds / rows.length,
    meanExclusiveMilliseconds: phase.exclusiveMilliseconds / rows.length,
    exclusiveShare: phase.exclusiveMilliseconds / totals.lm.inclusiveMilliseconds
  }]));
}

function groupPhases(phases) {
  if (!phases.lm) return {};
  const share = name => phases[name]?.exclusiveShare ?? 0;
  const grouped = {
    jacobian: share('jacobian') + share('residualInJacobian'),
    normalEquations: share('normalEquations'), residualOutsideJacobian: share('residualOutsideJacobian'),
    damping: share('damping'),
    linearSolver: phases.linearSolver.meanInclusiveMilliseconds / phases.lm.meanInclusiveMilliseconds,
    controlAndStopping: share('lm') + share('step')
  };
  assert.ok(Math.abs(Object.values(grouped).reduce((sum, value) => sum + value, 0) - 1) <= SHARE_TOLERANCE);
  return grouped;
}

const summaries = problems.flatMap(problem => BACKENDS.flatMap(backend => MODES.map(mode => {
  const matching = trials.filter(trial => trial.problem === problem.name && trial.backend === backend && trial.mode === mode);
  const controls = trials.filter(trial => trial.problem === problem.name && trial.backend === backend && trial.mode === 'unmodified');
  const phases = summarizePhases(matching);
  const solverShare = phases.lm ? phases.linearSolver.meanInclusiveMilliseconds / phases.lm.meanInclusiveMilliseconds : undefined;
  return {
    problem: problem.name, backend, mode, samples: matching.length,
    elapsedMilliseconds: summarizeNumeric(matching.map(trial => trial.elapsedMilliseconds)),
    pairedTimeRatio: summarizeNumeric(matching.map(trial => trial.elapsedMilliseconds /
      controls.find(control => control.repeatIndex === trial.repeatIndex).elapsedMilliseconds)),
    phases, groupedShares: mode === 'phases' ? groupPhases(phases) : {},
    solverShare, idealWholeSolverEliminationSpeedup: solverShare === undefined ? undefined : 1 / (1 - solverShare),
    residualEvaluations: matching[0].residualEvaluations, jacobianEvaluations: matching[0].jacobianEvaluations,
    iterations: matching[0].iterations, solveAttempts: matching[0].statistics.solveAttempts,
    totalFallbackAttempts: matching.reduce((sum, trial) => sum + trial.statistics.fallbackAttempts, 0)
  };
})));

const cpuProfiles = [];
for (const problem of [medium, analytical]) for (const backend of BACKENDS) {
  const filename = path.join(cpuDirectory, `${problem.name}-${backend}.cpuprofile`);
  const profile = await collectCpuProfile(originalApi, wasmModule, LinearAlgebra, problem, backend, filename);
  cpuProfiles.push({ ...profile, filename: path.relative(projectRoot, filename).replaceAll('\\', '/'), sha256: sha256(filename) });
}
const report = {
  environment: collectPaperEnvironment(), numopt: collectNumoptProvenance(), manifest,
  command: `npm run benchmark:kibo:profile -- ${tag}`,
  warmupCount: WARMUP_COUNT, sampleCount: SAMPLE_COUNT, initializationMilliseconds,
  problems: problems.map(problem => ({
    name: problem.name, initial: problem.initial, expected: problem.expected,
    tolerance: problem.tolerance, maximumCost: problem.maximumCost, seed: problem.seed,
    conditionUpperBound: problem.conditionUpperBound, analyticalJacobian: Boolean(problem.jacobian),
    rows: problem.fixedJacobian?.rows, columns: problem.fixedJacobian?.columns
  })),
  transformations: generated.transformations,
  measurement: {
    unmodified: 'Original public LM, original facade and existing counters; no added phase probes.',
    solverOnly: 'Original LM with two public-boundary timers: whole LM and whole adapter solve.',
    phases: 'Generated ignored module copies; nested child time is subtracted once. Source/dist/assets remain unchanged.',
    perturbation: 'Paired ratios compare each instrumented run with unmodified in the same rotated block; no overhead subtraction.',
    wasmExport: 'Time around the real exported LLT call includes JS/WASM dispatch and C ABI validation; not pure kernel time.',
    copyIn: 'Facade alias-protecting JS copies, heap-view creation and writes; includes checks in the same method.',
    copyOut: 'Facade output view creation and JS-owned slice.',
    solverOnlyShare: 'Use this low-probe variant to estimate an ideal upper bound from eliminating only the current solver boundary.',
    cpu: 'Independent sampling of original modules; LM shares exclude outside-LM and GC samples, which are reported separately.',
    analytical: 'Same 128x32 linear residual, seed, start, tolerances and truth; only the Jacobian source changes to its exact matrix.'
  }, summaries, cpuProfiles, trials
};
const filename = path.join(outputDirectory, `profile-${tag}.json`);
fs.writeFileSync(filename, `${JSON.stringify(report, null, 2)}\n`);
console.table(summaries.filter(row => row.problem.startsWith('dense')).map(row => ({
  problem: row.problem, backend: row.backend, mode: row.mode,
  medianMs: row.elapsedMilliseconds.median, probeRatio: row.pairedTimeRatio.median,
  residualCalls: row.residualEvaluations, solveCalls: row.solveAttempts
})));
console.table(summaries.filter(row => row.problem.startsWith('dense') && row.mode === 'phases').map(row => ({
  problem: row.problem, backend: row.backend, ...Object.fromEntries(Object.entries(row.groupedShares).map(([name, share]) =>
    [name, `${(share * 100).toFixed(1)}%`]))
})));
console.log(`Saved ${filename}; ${trials.length} numerical/equivalence/accounting checks passed.`);
delete globalThis[PROBE_NAME];
