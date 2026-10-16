/** Collect manual, copy-inclusive LM comparisons without making timing a CI gate. */
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import * as api from '../dist/index.js';
import { collectPaperEnvironment, runWarmedRepeats, summarizeNumeric } from './benchmark-harness';
import { assetDirectory, verifyKiboAssets, collectNumoptProvenance } from './kibo-assets.mjs';
import { smallProblems, mediumProblem, runProblem } from '../integration/kibo/scenarios.mjs';

const WARMUP_COUNT = 5;
const SAMPLE_COUNT = 30;
const manifest = verifyKiboAssets();
const { default: createModule } = await import(pathToFileURL(path.join(assetDirectory, 'kibo.mjs')).href);
const { LinearAlgebra } = await import(pathToFileURL(path.join(assetDirectory, 'facade.mjs')).href);
const initializationStarted = performance.now();
const wasmModule = await createModule();
const initializationMilliseconds = performance.now() - initializationStarted;
const problems = [...smallProblems(api), mediumProblem()];
const trials = problems.flatMap(problem => runWarmedRepeats({
  warmupCount: WARMUP_COUNT, repeatCount: SAMPLE_COUNT,
  run(repeatIndex) {
    const order = repeatIndex % 2 === 0 ? ['ml-matrix', 'kibo'] : ['kibo', 'ml-matrix'];
    return order.map((backend, orderIndex) => ({
      repeatIndex, orderIndex, ...runProblem(api, wasmModule, LinearAlgebra, problem, backend)
    }));
  }
}).flat());
const summaries = problems.flatMap(problem => ['ml-matrix', 'kibo'].map(backend => {
  const matching = trials.filter(trial => trial.problem === problem.name && trial.backend === backend);
  return {
    problem: problem.name, backend, samples: matching.length,
    elapsedMilliseconds: summarizeNumeric(matching.map(trial => trial.elapsedMilliseconds)),
    preparationMilliseconds: summarizeNumeric(matching.map(trial => trial.preparationMilliseconds)),
    conversionMilliseconds: summarizeNumeric(matching.map(trial => trial.statistics.conversionMilliseconds)),
    boundaryMilliseconds: summarizeNumeric(matching.map(trial => trial.statistics.boundaryMilliseconds)),
    iterations: matching.map(trial => trial.iterations),
    maximumParameterError: Math.max(...matching.map(trial => trial.parameterError)),
    totalFallbackAttempts: matching.reduce((sum, trial) => sum + trial.statistics.fallbackAttempts, 0)
  };
}));
const report = {
  environment: collectPaperEnvironment(), manifest, numopt: collectNumoptProvenance(),
  initializationMilliseconds, warmupCount: WARMUP_COUNT, sampleCount: SAMPLE_COUNT,
  notes: {
    timing: 'Same public LM; elapsed includes matrix conversion, facade copy-in/out, solve, and callbacks.',
    boundary: 'Backend boundary includes facade copies and LU fallback; it is not isolated C++ kernel time.',
    payload: 'Payload bytes count logical input/output crossing the facade, not all JS allocations or internal copies.',
    mediumFixture: 'seed 0x6b69626f; 128x32 embedded 2I plus perturbation with Frobenius norm <= 0.5; truth is all ones.'
  }, summaries, trials
};
fs.mkdirSync('benchmark-results/kibo', { recursive: true });
fs.writeFileSync('benchmark-results/kibo/comparison.json', `${JSON.stringify(report, null, 2)}\n`);
console.table(summaries.map(summary => ({
  problem: summary.problem, backend: summary.backend,
  medianMs: summary.elapsedMilliseconds.median, iqrMs: summary.elapsedMilliseconds.interquartileRange,
  fallbackAttempts: summary.totalFallbackAttempts
})));
