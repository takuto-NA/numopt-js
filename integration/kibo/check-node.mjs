/** Execute the public LM against the real, prepared kibo WASM module. */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import { Matrix } from 'ml-matrix';
import { createKiboLinearSolver } from '../../dist/index.js';
import createModule from '../../.cache/kibo/kibo.mjs';
import { LinearAlgebra } from '../../.cache/kibo/facade.mjs';
import * as publicApi from '../../dist/index.js';
import { runSmallComparisons, runResourceChecks } from './scenarios.mjs';
import { runConfigurationChecks, runFallbackChecks } from './failures.mjs';
import { collectNumoptProvenance, verifyKiboAssets } from '../../scripts/kibo-assets.mjs';

const wasmModule = await createModule();
const manifest = verifyKiboAssets();
const report = { runtime: process.version, manifest, numopt: collectNumoptProvenance() };
test.after(() => {
  fs.mkdirSync('benchmark-results/kibo', { recursive: true });
  fs.writeFileSync('benchmark-results/kibo/node.json', `${JSON.stringify(report, null, 2)}\n`);
});

test('both backends pass the same line, exponential and analytic-Jacobian examples', () => {
  const cases = runSmallComparisons(publicApi, wasmModule, LinearAlgebra);
  report.cases = cases;
  assert.equal(cases.length, 6);
});

test('prepared workspace, caller inputs and copied solutions survive growth and disposal', () => {
  report.resources = runResourceChecks(publicApi, wasmModule, LinearAlgebra);
  assert.equal(report.resources.disposedRejected, true);
});

test('public LM rejects fatal boundary errors without fallback or retries', () => {
  report.configurationFailures = runConfigurationChecks(publicApi, wasmModule, LinearAlgebra);
  assert.ok(report.configurationFailures.every(result => result.rejected && result.disposed));
});

test('public LM continues through LU fallback and retries overflowing LU before termination', () => {
  report.numericFallbacks = runFallbackChecks(publicApi, wasmModule, LinearAlgebra);
  assert.equal(report.numericFallbacks.length, 3);
});

test('reports the actual WASM payload and numerical failure before LU fallback', () => {
  const solver = createKiboLinearSolver(new LinearAlgebra(wasmModule, 2, 2));
  try {
    const solution = solver.solve(new Matrix([[0, 1], [1, 0]]), new Matrix([[2], [3]]));
    assert.deepEqual(Array.from(solution), [3, 2]);
    assert.equal(solver.statistics.fallbackAttempts, 1);
    assert.equal(solver.statistics.failureReasons['kibo-status-8'], 1);
    assert.equal(solver.statistics.inputPayloadBytes, 48);
    assert.ok(Number.isFinite(solver.statistics.boundaryMilliseconds));
  } finally { solver.dispose(); }
});

test('an overflowing LU fallback is not counted or returned as a successful solution', () => {
  const solver = createKiboLinearSolver(new LinearAlgebra(wasmModule, 1, 1));
  try {
    assert.throws(() => solver.solve(new Matrix([[1e-308]]), new Matrix([[1e308]])), /finite|overflow|singular/i);
    assert.equal(solver.statistics.solveSuccesses, 0);
  } finally { solver.dispose(); }
});
