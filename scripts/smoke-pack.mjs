/**
 * Verify packed package exports, files, and browser solver execution in a temporary consumer project.
 */

import { execSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

function run(cmd, cwd) {
  return execSync(cmd, {
    cwd,
    stdio: 'pipe',
    encoding: 'utf8',
    shell: true,
  });
}

function writeFile(p, content) {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, content, 'utf8');
}

function assert(cond, message) {
  if (!cond) {
    throw new Error(message);
  }
}

const repoRoot = process.cwd();

// 1) Create npm tarball (JSON output is easiest to parse)
const packJson = run('npm pack --json', repoRoot);
const parsedPack = JSON.parse(packJson);
// npm 12 keys this result by package name; earlier npm versions return an array.
const pack = Array.isArray(parsedPack) ? parsedPack : Object.values(parsedPack);
assert(Array.isArray(pack) && pack.length >= 1, 'npm pack did not return expected JSON output');
const tgzName = pack[0].filename;
assert(typeof tgzName === 'string' && tgzName.endsWith('.tgz'), 'npm pack did not return a .tgz filename');

const tgzPath = path.resolve(repoRoot, tgzName);
assert(fs.existsSync(tgzPath), `Packed tarball not found at ${tgzPath}`);

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'numopt-js-pack-smoke-'));

try {
  // 2) Install tarball into a temp project
  run('npm init -y', tmpDir);
  run(`npm install "${tgzPath}"`, tmpDir);

  // 3) Validate CommonJS require works
  writeFile(
    path.join(tmpDir, 'cjs.cjs'),
    `
const m = require('numopt-js');
if (typeof m.gradientDescent !== 'function') {
  throw new Error('CJS: gradientDescent export missing');
}
if (typeof m.adjointNewton !== 'function') {
  throw new Error('CJS: adjointNewton export missing');
}
if (typeof m.adjointNewtonExact !== 'function') {
  throw new Error('CJS: adjointNewtonExact export missing');
}
if (typeof m.adjointLbfgs !== 'function') throw new Error('CJS: adjointLbfgs export missing');
if (typeof m.computeEffectiveJacobian !== 'function') throw new Error('CJS: computeEffectiveJacobian export missing');
console.log('CJS: ok');
`.trimStart()
  );
  run('node cjs.cjs', tmpDir);

  // 4) Validate ESM import works (mjs forces ESM even in a CJS package)
  writeFile(
    path.join(tmpDir, 'esm.mjs'),
    `
import { gradientDescent, adjointNewton, adjointNewtonExact, adjointLbfgs, computeEffectiveJacobian } from 'numopt-js';
if (typeof gradientDescent !== 'function') {
  throw new Error('ESM: gradientDescent export missing');
}
if (typeof adjointNewton !== 'function') {
  throw new Error('ESM: adjointNewton export missing');
}
if (typeof adjointNewtonExact !== 'function') {
  throw new Error('ESM: adjointNewtonExact export missing');
}
if (typeof adjointLbfgs !== 'function') throw new Error('ESM: adjointLbfgs export missing');
if (typeof computeEffectiveJacobian !== 'function') throw new Error('ESM: computeEffectiveJacobian export missing');
console.log('ESM: ok');
`.trimStart()
  );
  run('node esm.mjs', tmpDir);

  // 5) Validate shipped files exist in installed package
  writeFile(
    path.join(tmpDir, 'check-files.cjs'),
    `
const fs = require('fs');
const path = require('path');

// Do not rely on requiring package.json (may be blocked by "exports").
// Resolve the main entry and walk up to the package root.
const entryPath = require.resolve('numopt-js');
const pkgDir = path.resolve(path.dirname(entryPath), '..');
const requiredFiles = [
  'dist/index.js',
  'dist/index.cjs',
  'dist/index.browser.js',
  'dist/index.d.ts',
];

for (const rel of requiredFiles) {
  const p = path.join(pkgDir, rel);
  if (!fs.existsSync(p)) {
    throw new Error('Missing file in published package: ' + rel);
  }
}

console.log('FILES: ok');
`.trimStart()
  );
  run('node check-files.cjs', tmpDir);

  // 6) Execute the browser ESM bundle in Node (no DOM APIs in the library).
  writeFile(
    path.join(tmpDir, 'browser-smoke.mjs'),
    `
import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const entryPath = require.resolve('numopt-js');
const pkgDir = path.resolve(path.dirname(entryPath), '..');
const browserBundleUrl = pathToFileURL(path.join(pkgDir, 'dist/index.browser.js')).href;
const { gradientDescent, adjointNewton, adjointNewtonExact, adjointLbfgs, computeEffectiveJacobian } = await import(browserBundleUrl);

const effective = computeEffectiveJacobian(new Float64Array([1.5]), new Float64Array([2.25]),
  (parameters, states) => new Float64Array([parameters[0] - 1, states[0] - 2]),
  (parameters, states) => new Float64Array([states[0] - parameters[0] ** 2, 2 * (states[0] - parameters[0] ** 2)]));
if (Math.abs(effective.get(0, 0) - 1) > 1e-6 || Math.abs(effective.get(1, 0) - 3) > 1e-6) {
  throw new Error('BROWSER: effective Jacobian did not recover redundant implicit derivatives');
}
console.log('BROWSER: computeEffectiveJacobian ok');

const cost = (parameters) => parameters[0] * parameters[0] + parameters[1] * parameters[1];
const gradient = (parameters) => new Float64Array([2 * parameters[0], 2 * parameters[1]]);
const result = gradientDescent(new Float64Array([5, -3]), cost, gradient, {
  maxIterations: 200,
  tolerance: 1e-8,
  useLineSearch: true,
});

if (!result.converged) {
  throw new Error('BROWSER: gradientDescent did not converge');
}
if (Math.abs(result.finalParameters[0]) > 1e-6 || Math.abs(result.finalParameters[1]) > 1e-6) {
  throw new Error('BROWSER: finalParameters not near origin');
}
console.log('BROWSER: ok');
const newtonResult = adjointNewton(new Float64Array([2]), new Float64Array([-1]),
  (parameters, states) => parameters[0] ** 2 + states[0] ** 2,
  (parameters, states) => new Float64Array([parameters[0] + states[0] - 1]));
if (!newtonResult.converged || Math.abs(newtonResult.finalParameters[0] - 0.5) > 1e-6) {
  throw new Error('BROWSER: adjointNewton did not recover the feasible minimum');
}
console.log('BROWSER: adjointNewton ok');
const fullResult = adjointNewtonExact(new Float64Array([1]), new Float64Array([1]),
  (parameters, states) => (parameters[0] - 1) ** 2 / 2 + states[0],
  (parameters, states) => new Float64Array([states[0] - parameters[0] ** 2]));
if (!fullResult.converged || Math.abs(fullResult.finalParameters[0] - 1 / 3) > 1e-6) {
  throw new Error('BROWSER: adjointNewtonExact missed nonlinear state curvature');
}
console.log('BROWSER: adjointNewtonExact ok');
const limitedResult = adjointLbfgs(new Float64Array([1.4]), new Float64Array([-10]),
  (parameters, states) => (states[0] - 4) ** 2 + 0.1 * (parameters[0] - 2) ** 2,
  (parameters, states) => new Float64Array([states[0] - parameters[0] ** 2]),
  { historySize: 5, tolerance: 1e-7, constraintTolerance: 1e-12 });
if (!limitedResult.converged || Math.abs(limitedResult.finalParameters[0] - 2) > 1e-6) {
  throw new Error('BROWSER: adjointLbfgs did not recover the nonlinear feasible minimum');
}
console.log('BROWSER: adjointLbfgs ok');
`.trimStart()
  );
  run('node browser-smoke.mjs', tmpDir);

  console.log(`Pack smoke OK (temp: ${tmpDir})`);
} finally {
  // Keep the working tree clean: npm pack leaves a root .tgz otherwise.
  if (fs.existsSync(tgzPath)) {
    fs.unlinkSync(tgzPath);
  }
}

