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
const pack = JSON.parse(packJson);
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
console.log('CJS: ok');
`.trimStart()
  );
  run('node cjs.cjs', tmpDir);

  // 4) Validate ESM import works (mjs forces ESM even in a CJS package)
  writeFile(
    path.join(tmpDir, 'esm.mjs'),
    `
import { gradientDescent } from 'numopt-js';
if (typeof gradientDescent !== 'function') {
  throw new Error('ESM: gradientDescent export missing');
}
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

  console.log(`Pack smoke OK (temp: ${tmpDir})`);
} finally {
  // Keep the working tree clean: npm pack leaves a root .tgz otherwise.
  if (fs.existsSync(tgzPath)) {
    fs.unlinkSync(tgzPath);
  }
}

