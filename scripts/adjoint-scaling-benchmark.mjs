/**
 * Serial, process-isolated time and peak-RSS scaling comparisons with resumable raw results.
 * Compiled public solvers are measured without a TypeScript-loader memory baseline.
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn, execFileSync } from 'node:child_process';
import { summarize, compareTiming } from './scaling/statistics.mjs';

const DEFAULT_TIME_REPEATS = 3;
const DEFAULT_MEMORY_REPEATS = 3;
const DEFAULT_WARMUPS = 2;
const DEFAULT_CHILD_TIMEOUT_MS = 240000;
const CHILD_HEAP_LIMIT_MIB = 1024;
const BYTES_PER_MIB = 1024 * 1024;
const METHODS = ['bfgs', 'approximate', 'full-dense', 'full-hvp'];
const AVAILABLE_METHODS = [...METHODS, 'approximate-fd', 'full-fd', 'lbfgs', 'lbfgs-5', 'lbfgs-20'];
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const WORKER = path.join(ROOT, 'scripts', 'scaling', 'worker.mjs');

function option(name, fallback) {
  const value = process.argv.find(argument => argument.startsWith(`--${name}=`));
  return value ? value.slice(name.length + '--='.length) : fallback;
}

function positiveInteger(value, name) {
  const number = Number(value);
  if (!Number.isInteger(number) || number <= 0) throw new Error(`${name} must be a positive integer`);
  return number;
}

const profile = option('profile', 'full');
const repeats = positiveInteger(option('repeats', DEFAULT_TIME_REPEATS), 'repeats');
const warmups = positiveInteger(option('warmups', DEFAULT_WARMUPS), 'warmups');
const childTimeoutMs = positiveInteger(option('timeout-ms', DEFAULT_CHILD_TIMEOUT_MS), 'timeout-ms');
const memoryRepeats = Number(option('memory-repeats', DEFAULT_MEMORY_REPEATS));
if (!Number.isInteger(memoryRepeats) || memoryRepeats < 0) throw new Error('memory-repeats must be a nonnegative integer');
const name = option('name', `adjoint-scaling-${profile}`);
if (!/^[a-z0-9-]+$/.test(name)) throw new Error('Invalid report name');
const families = option('families', 'easy,curved').split(',');
if (families.some(family => !['easy', 'curved', 'heterogeneous', 'ill-conditioned'].includes(family))) throw new Error('Invalid family');
const selectedMethods = option('methods', METHODS.join(',')).split(',');
if (selectedMethods.some(method => !AVAILABLE_METHODS.includes(method))) throw new Error('Invalid method');
const explicitToleranceFactor = option('solver-tolerance-factor', undefined);
const solverToleranceFactor = explicitToleranceFactor === undefined ? undefined : Number(explicitToleranceFactor);
if (solverToleranceFactor !== undefined && (!Number.isFinite(solverToleranceFactor) || solverToleranceFactor <= 0)) {
  throw new Error('solver-tolerance-factor must be finite and positive');
}

function grid() {
  const explicit = option('sizes', '');
  if (explicit) return explicit.split(',').map(pair => {
    const [parameters, states] = pair.split('x').map(value => positiveInteger(value, 'size'));
    if (!states) throw new Error('Sizes use parameterCount x stateCount');
    return { parameters, states };
  });
  if (profile === 'pilot') return [[16, 128], [64, 128], [128, 128], [32, 256], [128, 256], [256, 256]]
    .map(([parameters, states]) => ({ parameters, states }));
  if (profile !== 'full') throw new Error('Unknown profile');
  const pairs = [[8, 128], [16, 128], [32, 128], [64, 128], [128, 128], [256, 128], [512, 128],
    [16, 256], [16, 512], [16, 1024], [64, 256], [128, 256], [256, 256], [256, 512], [512, 512]];
  return pairs.map(([parameters, states]) => ({ parameters, states }));
}

function child(config) {
  return new Promise(resolve => {
    const processHandle = spawn(process.execPath, ['--expose-gc', `--max-old-space-size=${CHILD_HEAP_LIMIT_MIB}`,
      WORKER, JSON.stringify(config)], { cwd: ROOT, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    let output = '';
    let errors = '';
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; processHandle.kill(); }, childTimeoutMs);
    processHandle.stdout.on('data', chunk => { output += chunk; });
    processHandle.stderr.on('data', chunk => { errors += chunk; });
    processHandle.on('error', error => { clearTimeout(timeout); resolve({ status: 'spawn-error', message: String(error) }); });
    processHandle.on('close', code => {
      clearTimeout(timeout);
      if (timedOut) return resolve({ status: 'timeout', timeoutMs: childTimeoutMs });
      try { resolve(JSON.parse(output)); }
      catch { resolve({ status: 'child-error', code, message: errors.slice(-2000), output: output.slice(-1000) }); }
    });
  });
}

const outputDirectory = path.join(ROOT, 'benchmark-results');
fs.mkdirSync(outputDirectory, { recursive: true });
const jsonPath = path.join(outputDirectory, `${name}.json`);
const protocol = { profile, repeats, memoryRepeats, warmups, timeoutMs: childTimeoutMs,
  childHeapLimitMiB: CHILD_HEAP_LIMIT_MIB, families, methods: selectedMethods, sizes: grid(),
  gradientRmsTarget: 1e-6, parameterRmsTarget: 1e-6, constraintRmsTarget: 1e-12, solverToleranceFactor };
const environment = { timestamp: new Date().toISOString(), node: process.version, platform: process.platform,
  architecture: process.arch, cpu: os.cpus()[0]?.model, totalMemoryBytes: os.totalmem(),
  commit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim() };
const previous = process.argv.includes('--resume') && fs.existsSync(jsonPath) ? JSON.parse(fs.readFileSync(jsonPath, 'utf8')) : undefined;
if (previous && JSON.stringify(previous.protocol) !== JSON.stringify(protocol)) throw new Error('Resume protocol mismatch');
const rows = previous?.rows ?? [];
function save() { fs.writeFileSync(jsonPath, JSON.stringify({ environment, protocol, rows }, null, 2)); }

for (const family of families) {
  for (const size of protocol.sizes) {
    // Rotate method order deterministically without running competing measured processes.
    const rotation = protocol.sizes.indexOf(size) % selectedMethods.length;
    const methods = [...selectedMethods.slice(rotation), ...selectedMethods.slice(0, rotation)];
    for (const method of methods) {
      if (rows.some(row => row.family === family && row.parameters === size.parameters && row.states === size.states && row.method === method)) continue;
      const config = { ...size, family, method, repeats, warmups, solverToleranceFactor };
      const timing = await child({ ...config, kind: 'timing' });
      const memories = [];
      if (timing.trials?.every(trial => trial.status === 'success')) {
        for (let repeat = 0; repeat < memoryRepeats; repeat++) memories.push(await child({ ...config, kind: 'memory' }));
      }
      const validTimes = timing.trials?.filter(trial => trial.status === 'success') ?? [];
      const validMemories = memories.filter(memory => memory.trial?.status === 'success');
      const row = { ...size, family, method, timing, memories,
        status: validTimes.length === repeats && validMemories.length === memoryRepeats ? 'success' : 'incomplete',
        timeMs: validTimes.length ? summarize(validTimes.map(trial => trial.elapsedMs)) : undefined,
        iterations: validTimes.length ? summarize(validTimes.map(trial => trial.iterations)) : undefined,
        peakRssMiB: validMemories.length ? summarize(validMemories.map(memory => memory.peakRssBytes / BYTES_PER_MIB)) : undefined };
      rows.push(row);
      save();
      console.log(`${family} m=${size.parameters} n=${size.states} ${method}: ${row.status}; ` +
        `${row.timeMs?.median.toFixed(2) ?? timing.status} ms; peak RSS ${row.peakRssMiB?.median.toFixed(1) ?? '?'} MiB`);
    }
  }
}

const lines = ['# Newton–adjoint dimension scaling', '',
  `Environment: ${environment.node}; ${environment.cpu}; ${environment.platform}/${environment.architecture}; commit ${environment.commit}.`,
  `Timing: ${repeats} retained runs after ${warmups} warmup in an isolated process, forced GC outside each measured solve.`,
  `Memory: ${memoryRepeats} independent cold processes, peak RSS from process.resourceUsage().maxRSS (KiB converted to MiB).`,
  'Peak RSS is the entire process, including Node, imports, input matrices, JIT and solver allocations. It is not heapUsed or an allocation estimate.',
  'Time excludes Node startup and problem setup; memory includes them. Raw files retain pre-input and post-input RSS snapshots.',
  'A benchmark callback terminates at true reduced gradient RMS <= 1e-6, parameter error RMS <= 1e-6 and actual state residual RMS <= 1e-12.',
  `Solver stopping tolerance = ${solverToleranceFactor ?? 0.01} * 1e-6 * sqrt(m), while the common target remains unchanged.`,
  'First derivatives are analytical. Curvature is analytical except for methods ending in -fd (central second differences only).',
  'Full-dense, full-hvp and full-fd use the same complete mathematical Hessian model; HVP avoids the joint Hessian allocation.',
  'lbfgs stores 10 correction pairs; lbfgs-5 and lbfgs-20 select 5 and 20. Only the quasi-Newton update is O(historySize * m); state solves remain dense.',
  'Both families have a well-conditioned coupled lower-bidiagonal state Jacobian, represented by dense ml-matrix matrices.',
  'easy: reduced cost 1/2 ||p-1||². curved: sum[e^4/4 + e^3/3 + 3e²/2], e=p-1, with persistent state curvature.',
  'heterogeneous: the curved objective with distinct initial component values, rather than a replicated scalar trajectory.',
  'ill-conditioned: heterogeneous starts and logarithmically spaced parameter weights 1..1000 (reduced Hessian condition ratio 1000 at the solution); iteration limit 1000 instead of 100.',
  `Per-child wall limit ${childTimeoutMs / 1000}s and V8 old-heap cap ${CHILD_HEAP_LIMIT_MIB} MiB; failures/timeouts are not extrapolated timings.`, '',
  '| Family | m | n | Method | Status | Iterations | Time ms [IQR] | Peak RSS MiB [IQR] | Time / BFGS | Timing intervals vs BFGS |',
  '|---|---:|---:|---|---|---:|---:|---:|---:|---|'];
for (const row of rows) {
  const reference = rows.find(other => other.family === row.family && other.parameters === row.parameters && other.states === row.states && other.method === 'bfgs');
  const comparison = compareTiming(row.timeMs, reference?.timeMs);
  lines.push(`| ${row.family} | ${row.parameters} | ${row.states} | ${row.method} | ${row.status} | ${row.iterations?.median ?? 'n/a'} | ` +
    `${row.timeMs ? `${row.timeMs.median.toFixed(2)} [${row.timeMs.iqr.toFixed(2)}]` : 'n/a'} | ` +
    `${row.peakRssMiB ? `${row.peakRssMiB.median.toFixed(1)} [${row.peakRssMiB.iqr.toFixed(1)}]` : 'n/a'} | ` +
    `${comparison.ratio?.toFixed(2) ?? 'n/a'} | ${comparison.classification} |`);
}
lines.push('', 'Quartile separation is a descriptive comparison, not a confidence interval or a significance test.',
  'There is no dimension-only universal crossover: curvature, derivative representation, conditioning, and solver implementation matter.',
  'These measurements characterize the current dense implementation, not sparse PDE solvers.');
fs.writeFileSync(path.join(outputDirectory, `${name}.md`), `${lines.join('\n')}\n`);
console.log(`Reports: ${jsonPath}`);
if (rows.some(row => row.status !== 'success')) process.exitCode = 1;
