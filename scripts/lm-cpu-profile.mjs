/** Sample the original public LM independently of diagnostic phase timers. */
import fs from 'node:fs';
import inspector from 'node:inspector';
import { runProblem } from '../integration/kibo/scenarios.mjs';

const SAMPLING_INTERVAL_MICROSECONDS = 100;
const PROFILE_DURATION_MILLISECONDS = 1500;
const MICROSECONDS_PER_MILLISECOND = 1000;
const HOT_FUNCTION_COUNT = 8;

function classify(stack) {
  const names = stack.map(node => node.callFrame.functionName);
  if (names.includes('(garbage collector)')) return 'garbageCollection';
  if (!names.includes('levenbergMarquardt')) return 'outsideLm';
  if (names.includes('computeJacobianMatrix')) return 'jacobian';
  if (names.includes('computeNormalEquationsMatrices')) return 'normalEquations';
  if (stack.some(node => node.callFrame.url.endsWith('/core/lmLinearSolver.js') ||
      node.callFrame.url.endsWith('/core/kiboLinearSolver.js'))) return 'linearSolver';
  if (names.includes('tryLevenbergMarquardtStep')) return 'stepControl';
  return 'lmControl';
}

function summarize(profile) {
  const nodes = new Map(profile.nodes.map(node => [node.id, node]));
  const parents = new Map(profile.nodes.flatMap(node => (node.children ?? []).map(child => [child, node.id])));
  const groups = {}, hotFunctions = {};
  profile.samples.forEach((identifier, index) => {
    const stack = [];
    let current = identifier;
    while (nodes.has(current)) { stack.push(nodes.get(current)); current = parents.get(current); }
    const milliseconds = profile.timeDeltas[index] / MICROSECONDS_PER_MILLISECOND;
    const group = classify(stack);
    groups[group] = (groups[group] ?? 0) + milliseconds;
    if (group === 'outsideLm' || group === 'garbageCollection') return;
    const frame = stack[0].callFrame;
    const location = `${frame.functionName || '(anonymous)'} ${frame.url}:${frame.lineNumber + 1}`;
    hotFunctions[location] = (hotFunctions[location] ?? 0) + milliseconds;
  });
  const lmMilliseconds = Object.entries(groups).reduce((total, [name, elapsed]) =>
    total + (name === 'outsideLm' || name === 'garbageCollection' ? 0 : elapsed), 0);
  if (!lmMilliseconds) throw new Error('CPU profiler did not sample the public LM');
  return {
    samplingIntervalMicroseconds: SAMPLING_INTERVAL_MICROSECONDS, samples: profile.samples.length,
    sampledMilliseconds: groups, lmMilliseconds,
    lmShares: Object.fromEntries(Object.entries(groups).filter(([name]) =>
      name !== 'outsideLm' && name !== 'garbageCollection').map(([name, elapsed]) => [name, elapsed / lmMilliseconds])),
    hottestFunctions: Object.entries(hotFunctions).sort((left, right) => right[1] - left[1]).slice(0, HOT_FUNCTION_COUNT)
  };
}

export async function collectCpuProfile(api, wasmModule, LinearAlgebra, problem, backend, filename) {
  const session = new inspector.Session();
  session.connect();
  const post = (method, parameters = {}) => new Promise((resolve, reject) =>
    session.post(method, parameters, (error, response) => error ? reject(error) : resolve(response)));
  try {
    await post('Profiler.enable');
    await post('Profiler.setSamplingInterval', { interval: SAMPLING_INTERVAL_MICROSECONDS });
    await post('Profiler.start');
    const started = performance.now();
    let runs = 0;
    while (performance.now() - started < PROFILE_DURATION_MILLISECONDS) {
      runProblem(api, wasmModule, LinearAlgebra, problem, backend); runs++;
    }
    const { profile } = await post('Profiler.stop');
    fs.writeFileSync(filename, JSON.stringify(profile));
    return { problem: problem.name, backend, runs, filename, ...summarize(profile) };
  } finally { session.disconnect(); }
}
