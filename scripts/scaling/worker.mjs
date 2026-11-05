/**
 * Isolated scaling trials against built public solvers: warm timing or cold process peak RSS.
 * A benchmark-only callback stops at a common verified gradient, parameter, and feasibility target.
 */

import { performance } from 'node:perf_hooks';
import { adjointBfgs, adjointNewton, adjointNewtonExact } from '../../dist/index.js';
import { createScalingProblem } from './problems.mjs';

const KIBIBYTE = 1024;
const GRADIENT_TARGET = 1e-6;
const PARAMETER_TARGET = 1e-6;
const CONSTRAINT_TARGET = 1e-12;
const MAX_ITERATIONS = 100;
const SOLVER_TOLERANCE_FACTOR = 0.01;

export function rootMeanSquare(values) {
  let sum = 0;
  for (const value of values) sum += value * value;
  return Math.sqrt(sum / values.length);
}

function parameterError(parameters, target) {
  let sum = 0;
  for (const value of parameters) sum += (value - target) ** 2;
  return Math.sqrt(sum / parameters.length);
}

class TargetReached extends Error {
  constructor(point) { super('Benchmark accuracy target reached'); this.point = point; }
}

function runSolve(problem, config) {
  const counts = { objective: 0, constraint: 0, firstPartials: 0, secondPartials: 0 };
  let latestConstraint;
  const objective = (parameters, states) => { counts.objective++; return problem.objective(parameters, states); };
  const constraint = (parameters, states) => {
    counts.constraint++;
    const values = problem.constraint(parameters, states);
    latestConstraint = { parameters, states, values };
    return values;
  };
  const options = {
    maxIterations: MAX_ITERATIONS,
    tolerance: SOLVER_TOLERANCE_FACTOR * GRADIENT_TARGET * Math.sqrt(config.parameters),
    constraintTolerance: CONSTRAINT_TARGET * Math.sqrt(config.states)
  };
  for (const name of ['dfdp', 'dfdx', 'dcdp', 'dcdx']) {
    options[name] = (parameters, states) => {
      counts.firstPartials++;
      return problem.firstDerivatives[name](parameters, states);
    };
  }
  if (config.method === 'approximate') {
    for (const name of ['d2fdp2', 'd2fdx2']) options[name] = (parameters, states) => {
      counts.secondPartials++; return problem.approximateDerivatives[name](parameters, states);
    };
  } else if (config.method === 'full-dense') {
    options.lagrangianHessian = (parameters, states, lambda) => {
      counts.secondPartials++; return problem.lagrangianHessian(parameters, states, lambda);
    };
  } else if (config.method === 'full-hvp') {
    options.lagrangianHessianVector = (parameters, states, lambda, direction) => {
      counts.secondPartials++; return problem.lagrangianHessianVector(parameters, states, lambda, direction);
    };
  }
  const start = performance.now();
  options.onIteration = (iteration, cost, parameters) => {
    const gradientRms = rootMeanSquare(problem.reducedGradient(parameters));
    const errorRms = parameterError(parameters, problem.reference.target);
    if (gradientRms > GRADIENT_TARGET || errorRms > PARAMETER_TARGET) return;
    // Analytical derivative callbacks do not perturb the latest current-state constraint evaluation.
    if (!latestConstraint || !parameters.every((value, index) => value === latestConstraint.parameters[index])) {
      throw new Error('Cannot verify the state belonging to the target iterate');
    }
    const constraintRms = rootMeanSquare(latestConstraint.values);
    if (!Number.isFinite(constraintRms) || constraintRms > CONSTRAINT_TARGET || !Number.isFinite(cost)) return;
    throw new TargetReached({ status: 'success', iterations: iteration, elapsedMs: performance.now() - start,
      gradientRms, errorRms, constraintRms, counts: { ...counts } });
  };
  const solver = config.method === 'bfgs' ? adjointBfgs : config.method.startsWith('approximate') ? adjointNewton : adjointNewtonExact;
  try {
    const result = solver(problem.initialParameters, problem.initialStates, objective, constraint, options);
    const gradientRms = rootMeanSquare(problem.reducedGradient(result.finalParameters));
    const errorRms = parameterError(result.finalParameters, problem.reference.target);
    const constraintRms = rootMeanSquare(problem.constraint(result.finalParameters, result.finalStates));
    return { status: gradientRms <= GRADIENT_TARGET && errorRms <= PARAMETER_TARGET && constraintRms <= CONSTRAINT_TARGET ?
      'success' : 'target-missed', iterations: result.iterations, elapsedMs: performance.now() - start,
      gradientRms, errorRms, constraintRms, counts, solverStopReason: result.stopReason };
  } catch (error) {
    if (error instanceof TargetReached) return error.point;
    return { status: 'error', message: String(error), elapsedMs: performance.now() - start, counts };
  }
}

export function runWorker(config) {
  if (!['bfgs', 'approximate', 'full-dense', 'full-hvp', 'approximate-fd', 'full-fd'].includes(config.method)) throw new Error('Unknown method');
  if (!['timing', 'memory'].includes(config.kind)) throw new Error('Unknown worker kind');
  global.gc?.();
  const baseline = { memory: process.memoryUsage(), peakRssBytes: process.resourceUsage().maxRSS * KIBIBYTE };
  const setupStart = performance.now();
  const problem = createScalingProblem(config);
  const setupMs = performance.now() - setupStart;
  const inputMemory = process.memoryUsage();
  if (config.kind === 'timing') {
    for (let index = 0; index < config.warmups; index++) runSolve(problem, config);
    const trials = [];
    for (let index = 0; index < config.repeats; index++) {
      global.gc?.();
      trials.push(runSolve(problem, config));
    }
    return { config, setupMs, trials };
  }
  const trial = runSolve(problem, config);
  return { config, setupMs, baseline, inputMemory, trial,
    peakRssBytes: process.resourceUsage().maxRSS * KIBIBYTE,
    finalMemory: process.memoryUsage() };
}

if (process.argv[1]?.endsWith('worker.mjs')) {
  try { process.stdout.write(JSON.stringify(runWorker(JSON.parse(process.argv[2])))); }
  catch (error) { process.stdout.write(JSON.stringify({ status: 'worker-error', message: String(error) })); process.exitCode = 1; }
}
