/**
 * Shared timing / table helpers for constrained optimizer benchmarks
 * and the paper-grade protocol (counters, repeats, success).
 * Suite scripts own problem definitions; this module owns orchestration primitives.
 */

import { execSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import type {
  ConstrainedCostFn,
  ConstrainedResidualFn,
  ConstraintFn,
  CostFn,
  GradientFn,
  ResidualFn
} from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

const FIRST_QUARTILE_PROBABILITY = 0.25;
const THIRD_QUARTILE_PROBABILITY = 0.75;
const UNKNOWN_ENVIRONMENT_VALUE = 'unknown';

export type BenchmarkRow = {
  Method: string;
  Iterations: number | string;
  TimeMs: number;
  Converged: boolean;
  FinalCost: number | string;
  ConstraintNorm: number | string;
  Error: string;
  Penalty?: number | string;
};

export type ConstrainedSolveResult = {
  finalParameters: Float64Array;
  finalStates: Float64Array;
  iterations: number;
  converged: boolean;
  finalCost: number;
};

export type BenchmarkSolver<Problem> = {
  name: string;
  run: (problem: Problem, initial: { parameters: Float64Array; states: Float64Array }) => ConstrainedSolveResult;
};

export function concatParameterAndState(parameters: Float64Array, states: Float64Array): Float64Array {
  const combined = new Float64Array(parameters.length + states.length);
  combined.set(parameters, 0);
  combined.set(states, parameters.length);
  return combined;
}

export function splitParameterAndState(
  combined: Float64Array,
  parameterCount: number
): { parameters: Float64Array; states: Float64Array } {
  return {
    parameters: combined.slice(0, parameterCount),
    states: combined.slice(parameterCount)
  };
}

function appendPenalizedConstraint(
  baseResidual: Float64Array,
  constraintValues: Float64Array,
  penaltyWeight: number
): Float64Array {
  const sqrtPenaltyWeight = Math.sqrt(penaltyWeight);
  const penalized = new Float64Array(baseResidual.length + constraintValues.length);
  penalized.set(baseResidual, 0);
  for (let index = 0; index < constraintValues.length; index++) {
    penalized[baseResidual.length + index] = sqrtPenaltyWeight * constraintValues[index];
  }
  return penalized;
}

export function buildPenaltyResidual(options: {
  parameterCount: number;
  residual: (parameters: Float64Array, states: Float64Array) => Float64Array;
  constraint: (parameters: Float64Array, states: Float64Array) => Float64Array;
  penaltyWeight: number;
}): ResidualFn {
  return (combined: Float64Array): Float64Array => {
    const { parameters, states } = splitParameterAndState(combined, options.parameterCount);
    return appendPenalizedConstraint(
      options.residual(parameters, states),
      options.constraint(parameters, states),
      options.penaltyWeight
    );
  };
}

export function buildStateOnlyPenaltyResidual(options: {
  residual: (states: Float64Array) => Float64Array;
  constraint: (states: Float64Array) => Float64Array;
  penaltyWeight: number;
}): ResidualFn {
  return (states: Float64Array): Float64Array => {
    return appendPenalizedConstraint(
      options.residual(states),
      options.constraint(states),
      options.penaltyWeight
    );
  };
}

export function constraintNormFor(
  constraint: (parameters: Float64Array, states: Float64Array) => Float64Array,
  result: ConstrainedSolveResult
): number {
  return vectorNorm(constraint(result.finalParameters, result.finalStates));
}

export function timeConstrainedSolve(options: {
  methodName: string;
  run: () => ConstrainedSolveResult;
  constraintNorm: (result: ConstrainedSolveResult) => number;
}): BenchmarkRow {
  const start = performance.now();
  try {
    const result = options.run();
    const elapsedMs = performance.now() - start;
    return {
      Method: options.methodName,
      Iterations: result.iterations,
      TimeMs: Number(elapsedMs.toFixed(3)),
      Converged: result.converged,
      FinalCost: Number(result.finalCost.toExponential(3)),
      ConstraintNorm: Number(options.constraintNorm(result).toExponential(3)),
      Error: ''
    };
  } catch (error) {
    const elapsedMs = performance.now() - start;
    const message = error instanceof Error ? error.message : String(error);
    return {
      Method: options.methodName,
      Iterations: 'error',
      TimeMs: Number(elapsedMs.toFixed(3)),
      Converged: false,
      FinalCost: 'error',
      ConstraintNorm: 'error',
      Error: message
    };
  }
}

export function runSolverTable<Problem extends {
  constraint: (parameters: Float64Array, states: Float64Array) => Float64Array;
}>(options: {
  problem: Problem;
  initial: { parameters: Float64Array; states: Float64Array };
  solvers: Array<BenchmarkSolver<Problem>>;
}): BenchmarkRow[] {
  return options.solvers.map((solver) =>
    timeConstrainedSolve({
      methodName: solver.name,
      run: () => solver.run(options.problem, options.initial),
      constraintNorm: (result) => constraintNormFor(options.problem.constraint, result)
    })
  );
}

export type EvaluationCounters = {
  cost: number;
  gradient: number;
  residual: number;
  constraint: number;
};

export type PaperSuccessCriteria = {
  parameterError: number;
  parameterTolerance: number;
  constraintNorm?: number;
  constraintTolerance?: number;
};

export type PaperTrial = {
  elapsedMs: number;
  iterations: number;
  finalCost: number;
  parameterError: number;
  constraintNorm?: number;
  decisionVariableCount: number;
  counters: EvaluationCounters;
  threw: boolean;
  throwMessage?: string;
};

export type NumericSummary = {
  median: number;
  interquartileRange: number;
};

export type PaperEnvironment = {
  timestamp: string;
  nodeVersion: string;
  platform: string;
  architecture: string;
  cpuModel: string;
  packageVersion: string;
  gitCommit: string;
};

export function createEvaluationCounters(): EvaluationCounters {
  return { cost: 0, gradient: 0, residual: 0, constraint: 0 };
}

export function wrapCostFunction(costFunction: CostFn, counters: EvaluationCounters): CostFn {
  return (parameters) => {
    counters.cost += 1;
    return costFunction(parameters);
  };
}

export function wrapGradientFunction(
  gradientFunction: GradientFn,
  counters: EvaluationCounters
): GradientFn {
  return (parameters) => {
    counters.gradient += 1;
    return gradientFunction(parameters);
  };
}

export function wrapResidualFunction(
  residualFunction: ResidualFn,
  counters: EvaluationCounters
): ResidualFn {
  return (parameters) => {
    counters.residual += 1;
    return residualFunction(parameters);
  };
}

export function wrapConstraintFunction(
  constraintFunction: ConstraintFn,
  counters: EvaluationCounters
): ConstraintFn {
  return (parameters, states) => {
    counters.constraint += 1;
    return constraintFunction(parameters, states);
  };
}

export function wrapConstrainedCostFunction(
  costFunction: ConstrainedCostFn,
  counters: EvaluationCounters
): ConstrainedCostFn {
  return (parameters, states) => {
    counters.cost += 1;
    return costFunction(parameters, states);
  };
}

export function wrapConstrainedResidualFunction(
  residualFunction: ConstrainedResidualFn,
  counters: EvaluationCounters
): ConstrainedResidualFn {
  return (parameters, states) => {
    counters.residual += 1;
    return residualFunction(parameters, states);
  };
}

export function computeParameterError(parameters: Float64Array, expected: Float64Array): number {
  if (parameters.length !== expected.length) {
    throw new Error(
      `Parameter error requires matching lengths, got ${parameters.length} and ${expected.length}`
    );
  }
  const difference = new Float64Array(parameters.length);
  for (let index = 0; index < parameters.length; index++) {
    difference[index] = parameters[index] - expected[index];
  }
  return vectorNorm(difference);
}

export function isPaperSuccess(criteria: PaperSuccessCriteria): boolean {
  if (!Number.isFinite(criteria.parameterError) || criteria.parameterError > criteria.parameterTolerance) {
    return false;
  }
  if (criteria.constraintTolerance === undefined) {
    return true;
  }
  if (criteria.constraintNorm === undefined || !Number.isFinite(criteria.constraintNorm)) {
    return false;
  }
  return criteria.constraintNorm <= criteria.constraintTolerance;
}

export function computeMedian(values: number[]): number {
  if (values.length === 0) {
    throw new Error('Median requires at least one value');
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middleIndex = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return 0.5 * (sorted[middleIndex - 1] + sorted[middleIndex]);
  }
  return sorted[middleIndex];
}

export function computeQuantile(values: number[], probability: number): number {
  if (values.length === 0) {
    throw new Error('Quantile requires at least one value');
  }
  if (probability < 0 || probability > 1) {
    throw new Error(`Quantile probability must be in [0, 1], got ${probability}`);
  }
  const sorted = [...values].sort((left, right) => left - right);
  const interpolatedIndex = (sorted.length - 1) * probability;
  const lowerIndex = Math.floor(interpolatedIndex);
  const fraction = interpolatedIndex - lowerIndex;
  if (lowerIndex + 1 >= sorted.length) {
    return sorted[lowerIndex];
  }
  return sorted[lowerIndex] * (1 - fraction) + sorted[lowerIndex + 1] * fraction;
}

export function computeInterquartileRange(values: number[]): number {
  return computeQuantile(values, THIRD_QUARTILE_PROBABILITY) - computeQuantile(values, FIRST_QUARTILE_PROBABILITY);
}

export function summarizeNumeric(values: number[]): NumericSummary {
  return {
    median: computeMedian(values),
    interquartileRange: computeInterquartileRange(values)
  };
}

export function runWarmedRepeats<Result>(options: {
  warmupCount: number;
  repeatCount: number;
  run: (repeatIndex: number) => Result;
}): Result[] {
  for (let warmupIndex = 0; warmupIndex < options.warmupCount; warmupIndex++) {
    options.run(-1);
  }
  const timedResults: Result[] = [];
  for (let repeatIndex = 0; repeatIndex < options.repeatCount; repeatIndex++) {
    timedResults.push(options.run(repeatIndex));
  }
  return timedResults;
}

export type PaperTrialOutcome = Omit<PaperTrial, 'elapsedMs' | 'counters' | 'threw' | 'throwMessage'>;

export function timePaperTrial(run: (counters: EvaluationCounters) => PaperTrialOutcome): PaperTrial {
  const counters = createEvaluationCounters();
  const start = performance.now();
  try {
    const result = run(counters);
    return {
      ...result,
      elapsedMs: performance.now() - start,
      counters: { ...counters },
      threw: false
    };
  } catch (error) {
    return {
      elapsedMs: performance.now() - start,
      iterations: Number.NaN,
      finalCost: Number.POSITIVE_INFINITY,
      parameterError: Number.POSITIVE_INFINITY,
      decisionVariableCount: 0,
      counters: { ...counters },
      threw: true,
      throwMessage: error instanceof Error ? error.message : String(error)
    };
  }
}

export function collectPaperEnvironment(): PaperEnvironment {
  const packageJsonPath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8')) as { version?: string };
  return {
    timestamp: new Date().toISOString(),
    nodeVersion: process.version,
    platform: process.platform,
    architecture: process.arch,
    cpuModel: os.cpus()[0]?.model ?? UNKNOWN_ENVIRONMENT_VALUE,
    packageVersion: packageJson.version ?? UNKNOWN_ENVIRONMENT_VALUE,
    gitCommit: readGitCommit()
  };
}

function readGitCommit(): string {
  try {
    return execSync('git rev-parse HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  } catch {
    return UNKNOWN_ENVIRONMENT_VALUE;
  }
}
