/**
 * This script benchmarks CMA-ES on standard test functions.
 *
 * Role in system:
 * - Sanity-checks CMA-ES behavior on canonical problems
 * - Provides quick feedback on convergence and performance
 * - Helps compare settings against expected improvements
 *
 * For first-time readers:
 * - Run with: npx tsx scripts/cmaes-benchmark.ts
 * - Review PASS/FAIL status per benchmark case
 */

import { cmaEs } from '../src/index.js';
import type { CostFn } from '../src/core/types.js';

type BenchmarkCase = {
  name: string;
  dimension: number;
  initialValue: number;
  initialStepSize: number;
  maxIterations: number;
  populationSize: number | undefined;
  randomSeed: number;
  targetCost: number;
  expectedMaxFinalCost: number;
  restartStrategy?: 'none' | 'ipop';
  costFunction: CostFn;
};

const DEFAULT_RANDOM_SEED = 123456;
const DEFAULT_SPHERE_INITIAL_VALUE = 10.0;
const DEFAULT_ROSENBROCK_INITIAL_VALUE = -2.0;
const DEFAULT_RASTRIGIN_INITIAL_VALUE = 5.0;

const SPHERE_DIMENSION = 10;
const ROSENBROCK_DIMENSION = 5;
const RASTRIGIN_DIMENSION = 10;
const RASTRIGIN_EXTENDED_DIMENSION = 10;

const SPHERE_STEP_SIZE = 2.0;
const ROSENBROCK_STEP_SIZE = 1.0;
const RASTRIGIN_STEP_SIZE = 2.5;
const RASTRIGIN_EXTENDED_STEP_SIZE = 4.0;

const SPHERE_MAX_ITERATIONS = 200;
const ROSENBROCK_MAX_ITERATIONS = 300;
const RASTRIGIN_MAX_ITERATIONS = 400;
const RASTRIGIN_EXTENDED_MAX_ITERATIONS = 1200;

const SPHERE_TARGET_COST = 1e-8;
const ROSENBROCK_TARGET_COST = 1e-6;
const RASTRIGIN_TARGET_COST = 1e-4;
const RASTRIGIN_EXTENDED_TARGET_COST = 1e-6;

const SPHERE_EXPECTED_MAX_COST = 1e-4;
const ROSENBROCK_EXPECTED_MAX_COST = 1e-2;
const RASTRIGIN_EXPECTED_MAX_COST = 5.0;
const RASTRIGIN_EXTENDED_EXPECTED_MAX_COST = 1.0;
const RASTRIGIN_EXTENDED_POPULATION_SIZE = 80;

const ROSEBROCK_ALPHA = 100.0;
const RASTRIGIN_A = 10.0;
const TWO_PI = 2.0 * Math.PI;

function createInitialParameters(dimension: number, initialValue: number): Float64Array {
  const parameters = new Float64Array(dimension);
  parameters.fill(initialValue);
  return parameters;
}

function sphereCost(parameters: Float64Array): number {
  let sum = 0.0;
  for (let index = 0; index < parameters.length; index++) {
    const value = parameters[index];
    sum += value * value;
  }
  return sum;
}

function rosenbrockCost(parameters: Float64Array): number {
  let sum = 0.0;
  for (let index = 0; index < parameters.length - 1; index++) {
    const x = parameters[index];
    const y = parameters[index + 1];
    const term1 = 1.0 - x;
    const term2 = y - x * x;
    sum += term1 * term1 + ROSEBROCK_ALPHA * term2 * term2;
  }
  return sum;
}

function rastriginCost(parameters: Float64Array): number {
  let sum = RASTRIGIN_A * parameters.length;
  for (let index = 0; index < parameters.length; index++) {
    const value = parameters[index];
    sum += value * value - RASTRIGIN_A * Math.cos(TWO_PI * value);
  }
  return sum;
}

function buildCases(): BenchmarkCase[] {
  return [
    {
      name: 'Sphere',
      dimension: SPHERE_DIMENSION,
      initialValue: DEFAULT_SPHERE_INITIAL_VALUE,
      initialStepSize: SPHERE_STEP_SIZE,
      maxIterations: SPHERE_MAX_ITERATIONS,
      populationSize: undefined,
      randomSeed: DEFAULT_RANDOM_SEED,
      targetCost: SPHERE_TARGET_COST,
      expectedMaxFinalCost: SPHERE_EXPECTED_MAX_COST,
      restartStrategy: 'none',
      costFunction: sphereCost
    },
    {
      name: 'Rosenbrock',
      dimension: ROSENBROCK_DIMENSION,
      initialValue: DEFAULT_ROSENBROCK_INITIAL_VALUE,
      initialStepSize: ROSENBROCK_STEP_SIZE,
      maxIterations: ROSENBROCK_MAX_ITERATIONS,
      populationSize: undefined,
      randomSeed: DEFAULT_RANDOM_SEED,
      targetCost: ROSENBROCK_TARGET_COST,
      expectedMaxFinalCost: ROSENBROCK_EXPECTED_MAX_COST,
      restartStrategy: 'none',
      costFunction: rosenbrockCost
    },
    {
      name: 'Rastrigin',
      dimension: RASTRIGIN_DIMENSION,
      initialValue: DEFAULT_RASTRIGIN_INITIAL_VALUE,
      initialStepSize: RASTRIGIN_STEP_SIZE,
      maxIterations: RASTRIGIN_MAX_ITERATIONS,
      populationSize: 40,
      randomSeed: DEFAULT_RANDOM_SEED,
      targetCost: RASTRIGIN_TARGET_COST,
      expectedMaxFinalCost: RASTRIGIN_EXPECTED_MAX_COST,
      restartStrategy: 'none',
      costFunction: rastriginCost
    },
    {
      name: 'Rastrigin (extended budget)',
      dimension: RASTRIGIN_EXTENDED_DIMENSION,
      initialValue: DEFAULT_RASTRIGIN_INITIAL_VALUE,
      initialStepSize: RASTRIGIN_EXTENDED_STEP_SIZE,
      maxIterations: RASTRIGIN_EXTENDED_MAX_ITERATIONS,
      populationSize: RASTRIGIN_EXTENDED_POPULATION_SIZE,
      randomSeed: DEFAULT_RANDOM_SEED,
      targetCost: RASTRIGIN_EXTENDED_TARGET_COST,
      expectedMaxFinalCost: RASTRIGIN_EXTENDED_EXPECTED_MAX_COST,
      restartStrategy: 'none',
      costFunction: rastriginCost
    },
    {
      name: 'Rastrigin (IPOP)',
      dimension: RASTRIGIN_DIMENSION,
      initialValue: DEFAULT_RASTRIGIN_INITIAL_VALUE,
      initialStepSize: RASTRIGIN_STEP_SIZE,
      maxIterations: 1200,
      populationSize: 20,
      randomSeed: DEFAULT_RANDOM_SEED,
      targetCost: RASTRIGIN_TARGET_COST,
      expectedMaxFinalCost: 5.0,
      restartStrategy: 'ipop',
      costFunction: rastriginCost
    }
  ];
}

function computeInitialCost(caseConfig: BenchmarkCase): number {
  const initialParameters = createInitialParameters(caseConfig.dimension, caseConfig.initialValue);
  return caseConfig.costFunction(initialParameters);
}

function evaluateCase(caseConfig: BenchmarkCase): {
  name: string;
  dimension: number;
  initialCost: number;
  finalCost: number;
  iterations: number;
  functionEvaluations: number;
  finalStepSize: number;
  status: 'PASS' | 'FAIL';
  elapsedMs: number;
} {
  const initialParameters = createInitialParameters(caseConfig.dimension, caseConfig.initialValue);
  const initialCost = computeInitialCost(caseConfig);

  const startMs = Date.now();
  const result = cmaEs(initialParameters, caseConfig.costFunction, {
    maxIterations: caseConfig.maxIterations,
    populationSize: caseConfig.populationSize,
    initialStepSize: caseConfig.initialStepSize,
    randomSeed: caseConfig.randomSeed,
    targetCost: caseConfig.targetCost,
    restartStrategy: caseConfig.restartStrategy,
    profiling: true
  });
  const elapsedMs = Date.now() - startMs;

  const status = result.finalCost <= caseConfig.expectedMaxFinalCost ? 'PASS' : 'FAIL';

  return {
    name: caseConfig.name,
    dimension: caseConfig.dimension,
    initialCost,
    finalCost: result.finalCost,
    iterations: result.iterations,
    functionEvaluations: result.functionEvaluations,
    finalStepSize: result.finalStepSize,
    status,
    elapsedMs,
    profiling: result.profiling
  };
}

function printResults(results: Array<ReturnType<typeof evaluateCase>>): void {
  console.log('CMA-ES benchmark results');
  for (const result of results) {
    console.log(
      `${result.status} | ${result.name} (dim=${result.dimension})` +
        ` | initial=${result.initialCost.toExponential(3)}` +
        ` | final=${result.finalCost.toExponential(3)}` +
        ` | iters=${result.iterations}` +
        ` | fevals=${result.functionEvaluations}` +
        ` | sigma=${result.finalStepSize.toExponential(3)}` +
        ` | time=${result.elapsedMs}ms`
    );
    if (result.profiling) {
      console.log(
        `  profile(ms): total=${result.profiling.totalMs.toFixed(2)}, ` +
          `cost=${result.profiling.costMs.toFixed(2)}, ` +
          `cholesky=${result.profiling.choleskyMs.toFixed(2)}, ` +
          `sampling=${result.profiling.samplingMs.toFixed(2)}, ` +
          `update=${result.profiling.updateMs.toFixed(2)}`
      );
    }
  }
}

function main(): void {
  const cases = buildCases();
  const results = cases.map(evaluateCase);
  printResults(results);
}

main();

