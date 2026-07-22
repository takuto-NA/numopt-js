/**
 * CMA-ES unit tests: determinism, Sphere progress, evaluation budget, IPOP, profiling.
 */

import { cmaEs } from '../src/core/cmaEs';
import type { CostFn } from '../src/core/types';

const SPHERE_DIMENSION = 5;
const DETERMINISM_SEED = 123456;
const DETERMINISM_MAX_ITERATIONS = 120;
const DETERMINISM_INITIAL_STEP_SIZE = 2.0;
const DETERMINISM_POPULATION_SIZE = 20;

const SPHERE_MAX_ITERATIONS = 200;
const SPHERE_INITIAL_STEP_SIZE = 3.0;
const SPHERE_POPULATION_SIZE = 30;
const SPHERE_TARGET_COST = 1e-8;
const SPHERE_EXPECTED_MAX_FINAL_COST = 1e-4;

const BUDGET_MAX_FUNCTION_EVALUATIONS = 15;
const BUDGET_POPULATION_SIZE = 10;

const IPOP_DIMENSION = 10;
const IPOP_INITIAL_VALUE = 5.0;
const IPOP_MAX_ITERATIONS = 1200;
const IPOP_POPULATION_SIZE = 20;
const IPOP_INITIAL_STEP_SIZE = 2.5;
const IPOP_TARGET_COST = 1e-4;
const IPOP_EXPECTED_MAX_FINAL_COST = 5.0;
const RASTRIGIN_A = 10.0;
const TWO_PI = 2.0 * Math.PI;

const sphereCost: CostFn = (parameters: Float64Array) => {
  let sum = 0.0;
  for (let index = 0; index < parameters.length; index++) {
    const value = parameters[index];
    sum += value * value;
  }
  return sum;
};

const rastriginCost: CostFn = (parameters: Float64Array) => {
  let sum = RASTRIGIN_A * parameters.length;
  for (let index = 0; index < parameters.length; index++) {
    const value = parameters[index];
    sum += value * value - RASTRIGIN_A * Math.cos(TWO_PI * value);
  }
  return sum;
};

function createFilledParameters(dimension: number, value: number): Float64Array {
  const parameters = new Float64Array(dimension);
  parameters.fill(value);
  return parameters;
}

describe('CMA-ES', () => {
  it('should be deterministic with the same seed', () => {
    const initialParameters = new Float64Array([10.0, -7.0, 3.0]);
    const options = {
      maxIterations: DETERMINISM_MAX_ITERATIONS,
      initialStepSize: DETERMINISM_INITIAL_STEP_SIZE,
      populationSize: DETERMINISM_POPULATION_SIZE,
      randomSeed: DETERMINISM_SEED,
      functionTolerance: 1e-14,
      parameterTolerance: 1e-12
    };

    const resultA = cmaEs(initialParameters, sphereCost, options);
    const resultB = cmaEs(initialParameters, sphereCost, options);

    expect(resultA.finalCost).toBe(resultB.finalCost);
    expect(Array.from(resultA.finalParameters)).toEqual(Array.from(resultB.finalParameters));
    expect(resultA.functionEvaluations).toBe(resultB.functionEvaluations);
  });

  it('should reduce Sphere function value substantially', () => {
    const initialParameters = createFilledParameters(SPHERE_DIMENSION, 10.0);
    const result = cmaEs(initialParameters, sphereCost, {
      maxIterations: SPHERE_MAX_ITERATIONS,
      initialStepSize: SPHERE_INITIAL_STEP_SIZE,
      randomSeed: 42,
      populationSize: SPHERE_POPULATION_SIZE,
      targetCost: SPHERE_TARGET_COST
    });

    expect(result.finalCost).toBeLessThan(SPHERE_EXPECTED_MAX_FINAL_COST);
    expect(result.finalStepSize).toBeGreaterThan(0.0);
  });

  it('should stop when maxFunctionEvaluations is reached', () => {
    const initialParameters = new Float64Array([5.0, -5.0]);
    const result = cmaEs(initialParameters, sphereCost, {
      maxIterations: 1000,
      initialStepSize: 1.0,
      randomSeed: 7,
      populationSize: BUDGET_POPULATION_SIZE,
      maxFunctionEvaluations: BUDGET_MAX_FUNCTION_EVALUATIONS
    });

    expect(result.converged).toBe(false);
    expect(result.functionEvaluations).toBeGreaterThanOrEqual(BUDGET_MAX_FUNCTION_EVALUATIONS);
  });

  it('should expose profiling timings when enabled', () => {
    const initialParameters = new Float64Array([4.0, -3.0, 2.0]);
    const result = cmaEs(initialParameters, sphereCost, {
      maxIterations: 40,
      initialStepSize: 1.5,
      populationSize: 12,
      randomSeed: DETERMINISM_SEED,
      profiling: true
    });

    expect(result.profiling).toBeDefined();
    expect(result.profiling!.totalMs).toBeGreaterThan(0);
    expect(result.profiling!.costMs).toBeGreaterThan(0);
    expect(result.profiling!.choleskyMs).toBeGreaterThanOrEqual(0);
    expect(result.profiling!.samplingMs).toBeGreaterThanOrEqual(0);
    expect(result.profiling!.updateMs).toBeGreaterThanOrEqual(0);
  });

  it('should improve Rastrigin with IPOP restart strategy', () => {
    const initialParameters = createFilledParameters(IPOP_DIMENSION, IPOP_INITIAL_VALUE);
    const initialCost = rastriginCost(initialParameters);

    const result = cmaEs(initialParameters, rastriginCost, {
      maxIterations: IPOP_MAX_ITERATIONS,
      populationSize: IPOP_POPULATION_SIZE,
      initialStepSize: IPOP_INITIAL_STEP_SIZE,
      randomSeed: DETERMINISM_SEED,
      targetCost: IPOP_TARGET_COST,
      restartStrategy: 'ipop'
    });

    expect(result.finalCost).toBeLessThan(IPOP_EXPECTED_MAX_FINAL_COST);
    expect(result.finalCost).toBeLessThan(initialCost);
  });
});
