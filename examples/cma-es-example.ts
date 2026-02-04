/**
 * Example: CMA-ES Optimization (Black-box, Derivative-free)
 *
 * This example demonstrates how to use CMA-ES to minimize a simple sphere function:
 *   f(x) = ||x||^2
 *
 * Minimum is at x = 0.
 *
 * Notes:
 * - CMA-ES is stochastic. Use `randomSeed` for reproducible runs.
 * - `initialStepSize` (sigma0) is important for performance.
 */

import { cmaEs, printCmaEsResult } from '../src/index';
import type { CostFn } from '../src/core/types';

const sphereCost: CostFn = (parameters: Float64Array) => {
  let sum = 0.0;
  for (let index = 0; index < parameters.length; index++) {
    const value = parameters[index];
    sum += value * value;
  }
  return sum;
};

const initialParameters = new Float64Array([10.0, -7.0, 3.0, 5.0]);

console.log('Starting CMA-ES optimization...');
console.log('Initial parameters:', Array.from(initialParameters));

const result = cmaEs(initialParameters, sphereCost, {
  maxIterations: 200,
  populationSize: 20,
  initialStepSize: 2.0,
  randomSeed: 123456,
  targetCost: 1e-10,
  verbose: true,
  onIteration: (iteration, cost) => {
    const LOG_INTERVAL = 10;
    if (iteration % LOG_INTERVAL === 0) {
      console.log(`Iteration ${iteration}: best cost = ${cost.toExponential(3)}`);
    }
  }
});

printCmaEsResult(result);

