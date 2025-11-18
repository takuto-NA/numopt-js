/**
 * Example: Line Search Demonstration
 * 
 * This example shows how line search finds optimal step sizes
 * for gradient descent optimization.
 */

import { backtrackingLineSearch } from '../src/index';
import type { CostFn, GradientFn } from '../src/core/types';

// Define cost function: f(x) = x^2
const costFunction: CostFn = (params: Float64Array) => {
  return params[0] * params[0];
};

// Define gradient function: f'(x) = 2x
const gradientFunction: GradientFn = (params: Float64Array) => {
  return new Float64Array([2 * params[0]]);
};

// Test line search from different starting points
const testPoints = [5.0, 3.0, 1.0];

console.log('Line Search Examples:\n');

for (const startPoint of testPoints) {
  const currentParams = new Float64Array([startPoint]);
  const searchDirection = new Float64Array([-1.0]); // Negative gradient direction
  
  const stepSize = backtrackingLineSearch(
    costFunction,
    gradientFunction,
    currentParams,
    searchDirection,
    {
      initialStepSize: 1.0,
      contractionFactor: 0.5,
      armijoParameter: 0.1
    }
  );
  
  const newParams = new Float64Array([currentParams[0] + stepSize * searchDirection[0]]);
  const oldCost = costFunction(currentParams);
  const newCost = costFunction(newParams);
  
  console.log(`Starting point: ${startPoint}`);
  console.log(`  Step size found: ${stepSize.toFixed(6)}`);
  console.log(`  Old cost: ${oldCost.toFixed(6)}`);
  console.log(`  New cost: ${newCost.toFixed(6)}`);
  console.log(`  Cost reduction: ${(oldCost - newCost).toFixed(6)}\n`);
}

