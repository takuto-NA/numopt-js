/**
 * Example: Curve Fitting with Levenberg-Marquardt
 * 
 * This example demonstrates fitting a linear model y = ax + b
 * to data points using the Levenberg-Marquardt algorithm.
 */

import { levenbergMarquardt, printLevenbergMarquardtResult } from '../src/index';
import type { ResidualFn } from '../src/core/types';

// Sample data points
const xData = new Float64Array([1, 2, 3, 4, 5]);
const yData = new Float64Array([2.1, 3.9, 6.1, 8.0, 9.9]);

// Define residual function for linear regression: r_i = ax_i + b - y_i
const residualFunction: ResidualFn = (params: Float64Array) => {
  const a = params[0]; // slope
  const b = params[1]; // intercept
  const residuals = new Float64Array(xData.length);
  
  for (let i = 0; i < xData.length; i++) {
    const predicted = a * xData[i] + b;
    residuals[i] = predicted - yData[i];
  }
  
  return residuals;
};

// Run optimization
const initialParameters = new Float64Array([0.0, 0.0]);

console.log('Curve Fitting Example: Linear Regression');
console.log('Data points:');
for (let i = 0; i < xData.length; i++) {
  console.log(`  (${xData[i]}, ${yData[i]})`);
}
console.log('\nStarting Levenberg-Marquardt optimization...\n');

const result = levenbergMarquardt(initialParameters, residualFunction, {
  useNumericJacobian: true,
  maxIterations: 100,
  tolGradient: 1e-6,
  verbose: true,
  onIteration: (iteration, cost, params) => {
    if (iteration % 10 === 0) {
      console.log(`Iteration ${iteration}: a = ${params[0].toFixed(4)}, b = ${params[1].toFixed(4)}, cost = ${cost.toFixed(6)}`);
    }
  }
});

printLevenbergMarquardtResult(result);
console.log(`\nFitted line: y = ${result.parameters[0].toFixed(4)}x + ${result.parameters[1].toFixed(4)}`);

// Show predictions
console.log('\nPredictions vs Actual:');
for (let i = 0; i < xData.length; i++) {
  const predicted = result.parameters[0] * xData[i] + result.parameters[1];
  const actual = yData[i];
  const error = Math.abs(predicted - actual);
  console.log(`  x=${xData[i]}: predicted=${predicted.toFixed(2)}, actual=${actual.toFixed(2)}, error=${error.toFixed(2)}`);
}

