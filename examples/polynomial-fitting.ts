/**
 * Example: Polynomial Fitting with Gauss-Newton Method
 * 
 * This example demonstrates fitting a quadratic polynomial
 * to data points using the Gauss-Newton method.
 * 
 * Model: y = a*x^2 + b*x + c
 * Parameters: [a, b, c]
 */

import { gaussNewton } from '../src/index';
import type { ResidualFn } from '../src/core/types';

// Sample data points (quadratic relationship with noise)
const xData = new Float64Array([-2, -1, 0, 1, 2, 3, 4]);
const yData = new Float64Array([4.1, 0.9, -0.1, 1.1, 3.9, 8.9, 15.8]);

// True parameters: a = 1, b = 0, c = 0 (approximately y = x^2)
const trueParams = { a: 1.0, b: 0.0, c: 0.0 };

// Define residual function: r_i = a*x_i^2 + b*x_i + c - y_i
const residualFunction: ResidualFn = (params: Float64Array) => {
  const a = params[0];
  const b = params[1];
  const c = params[2];
  const residuals = new Float64Array(xData.length);
  
  for (let i = 0; i < xData.length; i++) {
    const predicted = a * xData[i] * xData[i] + b * xData[i] + c;
    residuals[i] = predicted - yData[i];
  }
  
  return residuals;
};

console.log('=== Polynomial Fitting: Quadratic Function ===\n');
console.log('Model: y = a*x^2 + b*x + c');
console.log('True parameters (approximately):');
console.log(`  a = ${trueParams.a}`);
console.log(`  b = ${trueParams.b}`);
console.log(`  c = ${trueParams.c}\n`);

console.log('Data points:');
for (let i = 0; i < xData.length; i++) {
  console.log(`  (${xData[i]}, ${yData[i]})`);
}
console.log('');

// Run optimization with Gauss-Newton method
const initialParameters = new Float64Array([0.5, 0.5, 0.5]);
console.log('Initial guess:', Array.from(initialParameters).map(x => x.toFixed(4)));
console.log('\nStarting Gauss-Newton optimization...\n');

const startTime = performance.now();
const result = gaussNewton(initialParameters, residualFunction, {
  useNumericJacobian: true,
  maxIterations: 100,
  tolGradient: 1e-6,
  verbose: false,
  onIteration: (iteration, cost) => {
    if (iteration % 10 === 0 || iteration < 5) {
      console.log(`Iteration ${iteration}: cost = ${cost.toFixed(8)}`);
    }
  }
});

console.log('\n=== Optimization Results ===');
console.log('Fitted parameters:');
console.log(`  a = ${result.parameters[0].toFixed(6)}`);
console.log(`  b = ${result.parameters[1].toFixed(6)}`);
console.log(`  c = ${result.parameters[2].toFixed(6)}`);
const endTime = performance.now();
const elapsedTime = endTime - startTime;

console.log('\nFinal residual norm:', Math.sqrt(result.finalCost).toFixed(8));
console.log('Converged:', result.converged);
console.log('Iterations:', result.iterations);
console.log(`Execution time: ${elapsedTime.toFixed(2)} ms (${(elapsedTime / 1000).toFixed(3)} seconds)`);
console.log(`Time per iteration: ${(elapsedTime / result.iterations).toFixed(3)} ms`);

// Show predictions vs actual
console.log('\n=== Predictions vs Actual Data ===');
for (let i = 0; i < xData.length; i++) {
  const predicted = result.parameters[0] * xData[i] * xData[i] + 
                    result.parameters[1] * xData[i] + 
                    result.parameters[2];
  const actual = yData[i];
  const error = Math.abs(predicted - actual);
  console.log(`x=${xData[i].toString().padStart(3)}: predicted=${predicted.toFixed(4)}, actual=${actual.toFixed(4)}, error=${error.toFixed(4)}`);
}

// Calculate R-squared
let ssRes = 0;
let ssTot = 0;
const yMean = yData.reduce((a, b) => a + b, 0) / yData.length;
for (let i = 0; i < xData.length; i++) {
  const predicted = result.parameters[0] * xData[i] * xData[i] + 
                    result.parameters[1] * xData[i] + 
                    result.parameters[2];
  ssRes += (predicted - yData[i]) ** 2;
  ssTot += (yData[i] - yMean) ** 2;
}
const rSquared = 1 - (ssRes / ssTot);
console.log(`\nR-squared: ${rSquared.toFixed(6)}`);

