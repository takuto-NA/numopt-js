/**
 * Example: Nonlinear Curve Fitting with Levenberg-Marquardt
 * 
 * This example demonstrates fitting an exponential decay function
 * to noisy data points using the Levenberg-Marquardt algorithm.
 * 
 * Model: y = A * exp(-lambda * x) + B
 * Parameters: [A, lambda, B]
 */

import { levenbergMarquardt, printLevenbergMarquardtResult } from '../src/index';
import type { ResidualFn } from '../src/core/types';

// Generate synthetic data with noise
// True parameters: A = 5.0, lambda = 0.3, B = 1.0
const trueParams = { A: 5.0, lambda: 0.3, B: 1.0 };

// Generate x values
const xData = new Float64Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const yData = new Float64Array(xData.length);

// Generate noisy data
for (let i = 0; i < xData.length; i++) {
  const x = xData[i];
  const trueY = trueParams.A * Math.exp(-trueParams.lambda * x) + trueParams.B;
  // Add Gaussian noise with std dev 0.1
  const noise = (Math.random() - 0.5) * 0.2;
  yData[i] = trueY + noise;
}

// Define residual function for exponential decay: r_i = A*exp(-lambda*x_i) + B - y_i
const residualFunction: ResidualFn = (params: Float64Array) => {
  const A = params[0];
  const lambda = params[1];
  const B = params[2];
  const residuals = new Float64Array(xData.length);
  
  for (let i = 0; i < xData.length; i++) {
    const predicted = A * Math.exp(-lambda * xData[i]) + B;
    residuals[i] = predicted - yData[i];
  }
  
  return residuals;
};

console.log('=== Nonlinear Curve Fitting: Exponential Decay ===\n');
console.log('Model: y = A * exp(-lambda * x) + B');
console.log('True parameters:');
console.log(`  A = ${trueParams.A}`);
console.log(`  lambda = ${trueParams.lambda}`);
console.log(`  B = ${trueParams.B}\n`);

console.log('Data points:');
for (let i = 0; i < Math.min(5, xData.length); i++) {
  console.log(`  (${xData[i]}, ${yData[i].toFixed(4)})`);
}
if (xData.length > 5) {
  console.log(`  ... and ${xData.length - 5} more points\n`);
}

// Run optimization
const initialParameters = new Float64Array([3.0, 0.5, 0.5]);
console.log('Initial guess:', Array.from(initialParameters).map(x => x.toFixed(4)));
console.log('\nStarting Levenberg-Marquardt optimization...\n');

const startTime = performance.now();
const result = levenbergMarquardt(initialParameters, residualFunction, {
  useNumericJacobian: true,
  maxIterations: 200,
  tolGradient: 1e-6,
  tolCost: 1e-8,
  verbose: false,
  onIteration: (iteration, cost) => {
    if (iteration % 20 === 0 || iteration < 5) {
      console.log(`Iteration ${iteration}: cost = ${cost.toFixed(8)}`);
    }
  }
});

const endTime = performance.now();
const elapsedTime = endTime - startTime;

printLevenbergMarquardtResult(result, {
  showExecutionTime: true,
  elapsedTimeMs: elapsedTime
});

console.log('\nParameter comparison:');
console.log(`  A = ${result.parameters[0].toFixed(6)} (true: ${trueParams.A}), error: ${Math.abs(result.parameters[0] - trueParams.A).toFixed(6)}`);
console.log(`  lambda = ${result.parameters[1].toFixed(6)} (true: ${trueParams.lambda}), error: ${Math.abs(result.parameters[1] - trueParams.lambda).toFixed(6)}`);
console.log(`  B = ${result.parameters[2].toFixed(6)} (true: ${trueParams.B}), error: ${Math.abs(result.parameters[2] - trueParams.B).toFixed(6)}`);

// Show predictions vs actual
console.log('\n=== Predictions vs Actual Data ===');
for (let i = 0; i < xData.length; i++) {
  const predicted = result.parameters[0] * Math.exp(-result.parameters[1] * xData[i]) + result.parameters[2];
  const actual = yData[i];
  const error = Math.abs(predicted - actual);
  console.log(`x=${xData[i].toFixed(1).padStart(4)}: predicted=${predicted.toFixed(4)}, actual=${actual.toFixed(4)}, error=${error.toFixed(4)}`);
}

