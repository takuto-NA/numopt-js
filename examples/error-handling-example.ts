/**
 * Example: Error Handling and Robust Optimization
 * 
 * This example demonstrates:
 * 1. How to handle errors gracefully
 * 2. How to check convergence status
 * 3. How to handle non-convergence cases
 * 4. How to use verbose logging for debugging
 */

import { levenbergMarquardt, gaussNewton } from '../src/index';
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

console.log('=== Error Handling Example ===\n');

// Example 1: Handling Jacobian errors
console.log('1. Handling Jacobian computation errors:');
console.log('   Attempting optimization with useNumericJacobian disabled...\n');

try {
  // This will fail because we're not providing a Jacobian and numerical Jacobian is disabled
  const result = levenbergMarquardt(
    new Float64Array([0, 0]),
    residualFunction,
    { useNumericJacobian: false } // Explicitly disable numerical Jacobian
  );
  console.log('   Success:', result.converged);
} catch (error) {
  console.log('   Caught error:', (error as Error).message);
  console.log('   Solution: Enable useNumericJacobian or provide a Jacobian function\n');
}

// Example 2: Checking convergence status
console.log('2. Checking convergence status:');
const initialParams = new Float64Array([0.0, 0.0]);

const result = levenbergMarquardt(initialParams, residualFunction, {
  useNumericJacobian: true,
  maxIterations: 10, // Very low to demonstrate non-convergence
  tolGradient: 1e-10, // Very strict tolerance
  verbose: false
});

if (result.converged) {
  console.log('   Optimization converged successfully!');
  console.log(`   Final residual norm: ${result.finalResidualNorm.toFixed(6)}`);
} else {
  console.log('   Optimization did not converge.');
  console.log(`   Final residual norm: ${result.finalResidualNorm.toFixed(6)}`);
  console.log(`   Iterations: ${result.iterations}`);
  console.log('   Suggestions:');
  console.log('     - Increase maxIterations');
  console.log('     - Relax tolerance values');
  console.log('     - Try different initial parameters');
  console.log('     - Check if the problem is well-posed\n');
}

// Example 3: Handling singular matrix errors (Gauss-Newton)
console.log('3. Handling singular matrix errors (Gauss-Newton):');
console.log('   Gauss-Newton may fail with singular matrices...\n');

try {
  const gnResult = gaussNewton(initialParams, residualFunction, {
    useNumericJacobian: true,
    maxIterations: 100
  });
  
  if (gnResult.converged) {
    console.log('   Gauss-Newton converged successfully!');
    console.log(`   Final residual norm: ${gnResult.finalResidualNorm?.toFixed(6) ?? 'N/A'}`);
  } else {
    console.log('   Gauss-Newton did not converge.');
    console.log('   Consider using Levenberg-Marquardt instead (handles singular matrices better)');
  }
} catch (error) {
  console.log('   Caught error:', (error as Error).message);
  console.log('   Solution: Use Levenberg-Marquardt algorithm instead\n');
}

// Example 4: Using verbose logging for debugging
console.log('4. Using verbose logging for debugging:');
console.log('   Running with verbose: true...\n');

const debugResult = levenbergMarquardt(initialParams, residualFunction, {
  useNumericJacobian: true,
  maxIterations: 5,
  verbose: true, // Enable verbose logging
  onIteration: (iteration, cost, params) => {
    console.log(`   Iteration ${iteration}: cost = ${cost.toFixed(6)}, params = [${params[0].toFixed(4)}, ${params[1].toFixed(4)}]`);
  }
});

console.log(`\n   Final result: converged = ${debugResult.converged}`);

// Example 5: Robust optimization with fallback strategies
console.log('\n5. Robust optimization with fallback strategies:');

function robustOptimize(initialParams: Float64Array, residualFn: ResidualFn) {
  // Try Levenberg-Marquardt first (most robust)
  let result = levenbergMarquardt(initialParams, residualFn, {
    useNumericJacobian: true,
    maxIterations: 100,
    tolGradient: 1e-6
  });
  
  if (result.converged) {
    console.log('   Levenberg-Marquardt succeeded!');
    return result;
  }
  
  // If LM failed, try with relaxed tolerances
  console.log('   Levenberg-Marquardt did not converge, trying relaxed tolerances...');
  result = levenbergMarquardt(initialParams, residualFn, {
    useNumericJacobian: true,
    maxIterations: 200,
    tolGradient: 1e-4, // Relaxed tolerance
    tolStep: 1e-4,
    tolResidual: 1e-4
  });
  
  if (result.converged) {
    console.log('   Succeeded with relaxed tolerances!');
  } else {
    console.log('   Still did not converge. Consider:');
    console.log('     - Checking initial parameters');
    console.log('     - Verifying residual function');
    console.log('     - Increasing maxIterations further');
  }
  
  return result;
}

const robustResult = robustOptimize(initialParams, residualFunction);
console.log(`   Final residual norm: ${robustResult.finalResidualNorm.toFixed(6)}`);

