/**
 * Example: Advanced Usage Patterns
 * 
 * This example demonstrates:
 * 1. Providing custom analytical Jacobian
 * 2. Comparing different algorithms
 * 3. Fine-tuning parameters
 * 4. Using progress callbacks for monitoring
 * 5. Converting between array types
 */

import { levenbergMarquardt, gaussNewton, gradientDescent, finiteDiffGradient, printLevenbergMarquardtResult, printOptimizationResult, printGradientDescentResult } from '../src/index';
import { Matrix } from 'ml-matrix';
import type { ResidualFn, CostFn, GradientFn, JacobianFn } from '../src/core/types';

console.log('=== Advanced Usage Examples ===\n');

// Example 1: Custom Analytical Jacobian
console.log('1. Using Custom Analytical Jacobian:');
console.log('   (More accurate and faster than numerical Jacobian)\n');

const xData = new Float64Array([1, 2, 3, 4, 5]);
const yData = new Float64Array([2.1, 3.9, 6.1, 8.0, 9.9]);

const residualFunction: ResidualFn = (params: Float64Array) => {
  const a = params[0];
  const b = params[1];
  const residuals = new Float64Array(xData.length);
  
  for (let i = 0; i < xData.length; i++) {
    residuals[i] = a * xData[i] + b - yData[i];
  }
  
  return residuals;
};

// Analytical Jacobian: J[i][0] = x_i, J[i][1] = 1
const analyticalJacobian: JacobianFn = (params: Float64Array) => {
  const jacobianData: number[][] = [];
  for (let i = 0; i < xData.length; i++) {
    jacobianData.push([xData[i], 1.0]);
  }
  return new Matrix(jacobianData);
};

const initialParams = new Float64Array([0.0, 0.0]);

const resultWithAnalytical = levenbergMarquardt(
  initialParams,
  residualFunction,
  {
    jacobian: analyticalJacobian, // Provide analytical Jacobian
    maxIterations: 100,
    tolGradient: 1e-6
  }
);

console.log(`   Result with analytical Jacobian:`);
printLevenbergMarquardtResult(resultWithAnalytical, {
  showSectionHeaders: false
});
console.log('');

// Example 2: Comparing Algorithms
console.log('2. Comparing Different Algorithms:\n');

// Same problem solved with different algorithms
const compareAlgorithms = () => {
  const params = new Float64Array([0.0, 0.0]);
  
  console.log('   Levenberg-Marquardt:');
  const lmResult = levenbergMarquardt(params, residualFunction, {
    useNumericJacobian: true,
    maxIterations: 100,
    tolGradient: 1e-6
  });
  printLevenbergMarquardtResult(lmResult, {
    showSectionHeaders: false
  });
  console.log('');
  
  console.log('   Gauss-Newton:');
  const gnResult = gaussNewton(params, residualFunction, {
    useNumericJacobian: true,
    maxIterations: 100,
    tolerance: 1e-6
  });
  printOptimizationResult(gnResult, {
    showSectionHeaders: false
  });
  console.log('');
};

compareAlgorithms();

// Example 3: Fine-tuning Parameters
console.log('3. Fine-tuning Parameters:');
console.log('   Demonstrating the effect of different tolerance values...\n');

const toleranceValues = [1e-4, 1e-6, 1e-8];

for (const tol of toleranceValues) {
  const result = levenbergMarquardt(initialParams, residualFunction, {
    useNumericJacobian: true,
    maxIterations: 100,
    tolGradient: tol,
    tolStep: tol,
    tolResidual: tol
  });
  
  console.log(`   Tolerance ${tol}:`);
  printLevenbergMarquardtResult(result, {
    showSectionHeaders: false
  });
  console.log('');
}

// Example 4: Using Progress Callbacks
console.log('4. Using Progress Callbacks for Monitoring:\n');

let iterationCount = 0;
const costHistory: number[] = [];

const resultWithCallback = levenbergMarquardt(initialParams, residualFunction, {
  useNumericJacobian: true,
  maxIterations: 50,
  onIteration: (iteration, cost, params) => {
    iterationCount++;
    costHistory.push(cost);
    
    if (iteration % 10 === 0 || iteration < 5) {
      console.log(`   Iteration ${iteration}: cost = ${cost.toFixed(6)}, params = [${params[0].toFixed(4)}, ${params[1].toFixed(4)}]`);
    }
  }
});

console.log(`\n   Total iterations: ${iterationCount}`);
console.log(`   Cost reduction: ${costHistory[0].toFixed(6)} -> ${costHistory[costHistory.length - 1].toFixed(6)}`);
console.log(`   Improvement: ${((1 - costHistory[costHistory.length - 1] / costHistory[0]) * 100).toFixed(2)}%\n`);

// Example 5: Converting Array Types
console.log('5. Converting Between Array Types:\n');

// Regular array to Float64Array
const regularArray = [1.0, 2.0, 3.0];
const float64Array = new Float64Array(regularArray);
console.log(`   Regular array: [${regularArray.join(', ')}]`);
console.log(`   Float64Array: [${Array.from(float64Array).join(', ')}]\n`);

// Using with gradient descent (requires Float64Array)
const costFn: CostFn = (params: Float64Array) => {
  return params[0] * params[0] + params[1] * params[1];
};

const gradFn: GradientFn = (params: Float64Array) => {
  return new Float64Array([2 * params[0], 2 * params[1]]);
};

// Convert regular array to Float64Array for optimization
const regularInitial = [5.0, -3.0];
const float64Initial = new Float64Array(regularInitial);

const gdResult = gradientDescent(float64Initial, costFn, gradFn, {
  maxIterations: 100,
  tolerance: 1e-6
});

// Convert back to regular array if needed
const regularResult = Array.from(gdResult.parameters);
console.log(`   Optimization result: [${regularResult.map(x => x.toFixed(4)).join(', ')}]\n`);

// Example 6: Using Numerical Gradient with Gradient Descent
console.log('6. Using Numerical Gradient:\n');

const costFunctionForNumGrad: CostFn = (params: Float64Array) => {
  // Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
  const x = params[0];
  const y = params[1];
  return Math.pow(1 - x, 2) + 100 * Math.pow(y - x * x, 2);
};

const numGradParams = new Float64Array([-1.2, 1.0]);

// Use numerical gradient instead of analytical
const numericalGradient = finiteDiffGradient(numGradParams, costFunctionForNumGrad, {
  stepSize: 1e-6
});

console.log(`   Numerical gradient at [${numGradParams[0]}, ${numGradParams[1]}]:`);
console.log(`     [${numericalGradient[0].toFixed(6)}, ${numericalGradient[1].toFixed(6)}]\n`);

// Use numerical gradient in gradient descent
const gdWithNumGrad = gradientDescent(numGradParams, costFunctionForNumGrad, (params) => {
  return finiteDiffGradient(params, costFunctionForNumGrad, { stepSize: 1e-6 });
}, {
  maxIterations: 1000,
  tolerance: 1e-6,
  useLineSearch: true
});

console.log(`   Gradient descent with numerical gradient:`);
printGradientDescentResult(gdWithNumGrad, {
  showSectionHeaders: false
});
console.log(`     Converged: ${gdWithNumGrad.converged}`);

