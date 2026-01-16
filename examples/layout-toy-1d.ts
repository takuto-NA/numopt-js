/**
 * Example: 1D Layout Optimization
 * 
 * This example demonstrates optimizing the positions of boxes in a 1D layout
 * to minimize overlap while maintaining desired spacing.
 * 
 * Problem: Arrange N boxes with given widths to minimize overlap
 * Residual: r_i = desired_distance - actual_distance
 */

import { levenbergMarquardt, printLevenbergMarquardtResult } from '../src/index';
import type { ResidualFn } from '../src/core/types';

// Box widths
const boxWidths = new Float64Array([2.0, 3.0, 2.5, 1.5]);
const desiredSpacing = 1.0; // Desired spacing between boxes

// Define residual function
// Parameters: [x0, x1, x2, x3] (positions of left edges of boxes)
// Residuals: spacing violations and boundary constraints
const residualFunction: ResidualFn = (params: Float64Array) => {
  const boxCount = boxWidths.length;
  const residualCount = boxCount - 1 + 2; // spacing residuals + boundary constraints
  const residuals = new Float64Array(residualCount);
  
  let residualIndex = 0;
  
  // Spacing constraints: desired spacing between boxes
  for (let i = 0; i < boxCount - 1; i++) {
    const rightEdgeOfBoxI = params[i] + boxWidths[i];
    const leftEdgeOfBoxI1 = params[i + 1];
    const actualSpacing = leftEdgeOfBoxI1 - rightEdgeOfBoxI;
    residuals[residualIndex++] = desiredSpacing - actualSpacing;
  }
  
  // Boundary constraint: first box starts at 0
  residuals[residualIndex++] = params[0];
  
  // Boundary constraint: last box ends at reasonable position (soft constraint)
  const lastBoxRightEdge = params[boxCount - 1] + boxWidths[boxCount - 1];
  const totalWidth = boxWidths.reduce((sum, width) => sum + width, 0) + desiredSpacing * (boxCount - 1);
  residuals[residualIndex++] = lastBoxRightEdge - totalWidth;
  
  return residuals;
};

// Run optimization
const initialParameters = new Float64Array([0.0, 3.0, 7.0, 10.0]);

console.log('1D Layout Optimization Example');
console.log('Box widths:', Array.from(boxWidths));
console.log('Desired spacing:', desiredSpacing);
console.log('Initial positions:', Array.from(initialParameters));
console.log('\nStarting optimization...\n');

const result = levenbergMarquardt(initialParameters, residualFunction, {
  useNumericJacobian: true,
  maxIterations: 100,
  tolGradient: 1e-6,
  verbose: true,
  onIteration: (iteration, cost, params) => {
    if (iteration % 10 === 0) {
      console.log(`Iteration ${iteration}: cost = ${cost.toFixed(6)}`);
    }
  }
});

printLevenbergMarquardtResult(result);

// Show layout
console.log('\nLayout visualization:');
for (let i = 0; i < boxWidths.length; i++) {
  const position = result.finalParameters[i];
  const width = boxWidths[i];
  const bar = '█'.repeat(Math.max(1, Math.round(width)));
  console.log(`Box ${i}: position=${position.toFixed(2)}, width=${width.toFixed(2)} ${bar}`);
}

