/**
 * Example: Advanced Constrained Optimization with Adjoint Method
 * 
 * This example demonstrates solving more complex constrained optimization problems
 * using the adjoint gradient descent method.
 * 
 * Problem 1: 2D Constrained Optimization
 * Minimize: f(p, x) = (p₁ - 1)² + (p₂ - 2)² + x₁² + x₂²
 * Subject to: c₁(p, x) = p₁ + x₁ - 1 = 0
 *             c₂(p, x) = p₂ + x₂ - 2 = 0
 * 
 * Analytical solution: p = [1, 2], x = [0, 0], f = 0
 * 
 * Problem 2: Nonlinear Constraint (Circle Constraint)
 * Minimize: f(p, x) = p² + x²
 * Subject to: c(p, x) = p² + x² - 1 = 0
 * 
 * Analytical solution: p = 1/√2, x = 1/√2, f = 1
 * 
 * Problem 3: Residual-based Problem
 * Minimize: f(p, x) = 1/2 ||r(p, x)||² where r = [p - 0.5, x - 0.5]
 * Subject to: c(p, x) = p + x - 1 = 0
 * 
 * Analytical solution: p = 0.5, x = 0.5, f = 0
 */

import { adjointGradientDescent, printAdjointGradientDescentResult } from '../src/index';
import type { ConstrainedCostFn, ConstraintFn, ConstrainedResidualFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

console.log('=== Advanced Constrained Optimization: Adjoint Method ===\n');

// ============================================================================
// Problem 1: 2D Constrained Optimization
// ============================================================================
console.log('Problem 1: 2D Constrained Optimization');
console.log('  Minimize: f(p, x) = (p₁ - 1)² + (p₂ - 2)² + x₁² + x₂²');
console.log('  Subject to: c₁(p, x) = p₁ + x₁ - 1 = 0');
console.log('             c₂(p, x) = p₂ + x₂ - 2 = 0');
console.log('  Analytical solution: p = [1, 2], x = [0, 0], f = 0\n');

const cost2D: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
  return Math.pow(p[0] - 1, 2) + Math.pow(p[1] - 2, 2) + x[0] * x[0] + x[1] * x[1];
};

const constraint2D: ConstraintFn = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([
    p[0] + x[0] - 1.0,
    p[1] + x[1] - 2.0
  ]);
};

const initialP2D = new Float64Array([3.0, 4.0]);
const initialX2D = new Float64Array([-2.0, -2.0]); // Satisfies constraints

console.log('Initial values:');
console.log(`  p₀ = [${initialP2D[0]}, ${initialP2D[1]}]`);
console.log(`  x₀ = [${initialX2D[0]}, ${initialX2D[1]}]`);
const initialConstraint2D = constraint2D(initialP2D, initialX2D);
console.log(`  c(p₀, x₀) = [${initialConstraint2D[0]}, ${initialConstraint2D[1]}]\n`);

console.log('Solving Problem 1...\n');

const result2D = adjointGradientDescent(
  initialP2D,
  initialX2D,
  cost2D,
  constraint2D,
  {
    maxIterations: 200,
    tolerance: 1e-6,
    useLineSearch: true,
    logLevel: 'INFO' // Use INFO to see convergence messages
  }
);

const finalConstraint2D = constraint2D(result2D.parameters, result2D.finalStates);
printAdjointGradientDescentResult(result2D, {
  showSectionHeaders: false
});
console.log(`  ||c(p, x)|| = ${vectorNorm(finalConstraint2D).toFixed(8)}`);
console.log(`  Analytical: p = [1, 2], x = [0, 0], f = 0\n`);

// ============================================================================
// Problem 2: Nonlinear Constraint with Complex Objective
// ============================================================================
console.log('\n' + '='.repeat(70) + '\n');
console.log('Problem 2: Nonlinear Constraint with Complex Objective');
console.log('  Minimize: f(p, x) = (p - 1)² + (x - 1)²');
console.log('  Subject to: c(p, x) = p² + x² - 2 = 0');
console.log('  Analytical solution: p = 1, x = 1, f = 0\n');
console.log('  Note: This is more challenging because the constraint is nonlinear');
console.log('        and the initial value may not satisfy the constraint.\n');

const costCircle: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
  return Math.pow(p[0] - 1, 2) + Math.pow(x[0] - 1, 2);
};

const constraintCircle: ConstraintFn = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([p[0] * p[0] + x[0] * x[0] - 2.0]);
};

// Start from a point that doesn't satisfy the constraint
// This makes the problem more challenging
const initialPCircle = new Float64Array([1.5]);
const initialXCircle = new Float64Array([0.5]); // Doesn't satisfy: 1.5² + 0.5² = 2.5 ≠ 2

console.log('Initial values:');
console.log(`  p₀ = ${initialPCircle[0]}`);
console.log(`  x₀ = ${initialXCircle[0]}`);
const initialConstraintCircle = constraintCircle(initialPCircle, initialXCircle);
console.log(`  c(p₀, x₀) = ${initialConstraintCircle[0]} (should be ≈ 0)\n`);

console.log('Solving Problem 2...\n');

const resultCircle = adjointGradientDescent(
  initialPCircle,
  initialXCircle,
  costCircle,
  constraintCircle,
  {
    maxIterations: 500,
    tolerance: 1e-4, // Slightly relaxed tolerance for this challenging problem
    useLineSearch: true,
    constraintTolerance: 1e-3, // Allow some constraint violation tolerance
    logLevel: 'DEBUG' // Use DEBUG to see detailed iteration information
  }
);

const finalConstraintCircle = constraintCircle(resultCircle.parameters, resultCircle.finalStates);
printAdjointGradientDescentResult(resultCircle, {
  showSectionHeaders: false
});
console.log(`  ||c(p, x)|| = ${vectorNorm(finalConstraintCircle).toFixed(8)}`);
console.log(`  Analytical: p = 1.0, x = 1.0, f = 0.0\n`);

// ============================================================================
// Problem 3: Residual-based Problem
// ============================================================================
console.log('\n' + '='.repeat(70) + '\n');
console.log('Problem 3: Residual-based Problem');
console.log('  Minimize: f(p, x) = 1/2 ||r(p, x)||² where r = [p - 0.5, x - 0.5]');
console.log('  Subject to: c(p, x) = p + x - 1 = 0');
console.log('  Analytical solution: p = 0.5, x = 0.5, f = 0\n');

const residualFn: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([
    p[0] - 0.5,
    x[0] - 0.5
  ]);
};

const constraintResidual: ConstraintFn = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([p[0] + x[0] - 1.0]);
};

const initialPResidual = new Float64Array([2.0]);
const initialXResidual = new Float64Array([-1.0]); // Satisfies constraint

console.log('Initial values:');
console.log(`  p₀ = ${initialPResidual[0]}`);
console.log(`  x₀ = ${initialXResidual[0]}`);
const initialConstraintResidual = constraintResidual(initialPResidual, initialXResidual);
console.log(`  c(p₀, x₀) = ${initialConstraintResidual[0]}\n`);

console.log('Solving Problem 3...\n');

const resultResidual = adjointGradientDescent(
  initialPResidual,
  initialXResidual,
  residualFn,
  constraintResidual,
  {
    maxIterations: 200,
    tolerance: 1e-6,
    useLineSearch: true,
    logLevel: 'INFO'
  }
);

const finalConstraintResidual = constraintResidual(resultResidual.parameters, resultResidual.finalStates);
printAdjointGradientDescentResult(resultResidual, {
  showSectionHeaders: false
});
console.log(`  ||c(p, x)|| = ${vectorNorm(finalConstraintResidual).toFixed(8)}`);
console.log(`  Analytical: p = 0.5, x = 0.5, f = 0.0\n`);

// ============================================================================
// Summary
// ============================================================================
console.log('\n' + '='.repeat(70) + '\n');
console.log('Summary:');
console.log(`  Problem 1 (2D): ${result2D.converged ? '✅ Converged' : '❌ Failed'} in ${result2D.iterations} iterations`);
console.log(`  Problem 2 (Nonlinear): ${resultCircle.converged ? '✅ Converged' : '❌ Failed'} in ${resultCircle.iterations} iterations`);
console.log(`  Problem 3 (Residual): ${resultResidual.converged ? '✅ Converged' : '❌ Failed'} in ${resultResidual.iterations} iterations`);

