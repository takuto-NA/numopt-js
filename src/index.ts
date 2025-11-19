/**
 * This file is the main entry point for the numopt-js library.
 * 
 * Role in system:
 * - Exports all public API functions and types
 * - Provides clean, focused interface for users
 * - Single import point for the entire library
 * 
 * For first-time readers:
 * - Import everything you need from this file
 * - Check individual algorithm files for detailed documentation
 * - All algorithms follow consistent patterns
 */

// Core algorithms
export { gradientDescent } from './core/gradientDescent.js';
export { backtrackingLineSearch } from './core/lineSearch.js';
export { gaussNewton } from './core/gaussNewton.js';
export { levenbergMarquardt } from './core/levenbergMarquardt.js';

// Numerical differentiation utilities
export { finiteDiffGradient, finiteDiffJacobian } from './core/finiteDiff.js';

// Type definitions
export type {
  ResidualFn,
  JacobianFn,
  CostFn,
  GradientFn,
  CommonOptimizationOptions,
  GradientDescentOptions,
  LineSearchOptions,
  NumericalDifferentiationOptions,
  GaussNewtonOptions,
  LevenbergMarquardtOptions,
  OptimizationResult,
  LevenbergMarquardtResult,
  GradientDescentResult
} from './core/types.js';

// Utility functions (exported for advanced users)
export {
  float64ArrayToMatrix,
  matrixToFloat64Array,
  matrixToFloat64Array2D,
  vectorNorm,
  dotProduct,
  addVectors,
  subtractVectors,
  scaleVector
} from './utils/matrix.js';

