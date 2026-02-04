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
export { backtrackingLineSearch, strongWolfeLineSearch } from './core/lineSearch.js';
export { gaussNewton } from './core/gaussNewton.js';
export { levenbergMarquardt } from './core/levenbergMarquardt.js';
export { bfgs } from './core/bfgs.js';
export { lbfgs } from './core/lbfgs.js';
export { cmaEs } from './core/cmaEs.js';
export { adjointGradientDescent } from './core/adjointGradientDescent.js';
export { constrainedGaussNewton } from './core/constrainedGaussNewton.js';
export { constrainedLevenbergMarquardt } from './core/constrainedLevenbergMarquardt.js';

// Numerical differentiation utilities
export {
  finiteDiffGradient,
  finiteDiffJacobian,
  finiteDiffPartialP,
  finiteDiffPartialX,
  finiteDiffConstraintPartialP,
  finiteDiffConstraintPartialX,
  finiteDiffResidualPartialP,
  finiteDiffResidualPartialX
} from './core/finiteDiff.js';
export { createFiniteDiffGradient, createFiniteDiffJacobian } from './core/createGradientFunction.js';

// Type definitions
export type {
  ResidualFn,
  JacobianFn,
  CostFn,
  GradientFn,
  ConstraintFn,
  ConstrainedCostFn,
  ConstrainedResidualFn,
  CommonOptimizationOptions,
  GradientDescentOptions,
  LineSearchOptions,
  StrongWolfeLineSearchOptions,
  NumericalDifferentiationOptions,
  GaussNewtonOptions,
  LevenbergMarquardtOptions,
  BfgsOptions,
  LbfgsOptions,
  CmaEsOptions,
  AdjointGradientDescentOptions,
  ConstrainedGaussNewtonOptions,
  ConstrainedLevenbergMarquardtOptions,
  OptimizationResult,
  LevenbergMarquardtResult,
  GradientDescentResult,
  CmaEsResult,
  AdjointGradientDescentResult,
  ConstrainedGaussNewtonResult,
  ConstrainedLevenbergMarquardtResult
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

// Result formatting utilities
export {
  formatOptimizationResult,
  formatGradientDescentResult,
  formatLevenbergMarquardtResult,
  formatCmaEsResult,
  formatConstrainedGaussNewtonResult,
  formatConstrainedLevenbergMarquardtResult,
  formatAdjointGradientDescentResult,
  formatResult,
  printOptimizationResult,
  printGradientDescentResult,
  printLevenbergMarquardtResult,
  printCmaEsResult,
  printConstrainedGaussNewtonResult,
  printConstrainedLevenbergMarquardtResult,
  printAdjointGradientDescentResult,
  printResult,
  type ResultFormatterOptions
} from './utils/resultFormatter.js';

