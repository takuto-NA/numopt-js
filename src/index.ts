/**
 * Public package entry: re-exports algorithms, helpers, and shared types.
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
export { adjointBfgs } from './core/adjointBfgs.js';
export { adjointLbfgs } from './core/adjointLbfgs.js';
export { adjointNewton } from './core/adjointNewton.js';
export { adjointNewtonExact } from './core/adjointNewtonExact.js';
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
export { computeEffectiveJacobian, type EffectiveJacobianOptions } from './core/effectiveJacobian.js';

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
  AdjointDerivativeOptions,
  AdjointGradientDescentOptions,
  AdjointBfgsOptions,
  AdjointLbfgsOptions,
  AdjointLbfgsResult,
  AdjointLbfgsStopReason,
  AdjointNewtonOptions,
  AdjointNewtonCommonOptions,
  AdjointNewtonExactOptions,
  AdjointNewtonExactResult,
  AdjointNewtonResult,
  AdjointNewtonStopReason,
  ConstrainedGaussNewtonOptions,
  ConstrainedLevenbergMarquardtOptions,
  OptimizationResult,
  LevenbergMarquardtResult,
  GradientDescentResult,
  CmaEsResult,
  AdjointGradientDescentResult,
  AdjointBfgsResult,
  AdjointBfgsStopReason,
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
  formatAdjointBfgsResult,
  formatResult,
  printOptimizationResult,
  printGradientDescentResult,
  printLevenbergMarquardtResult,
  printCmaEsResult,
  printConstrainedGaussNewtonResult,
  printConstrainedLevenbergMarquardtResult,
  printAdjointGradientDescentResult,
  printAdjointBfgsResult,
  printResult,
  type ResultFormatterOptions
} from './utils/resultFormatter.js';

