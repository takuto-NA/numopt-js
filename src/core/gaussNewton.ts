/**
 * This file implements the Gauss-Newton method for solving nonlinear least squares problems.
 * 
 * Role in system:
 * - Phase 2 intermediate algorithm (builds on gradient descent concepts)
 * - Foundation for Levenberg-Marquardt method
 * - Specifically designed for nonlinear least squares problems
 * 
 * For first-time readers:
 * - Start with gaussNewton function
 * - Understand how it solves normal equations: (J^T J) δ = -J^T r
 * - This is a special case of Newton's method for least squares
 */

import { Matrix, solve, CholeskyDecomposition } from 'ml-matrix';
import type {
  ResidualFn,
  JacobianFn,
  GaussNewtonOptions,
  OptimizationResult
} from './types.js';
import { float64ArrayToMatrix, matrixToFloat64Array, vectorNorm, computeSumOfSquaredResiduals } from '../utils/matrix.js';
import { checkStepSizeConvergence, checkResidualConvergence, createConvergenceResult } from './convergence.js';
import { computeJacobianMatrix } from './jacobianComputation.js';
import { Logger } from './logger.js';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_USE_NUMERIC_JACOBIAN = true;
const DEFAULT_JACOBIAN_STEP = 1e-6;
const NEGATIVE_COEFFICIENT = -1.0; // Coefficient for negative right-hand side in normal equations: (J^T J) δ = -J^T r


/**
 * Performs Gauss-Newton optimization for nonlinear least squares problems.
 * 
 * Algorithm:
 * 1. Start with initial parameters
 * 2. Compute residual vector r and Jacobian matrix J
 * 3. Solve normal equations: (J^T J) δ = -J^T r
 * 4. Update parameters: x_new = x_old + δ
 * 5. Repeat until convergence
 * 
 * The Gauss-Newton method approximates the Hessian as J^T J, which is
 * exact for linear least squares and a good approximation for nonlinear cases.
 */
export function gaussNewton(
  initialParameters: Float64Array,
  residualFunction: ResidualFn,
  options: GaussNewtonOptions = {}
): OptimizationResult {
  const actualOptions: GaussNewtonOptions = options;
  const jacobianFunction: JacobianFn | undefined = actualOptions.jacobian;

  const maxIterations = actualOptions.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const tolerance = actualOptions.tolerance ?? DEFAULT_TOLERANCE;
  const useNumericJacobian = actualOptions.useNumericJacobian ?? DEFAULT_USE_NUMERIC_JACOBIAN;
  const jacobianStep = actualOptions.jacobianStep ?? DEFAULT_JACOBIAN_STEP;
  const onIteration = actualOptions.onIteration;
  const logger = new Logger(actualOptions.logLevel, actualOptions.verbose);

  let currentParameters = new Float64Array(initialParameters);

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    // Compute residual vector
    const residual = residualFunction(currentParameters);
    const residualNorm = vectorNorm(residual);
    const cost = computeSumOfSquaredResiduals(residualNorm);

    // Call progress callback if provided
    if (onIteration) {
      onIteration(iteration, cost, currentParameters);
    }

    // Compute Jacobian matrix
    // Early return: use analytical Jacobian if provided
    const jacobianMatrix: Matrix = computeJacobianMatrix(
      jacobianFunction,
      residualFunction,
      currentParameters,
      useNumericJacobian,
      jacobianStep,
      'gaussNewton'
    );

    // Compute J^T J and J^T r
    const jacobianTranspose = jacobianMatrix.transpose();
    const jtj = jacobianTranspose.mmul(jacobianMatrix);
    const residualMatrix = float64ArrayToMatrix(residual);
    const jtr = jacobianTranspose.mmul(residualMatrix);

    // Solve normal equations: (J^T J) δ = -J^T r
    // This gives us: δ = -(J^T J)^(-1) J^T r
    // Try Cholesky decomposition first for efficiency (if J^T J is positive definite)
    let step: Float64Array;
    try {
      const negativeJtr = jtr.mul(NEGATIVE_COEFFICIENT);
      let stepMatrix: Matrix;
      try {
        const cholesky = new CholeskyDecomposition(jtj);
        if (cholesky.isPositiveDefinite()) {
          // Use Cholesky decomposition for efficiency (about 2x faster than LU)
          stepMatrix = cholesky.solve(negativeJtr);
        } else {
          // J^T J is not positive definite, fallback to LU decomposition
          stepMatrix = solve(jtj, negativeJtr);
        }
      } catch (choleskyError) {
        // Cholesky decomposition failed (non-symmetric or other issues), fallback to LU
        stepMatrix = solve(jtj, negativeJtr);
      }
      step = matrixToFloat64Array(stepMatrix);
    } catch (error) {
      // Handle singular matrix (J^T J is not invertible)
      logger.warn('gaussNewton', iteration, 'Singular matrix encountered. Consider using Levenberg-Marquardt method for better robustness.', [
        { key: 'Cost:', value: cost },
        { key: 'Residual norm:', value: residualNorm }
      ]);
      const result = createConvergenceResult(currentParameters, iteration, false, cost, undefined);
      return { ...result, finalResidualNorm: residualNorm };
    }

    // Check convergence: step size is small enough
    const stepNorm = vectorNorm(step);
    if (checkStepSizeConvergence(stepNorm, tolerance, iteration)) {
      logger.info('gaussNewton', iteration, 'Converged', [
        { key: 'Cost:', value: cost },
        { key: 'Residual norm:', value: residualNorm },
        { key: 'Step size:', value: stepNorm }
      ]);
      const result = createConvergenceResult(currentParameters, iteration, true, cost, undefined);
      return { ...result, finalResidualNorm: residualNorm };
    }

    // Update parameters: x_new = x_old + δ
    const newParameters = new Float64Array(currentParameters.length);
    for (let i = 0; i < currentParameters.length; i++) {
      newParameters[i] = currentParameters[i] + step[i];
    }

    // Compute residual for new parameters
    const newResidual = residualFunction(newParameters);
    const newResidualNorm = vectorNorm(newResidual);
    const newCost = computeSumOfSquaredResiduals(newResidualNorm);

    // Check convergence: residual norm is small enough
    if (checkResidualConvergence(newResidualNorm, tolerance, iteration)) {
      logger.info('gaussNewton', iteration, 'Converged', [
        { key: 'Cost:', value: newCost },
        { key: 'Residual norm:', value: newResidualNorm }
      ]);
      const result = createConvergenceResult(newParameters, iteration, true, newCost, undefined);
      return { ...result, finalResidualNorm: newResidualNorm };
    }

    logger.debug('gaussNewton', iteration, 'Progress', [
      { key: 'Cost:', value: cost },
      { key: 'Residual norm:', value: residualNorm },
      { key: 'Step norm:', value: stepNorm }
    ]);

    currentParameters = newParameters;
  }

  // Maximum iterations reached
  const finalResidual = residualFunction(currentParameters);
  const finalResidualNorm = vectorNorm(finalResidual);
  const finalCost = computeSumOfSquaredResiduals(finalResidualNorm);

  logger.warn('gaussNewton', undefined, 'Maximum iterations reached', [
    { key: 'Iterations:', value: maxIterations },
    { key: 'Final cost:', value: finalCost },
    { key: 'Final residual norm:', value: finalResidualNorm }
  ]);

  return {
    parameters: currentParameters,
    iterations: maxIterations,
    converged: false,
    finalCost: finalCost,
    finalGradientNorm: undefined,
    finalResidualNorm: finalResidualNorm
  };
}

