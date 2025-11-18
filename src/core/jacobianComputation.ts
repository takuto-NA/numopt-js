/**
 * This file provides a shared function for computing Jacobian matrices
 * using analytical functions or numerical differentiation.
 * 
 * Role in system:
 * - Eliminates code duplication between Gauss-Newton and Levenberg-Marquardt
 * - Centralizes Jacobian computation logic (DRY principle)
 * - Used by both least squares optimization algorithms
 * 
 * For first-time readers:
 * - This is a utility function used internally by optimization algorithms
 * - Prefers analytical Jacobian if provided, falls back to numerical differentiation
 */

import { Matrix } from 'ml-matrix';
import type { ResidualFn, JacobianFn } from './types';
import { finiteDiffJacobian } from './finiteDiff';

/**
 * Computes the Jacobian matrix using analytical function or numerical differentiation.
 * Early return pattern: prefers analytical Jacobian if available.
 * 
 * This function is shared between Gauss-Newton and Levenberg-Marquardt algorithms
 * to avoid code duplication.
 */
export function computeJacobianMatrix(
  jacobianFunction: JacobianFn | undefined,
  residualFunction: ResidualFn,
  parameters: Float64Array,
  useNumericJacobian: boolean,
  jacobianStep: number,
  algorithmName: string
): Matrix {
  // Early return: use analytical Jacobian if provided
  if (jacobianFunction) {
    return jacobianFunction(parameters);
  }

  // Early return: use numerical Jacobian if enabled
  if (useNumericJacobian) {
    return finiteDiffJacobian(residualFunction, parameters, { stepSize: jacobianStep });
  }

  // Neither provided: throw error with helpful message
  throw new Error(
    'Jacobian computation is required but not provided. ' +
    `Please either:\n` +
    `  1. Provide a jacobian in options: ${algorithmName}(params, residualFn, { jacobian: jacobianFn })\n` +
    `  2. Enable numerical Jacobian: ${algorithmName}(params, residualFn, { useNumericJacobian: true })\n` +
    'Note: Numerical Jacobian is enabled by default. If you see this error, it may have been explicitly disabled.'
  );
}

