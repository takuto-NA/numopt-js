/**
 * This file provides helper functions for creating gradient and Jacobian functions
 * from cost and residual functions using finite differences.
 * 
 * Role in system:
 * - Simplifies the API for users who want to use numerical differentiation
 * - Prevents common mistakes with parameter ordering
 * - Provides a more intuitive interface for optimization algorithms
 * 
 * For first-time readers:
 * - Use createFiniteDiffGradient when you have a cost function and need a gradient
 * - Use createFiniteDiffJacobian when you have a residual function and need a Jacobian
 * - These are convenience wrappers around finiteDiffGradient and finiteDiffJacobian
 */

import { finiteDiffGradient, finiteDiffJacobian } from './finiteDiff.js';
import type {
    CostFn,
    GradientFn,
    ResidualFn,
    JacobianFn,
    NumericalDifferentiationOptions
} from './types.js';

/**
 * Creates a gradient function from a cost function using finite differences.
 * 
 * This is a convenience wrapper around finiteDiffGradient that returns a gradient
 * function suitable for use with optimization algorithms like gradientDescent.
 * 
 * @param costFunction - The cost function to differentiate
 * @param options - Optional numerical differentiation settings
 * @returns A gradient function that can be passed to optimization algorithms
 * 
 * @example
 * ```typescript
 * import { gradientDescent, createFiniteDiffGradient } from 'numopt-js';
 * 
 * // Define your cost function
 * const costFn = (params) => Math.pow(params[0] - 3, 2) + Math.pow(params[1] - 2, 2);
 * 
 * // Create a gradient function (no need to worry about parameter order!)
 * const gradientFn = createFiniteDiffGradient(costFn);
 * 
 * // Use it with an optimizer
 * const result = gradientDescent(
 *   new Float64Array([0, 0]),
 *   costFn,
 *   gradientFn,
 *   { maxIterations: 100, tolerance: 1e-6 }
 * );
 * ```
 * 
 * @example
 * ```typescript
 * // With custom step size
 * const gradientFn = createFiniteDiffGradient(costFn, { stepSize: 1e-8 });
 * ```
 */
export function createFiniteDiffGradient(
    costFunction: CostFn,
    options?: NumericalDifferentiationOptions
): GradientFn {
    return (params: Float64Array): Float64Array => {
        return finiteDiffGradient(params, costFunction, options);
    };
}

/**
 * Creates a Jacobian function from a residual function using finite differences.
 * 
 * This is a convenience wrapper around finiteDiffJacobian that returns a Jacobian
 * function suitable for use with optimization algorithms like gaussNewton.
 * 
 * @param residualFunction - The residual function to differentiate
 * @param options - Optional numerical differentiation settings
 * @returns A Jacobian function that can be passed to optimization algorithms
 * 
 * @example
 * ```typescript
 * import { gaussNewton, createFiniteDiffJacobian } from 'numopt-js';
 * 
 * // Define your residual function
 * const residualFn = (params) => {
 *   // Return residuals for curve fitting, etc.
 *   return new Float64Array([...]);
 * };
 * 
 * // Create a Jacobian function
 * const jacobianFn = createFiniteDiffJacobian(residualFn);
 * 
 * // Use it with an optimizer
 * const result = gaussNewton(
 *   new Float64Array([1, 1]),
 *   residualFn,
 *   jacobianFn,
 *   { maxIterations: 100, tolerance: 1e-6 }
 * );
 * ```
 */
export function createFiniteDiffJacobian(
    residualFunction: ResidualFn,
    options?: NumericalDifferentiationOptions
): JacobianFn {
    return (params: Float64Array) => {
        return finiteDiffJacobian(residualFunction, params, options);
    };
}
