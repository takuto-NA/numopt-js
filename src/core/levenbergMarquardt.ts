/**
 * This file implements the Levenberg-Marquardt algorithm for solving
 * nonlinear least squares problems.
 * 
 * Role in system:
 * - Phase 3 advanced algorithm (main MVP target)
 * - Combines Gauss-Newton method with damping for robustness
 * - Handles cases where Gauss-Newton might fail (singular matrices, poor conditioning)
 * 
 * For first-time readers:
 * - Start with levenbergMarquardt function
 * - Understand lambda (damping parameter) update strategy
 * - Check convergence criteria implementation
 * - Debug features (callbacks, verbose logging) are top priority
 */

import { Matrix, solve } from 'ml-matrix';
import type {
  ResidualFn,
  JacobianFn,
  LevenbergMarquardtOptions,
  LevenbergMarquardtResult
} from './types';
import { finiteDiffJacobian } from './finiteDiff';
import { float64ArrayToMatrix, matrixToFloat64Array, vectorNorm } from '../utils/matrix';
import { checkGradientConvergence, checkStepSizeConvergence, checkResidualConvergence } from './convergence';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_LAMBDA_INITIAL = 1e-3;
const DEFAULT_LAMBDA_FACTOR = 10.0;
const DEFAULT_TOL_GRADIENT = 1e-6;
const DEFAULT_TOL_STEP = 1e-6;
const DEFAULT_TOL_RESIDUAL = 1e-6;
const DEFAULT_USE_NUMERIC_JACOBIAN = true;
const DEFAULT_JACOBIAN_STEP = 1e-6;
const MAXIMUM_LAMBDA_THRESHOLD = 1e10; // Maximum lambda before giving up (prevents infinite loop)

/**
 * Computes the Jacobian matrix using analytical function or numerical differentiation.
 * Early return pattern: prefers analytical Jacobian if available.
 */
function computeJacobianMatrix(
  jacobianFunction: JacobianFn | undefined,
  residualFunction: ResidualFn,
  parameters: Float64Array,
  useNumericJacobian: boolean,
  jacobianStep: number
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
    'Please either:\n' +
    '  1. Provide a jacobian in options: levenbergMarquardt(params, residualFn, { jacobian: jacobianFn })\n' +
    '  2. Enable numerical Jacobian: levenbergMarquardt(params, residualFn, { useNumericJacobian: true })\n' +
    'Note: Numerical Jacobian is enabled by default. If you see this error, it may have been explicitly disabled.'
  );
}

/**
 * Performs Levenberg-Marquardt optimization for nonlinear least squares problems.
 * 
 * Algorithm:
 * 1. Start with initial parameters and lambda (damping parameter)
 * 2. Compute residual vector r and Jacobian matrix J
 * 3. Solve damped normal equations: (J^T J + λI) δ = -J^T r
 * 4. Try step: x_new = x_old + δ
 * 5. If cost decreases: accept step, decrease lambda
 * 6. If cost increases: reject step, increase lambda
 * 7. Repeat until convergence
 * 
 * The damping parameter lambda interpolates between:
 * - Gauss-Newton (λ → 0): fast convergence near solution
 * - Gradient descent (λ → ∞): robust but slow
 */
export function levenbergMarquardt(
  initialParameters: Float64Array,
  residualFunction: ResidualFn,
  options: LevenbergMarquardtOptions = {}
): LevenbergMarquardtResult {
  const actualOptions: LevenbergMarquardtOptions = options;
  const jacobianFunction: JacobianFn | undefined = actualOptions.jacobian;

  const maxIterations = actualOptions.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const lambdaInitial = actualOptions.lambdaInitial ?? DEFAULT_LAMBDA_INITIAL;
  const lambdaFactor = actualOptions.lambdaFactor ?? DEFAULT_LAMBDA_FACTOR;
  const tolGradient = actualOptions.tolGradient ?? DEFAULT_TOL_GRADIENT;
  const tolStep = actualOptions.tolStep ?? DEFAULT_TOL_STEP;
  const tolResidual = actualOptions.tolResidual ?? DEFAULT_TOL_RESIDUAL;
  const useNumericJacobian = actualOptions.useNumericJacobian ?? DEFAULT_USE_NUMERIC_JACOBIAN;
  const jacobianStep = actualOptions.jacobianStep ?? DEFAULT_JACOBIAN_STEP;
  const onIteration = actualOptions.onIteration;
  const verbose = actualOptions.verbose ?? false;

  let currentParameters = new Float64Array(initialParameters);
  let currentLambda = lambdaInitial;
  let bestParameters = new Float64Array(initialParameters);
  let bestCost = Infinity;

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    // Compute residual vector
    const residual = residualFunction(currentParameters);
    const residualNorm = vectorNorm(residual);
    const cost = residualNorm * residualNorm; // Sum of squared residuals

    // Track best solution so far
    if (cost < bestCost) {
      bestCost = cost;
      bestParameters = new Float64Array(currentParameters);
    }

    // Call progress callback if provided (after first iteration)
    if (onIteration && iteration > 0) {
      onIteration(iteration - 1, cost, currentParameters);
    }

    // Compute Jacobian matrix
    // Early return: use analytical Jacobian if provided
    const jacobianMatrix: Matrix = computeJacobianMatrix(
      jacobianFunction,
      residualFunction,
      currentParameters,
      useNumericJacobian,
      jacobianStep
    );

    // Compute J^T J and J^T r
    const jacobianTranspose = jacobianMatrix.transpose();
    const jtj = jacobianTranspose.mmul(jacobianMatrix);
    const residualMatrix = float64ArrayToMatrix(residual);
    const jtr = jacobianTranspose.mmul(residualMatrix);

    // Compute gradient norm: ||J^T r||
    const gradientVector = matrixToFloat64Array(jtr);
    const gradientNorm = vectorNorm(gradientVector);

    // Check convergence: gradient norm is small enough
    if (checkGradientConvergence(gradientNorm, tolGradient, iteration)) {
      if (verbose) {
        console.log(`Converged at iteration ${iteration}: gradient norm = ${gradientNorm}`);
      }
      return {
        parameters: currentParameters,
        iterations: iteration + 1,
        converged: true,
        finalCost: cost,
        finalGradientNorm: gradientNorm,
        finalResidualNorm: residualNorm,
        finalLambda: currentLambda
      };
    }

    // Try to solve damped normal equations: (J^T J + λI) δ = -J^T r
    let step: Float64Array;
    let stepAccepted = false;

    while (!stepAccepted && currentLambda < MAXIMUM_LAMBDA_THRESHOLD) {
      try {
        // Add damping: J^T J + λI
        const parameterCount = jtj.rows;
        const identity = Matrix.eye(parameterCount, parameterCount);
        const dampedHessian = jtj.add(identity.mul(currentLambda));

        // Solve: (J^T J + λI) δ = -J^T r
        const negativeJtr = jtr.mul(-1.0);
        const stepMatrix = solve(dampedHessian, negativeJtr);
        step = matrixToFloat64Array(stepMatrix);

        // Check step size convergence
        const stepNorm = vectorNorm(step);
        if (checkStepSizeConvergence(stepNorm, tolStep, iteration)) {
          if (verbose) {
            console.log(`Converged at iteration ${iteration}: step size = ${stepNorm}`);
          }
          return {
            parameters: currentParameters,
            iterations: iteration + 1,
            converged: true,
            finalCost: cost,
            finalGradientNorm: gradientNorm,
            finalResidualNorm: residualNorm,
            finalLambda: currentLambda
          };
        }

        // Try the step: x_new = x_old + δ
        const newParameters = new Float64Array(currentParameters.length);
        for (let i = 0; i < currentParameters.length; i++) {
          newParameters[i] = currentParameters[i] + step[i];
        }

        const newResidual = residualFunction(newParameters);
        const newResidualNorm = vectorNorm(newResidual);
        const newCost = newResidualNorm * newResidualNorm;

        // Check if step improved the cost
        // Early return: handle successful step first
        if (newCost < cost) {
          // Step successful: accept it and decrease lambda
          currentParameters = newParameters;
          currentLambda = currentLambda / lambdaFactor;
          stepAccepted = true;

          if (verbose) {
            console.log(
              `Iteration ${iteration}: cost improved ${cost} -> ${newCost}, lambda = ${currentLambda}`
            );
          }
          continue;
        }

        // Step failed: reject it and increase lambda
        currentLambda = currentLambda * lambdaFactor;
        if (verbose) {
          console.log(
            `Iteration ${iteration}: step rejected, cost increased ${cost} -> ${newCost}, lambda = ${currentLambda}`
          );
        }
      } catch (error) {
        // Singular matrix or numerical issues: increase lambda and retry
        currentLambda = currentLambda * lambdaFactor;

        if (verbose) {
          console.log(`Singular matrix at iteration ${iteration}, increasing lambda to ${currentLambda}`);
        }

        // Early return: lambda too large, give up
        if (currentLambda >= MAXIMUM_LAMBDA_THRESHOLD) {
          // Lambda too large, give up (prevents infinite loop when matrix is severely ill-conditioned)
          if (verbose) {
            console.log(`Lambda too large, stopping optimization`);
          }
          return {
            parameters: bestParameters,
            iterations: iteration + 1,
            converged: false,
            finalCost: bestCost,
            finalGradientNorm: gradientNorm,
            finalResidualNorm: residualNorm,
            finalLambda: currentLambda
          };
        }
      }
    }

    // Check residual norm convergence
    const currentResidual = residualFunction(currentParameters);
    const currentResidualNorm = vectorNorm(currentResidual);
    if (checkResidualConvergence(currentResidualNorm, tolResidual, iteration)) {
      if (verbose) {
        console.log(`Converged at iteration ${iteration}: residual norm = ${currentResidualNorm}`);
      }
      return {
        parameters: currentParameters,
        iterations: iteration + 1,
        converged: true,
        finalCost: cost,
        finalGradientNorm: gradientNorm,
        finalResidualNorm: currentResidualNorm,
        finalLambda: currentLambda
      };
    }
  }

  // Maximum iterations reached - return best solution found
  const finalResidual = residualFunction(bestParameters);
  const finalResidualNorm = vectorNorm(finalResidual);
  const finalGradient = jacobianFunction
    ? matrixToFloat64Array(
        jacobianFunction(bestParameters).transpose().mmul(float64ArrayToMatrix(finalResidual))
      )
    : undefined;
  const finalGradientNorm = finalGradient ? vectorNorm(finalGradient) : undefined;

  if (verbose) {
    console.log(`Maximum iterations reached: ${maxIterations}`);
  }

  return {
    parameters: bestParameters,
    iterations: maxIterations,
    converged: false,
    finalCost: bestCost,
    finalGradientNorm: finalGradientNorm,
    finalResidualNorm: finalResidualNorm,
    finalLambda: currentLambda
  };
}

