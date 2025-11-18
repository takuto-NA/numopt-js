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
import { float64ArrayToMatrix, matrixToFloat64Array, vectorNorm, computeSumOfSquaredResiduals } from '../utils/matrix';
import { checkGradientConvergence, checkStepSizeConvergence, checkResidualConvergence } from './convergence';
import { computeJacobianMatrix } from './jacobianComputation';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_LAMBDA_INITIAL = 1e-3;
const DEFAULT_LAMBDA_FACTOR = 10.0;
const DEFAULT_TOL_GRADIENT = 1e-6;
const DEFAULT_TOL_STEP = 1e-6;
const DEFAULT_TOL_RESIDUAL = 1e-6;
const DEFAULT_USE_NUMERIC_JACOBIAN = true;
const DEFAULT_JACOBIAN_STEP = 1e-6;
const MAXIMUM_LAMBDA_THRESHOLD = 1e10; // Maximum lambda before giving up (prevents infinite loop)
const NEGATIVE_COEFFICIENT = -1.0; // Coefficient for negative right-hand side in damped normal equations: (J^T J + λI) δ = -J^T r

/**
 * Computes J^T J and J^T r matrices needed for normal equations.
 * Returns both matrices for use in solving damped normal equations.
 */
function computeNormalEquationsMatrices(
  jacobianMatrix: Matrix,
  residual: Float64Array
): { jtj: Matrix; jtr: Matrix } {
  const jacobianTranspose = jacobianMatrix.transpose();
  const jtj = jacobianTranspose.mmul(jacobianMatrix);
  const residualMatrix = float64ArrayToMatrix(residual);
  const jtr = jacobianTranspose.mmul(residualMatrix);
  return { jtj, jtr };
}

/**
 * Creates a convergence result object for Levenberg-Marquardt algorithm.
 * Centralizes result creation to avoid code duplication.
 */
function createConvergenceResultForLM(
  parameters: Float64Array,
  iteration: number,
  converged: boolean,
  finalCost: number,
  finalGradientNorm: number,
  finalResidualNorm: number,
  finalLambda: number
): LevenbergMarquardtResult {
  return {
    parameters,
    iterations: iteration + 1,
    converged,
    finalCost,
    finalGradientNorm,
    finalResidualNorm,
    finalLambda
  };
}

/**
 * Tries a Levenberg-Marquardt step by solving damped normal equations.
 * Returns whether step was accepted and updated parameters/lambda.
 */
function tryLevenbergMarquardtStep(
  jtj: Matrix,
  jtr: Matrix,
  currentParameters: Float64Array,
  currentLambda: number,
  lambdaFactor: number,
  residualFunction: ResidualFn,
  currentCost: number,
  tolStep: number,
  iteration: number,
  verbose: boolean
): {
  stepAccepted: boolean;
  newParameters?: Float64Array;
  newLambda: number;
  stepNorm?: number;
  shouldStop?: boolean;
} {
  // Early return: lambda too large
  if (currentLambda >= MAXIMUM_LAMBDA_THRESHOLD) {
    if (verbose) {
      console.log(`Lambda too large, stopping optimization`);
    }
    return { stepAccepted: false, newLambda: currentLambda, shouldStop: true };
  }

  try {
    // Add damping: J^T J + λI
    const parameterCount = jtj.rows;
    const identity = Matrix.eye(parameterCount, parameterCount);
    const dampedHessian = jtj.add(identity.mul(currentLambda));

    // Solve: (J^T J + λI) δ = -J^T r
    const negativeJtr = jtr.mul(NEGATIVE_COEFFICIENT);
    const stepMatrix = solve(dampedHessian, negativeJtr);
    const step = matrixToFloat64Array(stepMatrix);
    const stepNorm = vectorNorm(step);

    // Check step size convergence
    if (checkStepSizeConvergence(stepNorm, tolStep, iteration)) {
      return { stepAccepted: false, newLambda: currentLambda, stepNorm };
    }

    // Try the step: x_new = x_old + δ
    const newParameters = new Float64Array(currentParameters.length);
    for (let i = 0; i < currentParameters.length; i++) {
      newParameters[i] = currentParameters[i] + step[i];
    }

    const newResidual = residualFunction(newParameters);
    const newResidualNorm = vectorNorm(newResidual);
    const newCost = computeSumOfSquaredResiduals(newResidualNorm);

    // Check if step improved the cost
    if (newCost < currentCost) {
      // Step successful: accept it and decrease lambda
      const newLambda = currentLambda / lambdaFactor;
      if (verbose) {
        console.log(
          `Iteration ${iteration}: cost improved ${currentCost} -> ${newCost}, lambda = ${newLambda}`
        );
      }
      return { stepAccepted: true, newParameters, newLambda };
    }

    // Step failed: reject it and increase lambda
    const newLambda = currentLambda * lambdaFactor;
    if (verbose) {
      console.log(
        `Iteration ${iteration}: step rejected, cost increased ${currentCost} -> ${newCost}, lambda = ${newLambda}`
      );
    }
    return { stepAccepted: false, newLambda };
  } catch (error) {
    // Singular matrix or numerical issues: increase lambda and retry
    const newLambda = currentLambda * lambdaFactor;
    if (verbose) {
      console.log(`Singular matrix at iteration ${iteration}, increasing lambda to ${newLambda}`);
    }
    return { stepAccepted: false, newLambda };
  }
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
    const cost = computeSumOfSquaredResiduals(residualNorm);

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
      jacobianStep,
      'levenbergMarquardt'
    );

    // Compute J^T J and J^T r
    const { jtj, jtr } = computeNormalEquationsMatrices(jacobianMatrix, residual);

    // Compute gradient norm: ||J^T r||
    const gradientVector = matrixToFloat64Array(jtr);
    const gradientNorm = vectorNorm(gradientVector);

    // Check convergence: gradient norm is small enough
    if (checkGradientConvergence(gradientNorm, tolGradient, iteration)) {
      if (verbose) {
        console.log(`Converged at iteration ${iteration}: gradient norm = ${gradientNorm}`);
      }
      return createConvergenceResultForLM(
        currentParameters,
        iteration,
        true,
        cost,
        gradientNorm,
        residualNorm,
        currentLambda
      );
    }

    // Try to solve damped normal equations: (J^T J + λI) δ = -J^T r
    let stepAccepted = false;
    while (!stepAccepted && currentLambda < MAXIMUM_LAMBDA_THRESHOLD) {
      const stepResult = tryLevenbergMarquardtStep(
        jtj,
        jtr,
        currentParameters,
        currentLambda,
        lambdaFactor,
        residualFunction,
        cost,
        tolStep,
        iteration,
        verbose
      );

      // Early return: lambda too large
      if (stepResult.shouldStop) {
        return createConvergenceResultForLM(
          bestParameters,
          iteration,
          false,
          bestCost,
          gradientNorm,
          residualNorm,
          stepResult.newLambda
        );
      }

      // Early return: step size convergence
      if (stepResult.stepNorm !== undefined && checkStepSizeConvergence(stepResult.stepNorm, tolStep, iteration)) {
        if (verbose) {
          console.log(`Converged at iteration ${iteration}: step size = ${stepResult.stepNorm}`);
        }
        return createConvergenceResultForLM(
          currentParameters,
          iteration,
          true,
          cost,
          gradientNorm,
          residualNorm,
          currentLambda
        );
      }

      // Update lambda
      currentLambda = stepResult.newLambda;

      // Accept step if successful
      if (stepResult.stepAccepted && stepResult.newParameters) {
        currentParameters = new Float64Array(stepResult.newParameters);
        stepAccepted = true;
      }
    }

    // Check residual norm convergence
    const currentResidual = residualFunction(currentParameters);
    const currentResidualNorm = vectorNorm(currentResidual);
    if (checkResidualConvergence(currentResidualNorm, tolResidual, iteration)) {
      if (verbose) {
        console.log(`Converged at iteration ${iteration}: residual norm = ${currentResidualNorm}`);
      }
      return createConvergenceResultForLM(
        currentParameters,
        iteration,
        true,
        cost,
        gradientNorm,
        currentResidualNorm,
        currentLambda
      );
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

