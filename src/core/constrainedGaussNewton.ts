/**
 * This file implements the constrained Gauss-Newton method for solving
 * nonlinear least squares problems with constraints.
 * 
 * The constrained Gauss-Newton method uses the effective Jacobian concept:
 * J_eff = r_p - r_x C_x^+ C_p, which captures all constraint effects.
 * This allows the algorithm to use the same structure as unconstrained
 * Gauss-Newton: (J_eff^T J_eff) δ = -J_eff^T r
 * 
 * Role in system:
 * - Constrained version of Gauss-Newton method
 * - Uses effective Jacobian computed via adjoint method
 * - Updates both parameters and states to maintain constraint satisfaction
 * - Foundation for constrained Levenberg-Marquardt method
 * 
 * For first-time readers:
 * - Start with constrainedGaussNewton function
 * - Understand how effective Jacobian replaces regular Jacobian
 * - Check how states are updated using linear approximation
 */

import { Matrix, solve, CholeskyDecomposition } from 'ml-matrix';
import type {
  ConstrainedResidualFn,
  ConstraintFn,
  ConstrainedGaussNewtonOptions,
  ConstrainedGaussNewtonResult
} from './types.js';
import { float64ArrayToMatrix, matrixToFloat64Array, vectorNorm, computeSumOfSquaredResiduals } from '../utils/matrix.js';
import { checkStepSizeConvergence, checkResidualConvergence, createConvergenceResult } from './convergence.js';
import { computeEffectiveJacobian, type EffectiveJacobianOptions } from './effectiveJacobian.js';
import { updateStates, validateInitialConditions, projectStatesToConstraints } from './constrainedUtils.js';
import { Logger } from './logger.js';
import {
  finiteDiffConstraintPartialP,
  finiteDiffConstraintPartialX
} from './finiteDiff.js';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_CONSTRAINT_TOLERANCE = 1e-6;
const DEFAULT_STEP_SIZE_P = 1e-6;
const DEFAULT_STEP_SIZE_X = 1e-6;
const NEGATIVE_COEFFICIENT = -1.0; // Coefficient for negative right-hand side in normal equations: (J_eff^T J_eff) δ = -J_eff^T r
const DEFAULT_RIDGE_REGULARIZATION = 1e-8; // Small ridge regularization to recover positive definiteness when Cholesky fails

/**
 * Checks constraint violation and logs warning if needed.
 */
function checkConstraintViolation(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  constraintFunction: ConstraintFn,
  constraintTolerance: number,
  iteration: number,
  logger: Logger
): { constraint: Float64Array; constraintNorm: number } {
  const constraint = constraintFunction(currentParameters, currentStates);
  const constraintNorm = vectorNorm(constraint);
  if (constraintNorm > constraintTolerance) {
    logger.warn('constrainedGaussNewton', iteration, 'Constraint violation detected', [
      { key: '||c(p,x)||:', value: constraintNorm },
      { key: 'Tolerance:', value: constraintTolerance }
    ]);
  }
  return { constraint, constraintNorm };
}

/**
 * Solves normal equations for constrained Gauss-Newton: (J_eff^T J_eff) δ = -J_eff^T r
 * Returns the step vector δ, or throws if matrix is singular.
 */
function solveNormalEquationsForConstrainedGN(
  effectiveJacobian: Matrix,
  residual: Float64Array
): Float64Array {
  const effectiveJacobianTranspose = effectiveJacobian.transpose();
  const jacobianTransposeJacobian = effectiveJacobianTranspose.mmul(effectiveJacobian);
  const residualMatrix = float64ArrayToMatrix(residual);
  const jacobianTransposeResidual = effectiveJacobianTranspose.mmul(residualMatrix);

  const negativeJacobianTransposeResidual = jacobianTransposeResidual.mul(NEGATIVE_COEFFICIENT);
  const jittered = jacobianTransposeJacobian.add(Matrix.eye(jacobianTransposeJacobian.rows, jacobianTransposeJacobian.columns).mul(DEFAULT_RIDGE_REGULARIZATION));

  // Try Cholesky on original matrix first
  try {
    const cholesky = new CholeskyDecomposition(jacobianTransposeJacobian);
    if (cholesky.isPositiveDefinite()) {
      return matrixToFloat64Array(cholesky.solve(negativeJacobianTransposeResidual));
    }
  } catch (choleskyError) {
    // Fall through to ridge regularization
  }

  // Ridge regularization helps recover positive definiteness when matrix is near-singular
  // This improves numerical stability by adding small diagonal terms
  try {
    const choleskyRidge = new CholeskyDecomposition(jittered);
    if (choleskyRidge.isPositiveDefinite()) {
      return matrixToFloat64Array(choleskyRidge.solve(negativeJacobianTransposeResidual));
    }
  } catch (ridgeError) {
    // Fall through to general solver
  }

  // Final fallback: general solver (may fail if matrix is truly singular)
  return matrixToFloat64Array(solve(jacobianTransposeJacobian, negativeJacobianTransposeResidual));
}

/**
 * Updates parameters and states for constrained Gauss-Newton iteration.
 * Parameters are updated directly: p_new = p_old + δ
 * States are updated using linear approximation to maintain constraint satisfaction.
 */
function updateParametersAndStatesForConstrainedGN(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  step: Float64Array,
  constraintFunction: ConstraintFn,
  stepSizeP: number,
  stepSizeX: number,
  constraintTolerance: number,
  logger: Logger,
  dcdp?: (parameters: Float64Array, states: Float64Array) => Matrix,
  dcdx?: (parameters: Float64Array, states: Float64Array) => Matrix
): { newParameters: Float64Array; newStates: Float64Array } {
  const newParameters = new Float64Array(currentParameters.length) as Float64Array;
  for (let i = 0; i < currentParameters.length; i++) {
    newParameters[i] = currentParameters[i] + step[i];
  }

  const constraintJacobianX = dcdx
    ? dcdx(currentParameters, currentStates)
    : finiteDiffConstraintPartialX(currentParameters, currentStates, constraintFunction, { stepSize: stepSizeX });
  const constraintJacobianP = dcdp
    ? dcdp(currentParameters, currentStates)
    : finiteDiffConstraintPartialP(currentParameters, currentStates, constraintFunction, { stepSize: stepSizeP });

  const newStates = updateStates(currentStates, constraintJacobianX, constraintJacobianP, step, logger, 'constrainedGaussNewton') as Float64Array;
  const projectedStates = projectStatesToConstraints(
    newParameters,
    newStates,
    constraintFunction,
    stepSizeX,
    constraintTolerance,
    logger,
    'constrainedGaussNewton'
  );
  return { newParameters, newStates: projectedStates };
}

/**
 * Computes step vector for constrained Gauss-Newton iteration.
 * Handles singular matrix errors by returning convergence result.
 */
function computeStepForGN(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  effectiveJacobianOptions: EffectiveJacobianOptions,
  iteration: number,
  logger: Logger,
  cost: number,
  residualNorm: number,
  constraintNorm: number
): { step: Float64Array } | { converged: boolean; result: ConstrainedGaussNewtonResult } {
  const effectiveJacobian = computeEffectiveJacobian(
    currentParameters,
    currentStates,
    residualFunction,
    constraintFunction,
    effectiveJacobianOptions,
    logger,
    'constrainedGaussNewton'
  );

  const residual = residualFunction(currentParameters, currentStates);
  let step: Float64Array;
  try {
    step = solveNormalEquationsForConstrainedGN(effectiveJacobian, residual);
  } catch (error) {
    logger.warn('constrainedGaussNewton', iteration, 'Singular matrix encountered. Consider using constrained Levenberg-Marquardt method for better robustness.', [
      { key: 'Cost:', value: cost },
      { key: 'Residual norm:', value: residualNorm },
      { key: 'Constraint norm:', value: constraintNorm }
    ]);
    const result = createConvergenceResult(currentParameters, iteration, false, cost, undefined);
    return {
      converged: false,
      result: {
        ...result,
        finalResidualNorm: residualNorm,
        finalStates: currentStates,
        finalConstraintNorm: constraintNorm
      }
    };
  }
  return { step };
}

/**
 * Checks step size convergence for constrained Gauss-Newton.
 * Returns convergence result if converged, null otherwise.
 */
function checkStepSizeConvergenceForGN(
  stepNorm: number,
  constraintSatisfied: boolean,
  tolerance: number,
  iteration: number,
  currentParameters: Float64Array,
  currentStates: Float64Array,
  cost: number,
  residualNorm: number,
  constraintNorm: number,
  logger: Logger
): ConstrainedGaussNewtonResult | null {
  if (constraintSatisfied && checkStepSizeConvergence(stepNorm, tolerance, iteration)) {
    logger.info('constrainedGaussNewton', iteration, 'Converged', [
      { key: 'Cost:', value: cost },
      { key: 'Residual norm:', value: residualNorm },
      { key: 'Step size:', value: stepNorm },
      { key: 'Constraint norm:', value: constraintNorm }
    ]);
    const result = createConvergenceResult(currentParameters, iteration, true, cost, undefined);
    return {
      ...result,
      finalResidualNorm: residualNorm,
      finalStates: currentStates,
      finalConstraintNorm: constraintNorm
    };
  }
  return null;
}

/**
 * Checks residual convergence after parameter/state update.
 * Returns convergence result if converged, null otherwise.
 */
function checkResidualConvergenceForGN(
  newParameters: Float64Array,
  newStates: Float64Array,
  newResidualNorm: number,
  newCost: number,
  constraintFunction: ConstraintFn,
  constraintTolerance: number,
  tolerance: number,
  iteration: number,
  constraintNorm: number,
  logger: Logger
): ConstrainedGaussNewtonResult | null {
  if (checkResidualConvergence(newResidualNorm, tolerance, iteration)) {
    logger.info('constrainedGaussNewton', iteration, 'Converged', [
      { key: 'Cost:', value: newCost },
      { key: 'Residual norm:', value: newResidualNorm },
      { key: 'Constraint norm:', value: constraintNorm }
    ]);
    const finalConstraint = constraintFunction(newParameters, newStates);
    const finalConstraintNorm = vectorNorm(finalConstraint);
    if (finalConstraintNorm <= constraintTolerance) {
      const result = createConvergenceResult(newParameters, iteration, true, newCost, undefined);
      return {
        ...result,
        finalResidualNorm: newResidualNorm,
        finalStates: newStates,
        finalConstraintNorm: finalConstraintNorm
      };
    }
  }
  return null;
}

/**
 * Performs one iteration of constrained Gauss-Newton optimization.
 * Returns updated parameters/states or null if converged/error occurred.
 */
function performConstrainedGaussNewtonIteration(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  effectiveJacobianOptions: EffectiveJacobianOptions,
  tolerance: number,
  constraintTolerance: number,
  stepSizeP: number,
  stepSizeX: number,
  iteration: number,
  logger: Logger,
  onIteration?: (iteration: number, cost: number, parameters: Float64Array) => void,
  dcdp?: (parameters: Float64Array, states: Float64Array) => Matrix,
  dcdx?: (parameters: Float64Array, states: Float64Array) => Matrix
): {
  converged: boolean;
  result?: ConstrainedGaussNewtonResult;
  newParameters?: Float64Array;
  newStates?: Float64Array;
} {
  const { constraintNorm } = checkConstraintViolation(
    currentParameters,
    currentStates,
    constraintFunction,
    constraintTolerance,
    iteration,
    logger
  );

  const residual = residualFunction(currentParameters, currentStates);
  const residualNorm = vectorNorm(residual);
  const cost = computeSumOfSquaredResiduals(residualNorm);
  const constraintSatisfied = constraintNorm <= constraintTolerance;

  if (onIteration) {
    onIteration(iteration, cost, currentParameters);
  }

  const stepResult = computeStepForGN(
    currentParameters,
    currentStates,
    residualFunction,
    constraintFunction,
    effectiveJacobianOptions,
    iteration,
    logger,
    cost,
    residualNorm,
    constraintNorm
  );

  if ('converged' in stepResult) {
    return stepResult;
  }

  const step = stepResult.step;
  const stepNorm = vectorNorm(step);
  
  const stepSizeConvergenceResult = checkStepSizeConvergenceForGN(
    stepNorm,
    constraintSatisfied,
    tolerance,
    iteration,
    currentParameters,
    currentStates,
    cost,
    residualNorm,
    constraintNorm,
    logger
  );

  if (stepSizeConvergenceResult) {
    return {
      converged: true,
      result: stepSizeConvergenceResult
    };
  }

  const { newParameters, newStates } = updateParametersAndStatesForConstrainedGN(
    currentParameters,
    currentStates,
    step,
  constraintFunction,
  stepSizeP,
  stepSizeX,
  constraintTolerance,
  logger,
  dcdp,
  dcdx
);

  const newResidual = residualFunction(newParameters, newStates);
  const newResidualNorm = vectorNorm(newResidual);
  const newCost = computeSumOfSquaredResiduals(newResidualNorm);

  const residualConvergenceResult = checkResidualConvergenceForGN(
    newParameters,
    newStates,
    newResidualNorm,
    newCost,
    constraintFunction,
    constraintTolerance,
    tolerance,
    iteration,
    constraintNorm,
    logger
  );

  if (residualConvergenceResult) {
    return {
      converged: true,
      result: residualConvergenceResult
    };
  }

  logger.debug('constrainedGaussNewton', iteration, 'Progress', [
    { key: 'Cost:', value: cost },
    { key: 'Residual norm:', value: residualNorm },
    { key: 'Step norm:', value: stepNorm },
    { key: 'Constraint norm:', value: constraintNorm }
  ]);

  return { converged: false, newParameters, newStates };
}

/**
 * Runs the main iteration loop for constrained Gauss-Newton optimization.
 * Returns the result if converged, or final state if max iterations reached.
 */
function runGaussNewtonIterations(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  effectiveJacobianOptions: EffectiveJacobianOptions,
  tolerance: number,
  constraintTolerance: number,
  stepSizeP: number,
  stepSizeX: number,
  maxIterations: number,
  logger: Logger,
  onIteration?: (iteration: number, cost: number, parameters: Float64Array) => void,
  dcdp?: (parameters: Float64Array, states: Float64Array) => Matrix,
  dcdx?: (parameters: Float64Array, states: Float64Array) => Matrix
): { result: ConstrainedGaussNewtonResult } | { finalParameters: Float64Array; finalStates: Float64Array } {
  let currentParameters: Float64Array = new Float64Array(initialParameters);
  let currentStates: Float64Array = new Float64Array(initialStates);

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const iterationResult = performConstrainedGaussNewtonIteration(
      currentParameters,
      currentStates,
      residualFunction,
      constraintFunction,
      effectiveJacobianOptions,
      tolerance,
      constraintTolerance,
      stepSizeP,
      stepSizeX,
      iteration,
      logger,
      onIteration,
      dcdp,
      dcdx
    );

    if (iterationResult.converged && iterationResult.result) {
      return { result: iterationResult.result };
    }

    if (iterationResult.newParameters && iterationResult.newStates) {
      currentParameters = iterationResult.newParameters as Float64Array;
      currentStates = iterationResult.newStates as Float64Array;
    } else {
      break;
    }
  }

  return { finalParameters: currentParameters, finalStates: currentStates };
}

/**
 * Performs constrained Gauss-Newton optimization for nonlinear least squares problems.
 * 
 * Algorithm:
 * 1. Start with initial parameters p0 and states x0 (satisfying c(p0, x0) = 0)
 * 2. Compute effective Jacobian J_eff = r_p - r_x C_x^+ C_p
 * 3. Solve normal equations: (J_eff^T J_eff) δ = -J_eff^T r
 * 4. Update parameters: p_new = p_old + δ
 * 5. Update states: x_new = x_old + dx where (∂c/∂x) dx = -∂c/∂p · δ (linear approximation)
 * 6. Repeat until convergence
 * 
 * The effective Jacobian captures all constraint effects, allowing the algorithm
 * to use the same structure as unconstrained Gauss-Newton.
 * 
 * @param initialParameters - Initial parameter vector p0
 * @param initialStates - Initial state vector x0 (should satisfy c(p0, x0) = 0)
 * @param residualFunction - Residual function r(p, x)
 * @param constraintFunction - Constraint function c(p, x) = 0
 * @param options - Optimization options
 * @returns Optimization result with final parameters, states, and constraint norm
 */
export function constrainedGaussNewton(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: ConstrainedGaussNewtonOptions = {}
): ConstrainedGaussNewtonResult {
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const constraintTolerance = options.constraintTolerance ?? DEFAULT_CONSTRAINT_TOLERANCE;
  const stepSizeP = options.stepSizeP ?? DEFAULT_STEP_SIZE_P;
  const stepSizeX = options.stepSizeX ?? DEFAULT_STEP_SIZE_X;
  const onIteration = options.onIteration;
  const logger = new Logger(options.logLevel, options.verbose);

  validateInitialConditions(
    initialParameters,
    initialStates,
    constraintFunction,
    constraintTolerance,
    logger,
    'constrainedGaussNewton'
  );

  const effectiveJacobianOptions: EffectiveJacobianOptions = {
    drdp: options.drdp,
    drdx: options.drdx,
    dcdp: options.dcdp,
    dcdx: options.dcdx,
    stepSizeP,
    stepSizeX
  };

  const iterationResult = runGaussNewtonIterations(
    initialParameters,
    initialStates,
    residualFunction,
    constraintFunction,
    effectiveJacobianOptions,
    tolerance,
    constraintTolerance,
    stepSizeP,
    stepSizeX,
    maxIterations,
    logger,
    onIteration,
    options.dcdp,
    options.dcdx
  );

  if ('result' in iterationResult) {
    return iterationResult.result;
  }

  const currentParameters = iterationResult.finalParameters;
  const currentStates = iterationResult.finalStates;
  const finalResidual = residualFunction(currentParameters, currentStates);
  const finalResidualNorm = vectorNorm(finalResidual);
  const finalCost = computeSumOfSquaredResiduals(finalResidualNorm);
  const finalConstraint = constraintFunction(currentParameters, currentStates);
  const finalConstraintNorm = vectorNorm(finalConstraint);

  logger.warn('constrainedGaussNewton', undefined, 'Maximum iterations reached', [
    { key: 'Iterations:', value: maxIterations },
    { key: 'Final cost:', value: finalCost },
    { key: 'Final residual norm:', value: finalResidualNorm },
    { key: 'Final constraint norm:', value: finalConstraintNorm }
  ]);

  return {
    finalParameters: currentParameters,
    iterations: maxIterations,
    converged: false,
    finalCost: finalCost,
    finalGradientNorm: undefined,
    finalResidualNorm: finalResidualNorm,
    finalStates: currentStates,
    finalConstraintNorm: finalConstraintNorm
  };
}

