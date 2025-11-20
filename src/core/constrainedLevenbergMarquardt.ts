/**
 * This file implements the constrained Levenberg-Marquardt algorithm for solving
 * nonlinear least squares problems with constraints.
 * 
 * The constrained Levenberg-Marquardt method uses the effective Jacobian concept:
 * J_eff = r_p - r_x C_x^+ C_p, which captures all constraint effects.
 * This allows the algorithm to use the same structure as unconstrained
 * Levenberg-Marquardt: (J_eff^T J_eff + λI) δ = -J_eff^T r
 * 
 * Role in system:
 * - Constrained version of Levenberg-Marquardt method
 * - Uses effective Jacobian computed via adjoint method
 * - Updates both parameters and states to maintain constraint satisfaction
 * - More robust than constrained Gauss-Newton (handles singular matrices)
 * 
 * For first-time readers:
 * - Start with constrainedLevenbergMarquardt function
 * - Understand how effective Jacobian replaces regular Jacobian
 * - Check lambda update strategy and damping mechanism
 */

import { Matrix, solve, CholeskyDecomposition } from 'ml-matrix';
import type {
  ConstrainedResidualFn,
  ConstraintFn,
  ConstrainedLevenbergMarquardtOptions,
  ConstrainedLevenbergMarquardtResult
} from './types.js';
import { float64ArrayToMatrix, matrixToFloat64Array, vectorNorm, computeSumOfSquaredResiduals } from '../utils/matrix.js';
import { checkGradientConvergence, checkStepSizeConvergence, checkResidualConvergence } from './convergence.js';
import { computeEffectiveJacobian, type EffectiveJacobianOptions } from './effectiveJacobian.js';
import { updateStates, validateInitialConditions } from './constrainedUtils.js';
import { Logger } from './logger.js';
import {
  finiteDiffConstraintPartialP,
  finiteDiffConstraintPartialX
} from './finiteDiff.js';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_LAMBDA_INITIAL = 1e-3;
const DEFAULT_LAMBDA_FACTOR = 10.0;
const DEFAULT_TOL_GRADIENT = 1e-6;
const DEFAULT_TOL_STEP = 1e-6;
const DEFAULT_TOL_RESIDUAL = 1e-6;
const DEFAULT_CONSTRAINT_TOLERANCE = 1e-6;
const DEFAULT_STEP_SIZE_P = 1e-6;
const DEFAULT_STEP_SIZE_X = 1e-6;
const MAXIMUM_LAMBDA_THRESHOLD = 1e10; // Maximum lambda before giving up (prevents infinite loop)
const NEGATIVE_COEFFICIENT = -1.0; // Coefficient for negative right-hand side in damped normal equations: (J_eff^T J_eff + λI) δ = -J_eff^T r

/**
 * Computes J_eff^T J_eff and J_eff^T r matrices needed for normal equations.
 * Returns both matrices for use in solving damped normal equations.
 */
function computeNormalEquationsMatrices(
  effectiveJacobian: Matrix,
  residual: Float64Array
): { jtj: Matrix; jtr: Matrix } {
  const jacobianTranspose = effectiveJacobian.transpose();
  const jtj = jacobianTranspose.mmul(effectiveJacobian);
  const residualMatrix = float64ArrayToMatrix(residual);
  const jtr = jacobianTranspose.mmul(residualMatrix);
  return { jtj, jtr };
}

/**
 * Creates a convergence result object for constrained Levenberg-Marquardt algorithm.
 * Centralizes result creation to avoid code duplication.
 */
function createConvergenceResultForLM(
  parameters: Float64Array,
  states: Float64Array,
  iteration: number,
  converged: boolean,
  finalCost: number,
  finalGradientNorm: number,
  finalResidualNorm: number,
  finalConstraintNorm: number,
  finalLambda: number
): ConstrainedLevenbergMarquardtResult {
  return {
    parameters,
    iterations: iteration + 1,
    converged,
    finalCost,
    finalGradientNorm,
    finalResidualNorm,
    finalLambda,
    finalStates: states,
    finalConstraintNorm
  };
}

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
): { constraintNorm: number } {
  const constraint = constraintFunction(currentParameters, currentStates);
  const constraintNorm = vectorNorm(constraint);
  if (constraintNorm > constraintTolerance) {
    logger.warn('constrainedLevenbergMarquardt', iteration, 'Constraint violation detected', [
      { key: '||c(p,x)||:', value: constraintNorm },
      { key: 'Tolerance:', value: constraintTolerance }
    ]);
  }
  return { constraintNorm };
}

/**
 * Tries a constrained Levenberg-Marquardt step by solving damped normal equations.
 * Returns whether step was accepted and updated parameters/states/lambda.
 */
function tryConstrainedLevenbergMarquardtStep(
  jtj: Matrix,
  jtr: Matrix,
  currentParameters: Float64Array,
  currentStates: Float64Array,
  currentLambda: number,
  lambdaFactor: number,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  currentCost: number,
  tolStep: number,
  iteration: number,
  stepSizeP: number,
  stepSizeX: number,
  dcdp?: (parameters: Float64Array, states: Float64Array) => Matrix,
  dcdx?: (parameters: Float64Array, states: Float64Array) => Matrix,
  logger?: Logger
): {
  stepAccepted: boolean;
  newParameters?: Float64Array;
  newStates?: Float64Array;
  newLambda: number;
  stepNorm?: number;
  shouldStop?: boolean;
} {
  // Early return: lambda too large
  if (currentLambda >= MAXIMUM_LAMBDA_THRESHOLD) {
    if (logger) {
      logger.warn('constrainedLevenbergMarquardt', iteration, 'Lambda too large, stopping optimization', [
        { key: 'Lambda:', value: currentLambda },
        { key: 'Cost:', value: currentCost }
      ]);
    }
    return { stepAccepted: false, newLambda: currentLambda, shouldStop: true };
  }

  try {
    // Add damping: J_eff^T J_eff + λI
    const parameterCount = jtj.rows;
    const identity = Matrix.eye(parameterCount, parameterCount);
    const dampedHessian = jtj.add(identity.mul(currentLambda));

    // Solve: (J_eff^T J_eff + λI) δ = -J_eff^T r
    // Use Cholesky decomposition for efficiency (dampedHessian is always positive definite when λ > 0)
    const negativeJtr = jtr.mul(NEGATIVE_COEFFICIENT);
    let stepMatrix: Matrix;
    try {
      const cholesky = new CholeskyDecomposition(dampedHessian);
      if (cholesky.isPositiveDefinite()) {
        stepMatrix = cholesky.solve(negativeJtr);
      } else {
        stepMatrix = solve(dampedHessian, negativeJtr);
      }
    } catch (choleskyError) {
      stepMatrix = solve(dampedHessian, negativeJtr);
    }
    const step = matrixToFloat64Array(stepMatrix);
    const stepNorm = vectorNorm(step);

    // Check step size convergence
    if (checkStepSizeConvergence(stepNorm, tolStep, iteration)) {
      return { stepAccepted: false, newLambda: currentLambda, stepNorm };
    }

    // Try the step: p_new = p_old + δ
    const newParameters = new Float64Array(currentParameters.length);
    for (let i = 0; i < currentParameters.length; i++) {
      newParameters[i] = currentParameters[i] + step[i];
    }

    // Update states using linear approximation
    const c_x = dcdx
      ? dcdx(currentParameters, currentStates)
      : finiteDiffConstraintPartialX(currentParameters, currentStates, constraintFunction, { stepSize: stepSizeX });
    const c_p = dcdp
      ? dcdp(currentParameters, currentStates)
      : finiteDiffConstraintPartialP(currentParameters, currentStates, constraintFunction, { stepSize: stepSizeP });

    const newStates = updateStates(currentStates, c_x, c_p, step) as Float64Array;

    const newResidual = residualFunction(newParameters, newStates);
    const newResidualNorm = vectorNorm(newResidual);
    const newCost = computeSumOfSquaredResiduals(newResidualNorm);

    // Check if step improved the cost
    if (newCost < currentCost) {
      // Step successful: accept it and decrease lambda
      const newLambda = currentLambda / lambdaFactor;
      if (logger) {
        logger.debug('constrainedLevenbergMarquardt', iteration, 'Step accepted', [
          { key: 'Cost:', value: currentCost },
          { key: 'New cost:', value: newCost },
          { key: 'Lambda:', value: newLambda }
        ]);
      }
      return { stepAccepted: true, newParameters, newStates, newLambda };
    }

    // Step failed: reject it and increase lambda
    const newLambda = currentLambda * lambdaFactor;
    if (logger) {
      logger.debug('constrainedLevenbergMarquardt', iteration, 'Step rejected', [
        { key: 'Cost:', value: currentCost },
        { key: 'New cost:', value: newCost },
        { key: 'Lambda:', value: newLambda }
      ]);
    }
    return { stepAccepted: false, newLambda };
  } catch (error) {
    // Singular matrix or numerical issues: increase lambda and retry
    const newLambda = currentLambda * lambdaFactor;
    if (logger) {
      logger.warn('constrainedLevenbergMarquardt', iteration, 'Singular matrix encountered, increasing lambda', [
        { key: 'Lambda:', value: newLambda },
        { key: 'Cost:', value: currentCost }
      ]);
    }
    return { stepAccepted: false, newLambda };
  }
}

/**
 * Performs one iteration of constrained Levenberg-Marquardt optimization.
 * Returns updated parameters/states/lambda or null if converged/error occurred.
 */
function performConstrainedLevenbergMarquardtIteration(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  currentLambda: number,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  effectiveJacobianOptions: EffectiveJacobianOptions,
  tolGradient: number,
  tolStep: number,
  tolResidual: number,
  constraintTolerance: number,
  stepSizeP: number,
  stepSizeX: number,
  lambdaFactor: number,
  iteration: number,
  logger: Logger,
  onIteration?: (iteration: number, cost: number, parameters: Float64Array) => void,
  dcdp?: (parameters: Float64Array, states: Float64Array) => Matrix,
  dcdx?: (parameters: Float64Array, states: Float64Array) => Matrix
): {
  converged: boolean;
  result?: ConstrainedLevenbergMarquardtResult;
  newParameters?: Float64Array;
  newStates?: Float64Array;
  newLambda?: number;
  bestCost?: number;
  bestParameters?: Float64Array;
  bestStates?: Float64Array;
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

  if (onIteration) {
    onIteration(iteration, cost, currentParameters);
  }

  const effectiveJacobian = computeEffectiveJacobian(
    currentParameters,
    currentStates,
    residualFunction,
    constraintFunction,
    effectiveJacobianOptions,
    logger,
    'constrainedLevenbergMarquardt'
  );

  const { jtj, jtr } = computeNormalEquationsMatrices(effectiveJacobian, residual);
  const gradientVector = matrixToFloat64Array(jtr);
  const gradientNorm = vectorNorm(gradientVector);

  if (checkGradientConvergence(gradientNorm, tolGradient, iteration)) {
    logger.info('constrainedLevenbergMarquardt', iteration, 'Converged', [
      { key: 'Cost:', value: cost },
      { key: 'Gradient norm:', value: gradientNorm },
      { key: 'Residual norm:', value: residualNorm },
      { key: 'Constraint norm:', value: constraintNorm },
      { key: 'Lambda:', value: currentLambda }
    ]);
    return {
      converged: true,
      result: createConvergenceResultForLM(
        currentParameters,
        currentStates,
        iteration,
        true,
        cost,
        gradientNorm,
        residualNorm,
        constraintNorm,
        currentLambda
      )
    };
  }

  let stepAccepted = false;
  let updatedLambda = currentLambda;
  let updatedParameters: Float64Array | undefined;
  let updatedStates: Float64Array | undefined;

  while (!stepAccepted && updatedLambda < MAXIMUM_LAMBDA_THRESHOLD) {
    const stepResult = tryConstrainedLevenbergMarquardtStep(
      jtj,
      jtr,
      currentParameters,
      currentStates,
      updatedLambda,
      lambdaFactor,
      residualFunction,
      constraintFunction,
      cost,
      tolStep,
      iteration,
      stepSizeP,
      stepSizeX,
      dcdp,
      dcdx,
      logger
    );

    if (stepResult.shouldStop) {
      return { converged: false };
    }

    if (stepResult.stepNorm !== undefined && checkStepSizeConvergence(stepResult.stepNorm, tolStep, iteration)) {
      logger.info('constrainedLevenbergMarquardt', iteration, 'Converged', [
        { key: 'Cost:', value: cost },
        { key: 'Gradient norm:', value: gradientNorm },
        { key: 'Residual norm:', value: residualNorm },
        { key: 'Step size:', value: stepResult.stepNorm },
        { key: 'Constraint norm:', value: constraintNorm },
        { key: 'Lambda:', value: updatedLambda }
      ]);
      return {
        converged: true,
        result: createConvergenceResultForLM(
          currentParameters,
          currentStates,
          iteration,
          true,
          cost,
          gradientNorm,
          residualNorm,
          constraintNorm,
          updatedLambda
        )
      };
    }

    updatedLambda = stepResult.newLambda;

    if (stepResult.stepAccepted && stepResult.newParameters && stepResult.newStates) {
      updatedParameters = stepResult.newParameters as Float64Array;
      updatedStates = stepResult.newStates as Float64Array;
      stepAccepted = true;
    }
  }

  if (!stepAccepted && updatedLambda >= MAXIMUM_LAMBDA_THRESHOLD) {
    logger.warn('constrainedLevenbergMarquardt', iteration, 'Could not find acceptable step even with maximum lambda. Stopping optimization.', [
      { key: 'Lambda:', value: updatedLambda },
      { key: 'Cost:', value: cost }
    ]);
    return { converged: false };
  }

  if (updatedParameters && updatedStates) {
    const currentResidual = residualFunction(updatedParameters, updatedStates);
    const currentResidualNorm = vectorNorm(currentResidual);
    const currentCost = computeSumOfSquaredResiduals(currentResidualNorm);
    const currentConstraint = constraintFunction(updatedParameters, updatedStates);
    const currentConstraintNorm = vectorNorm(currentConstraint);

    if (checkResidualConvergence(currentResidualNorm, tolResidual, iteration)) {
      logger.info('constrainedLevenbergMarquardt', iteration, 'Converged', [
        { key: 'Cost:', value: currentCost },
        { key: 'Gradient norm:', value: gradientNorm },
        { key: 'Residual norm:', value: currentResidualNorm },
        { key: 'Constraint norm:', value: currentConstraintNorm },
        { key: 'Lambda:', value: updatedLambda }
      ]);
      return {
        converged: true,
        result: createConvergenceResultForLM(
          updatedParameters,
          updatedStates,
          iteration,
          true,
          currentCost,
          gradientNorm,
          currentResidualNorm,
          currentConstraintNorm,
          updatedLambda
        )
      };
    }

    return {
      converged: false,
      newParameters: updatedParameters,
      newStates: updatedStates,
      newLambda: updatedLambda,
      bestCost: cost,
      bestParameters: currentParameters,
      bestStates: currentStates
    };
  }

  return { converged: false, newLambda: updatedLambda };
}

/**
 * Performs constrained Levenberg-Marquardt optimization for nonlinear least squares problems.
 * 
 * Algorithm:
 * 1. Start with initial parameters p0, states x0, and lambda (damping parameter)
 * 2. Compute effective Jacobian J_eff = r_p - r_x C_x^+ C_p
 * 3. Solve damped normal equations: (J_eff^T J_eff + λI) δ = -J_eff^T r
 * 4. Try step: p_new = p_old + δ, x_new updated using linear approximation
 * 5. If cost decreases: accept step, decrease lambda
 * 6. If cost increases: reject step, increase lambda
 * 7. Repeat until convergence
 * 
 * The damping parameter lambda interpolates between:
 * - Constrained Gauss-Newton (λ → 0): fast convergence near solution
 * - Constrained gradient descent (λ → ∞): robust but slow
 * 
 * @param initialParameters - Initial parameter vector p0
 * @param initialStates - Initial state vector x0 (should satisfy c(p0, x0) = 0)
 * @param residualFunction - Residual function r(p, x)
 * @param constraintFunction - Constraint function c(p, x) = 0
 * @param options - Optimization options
 * @returns Optimization result with final parameters, states, constraint norm, and lambda
 */
export function constrainedLevenbergMarquardt(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: ConstrainedLevenbergMarquardtOptions = {}
): ConstrainedLevenbergMarquardtResult {
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const lambdaInitial = options.lambdaInitial ?? DEFAULT_LAMBDA_INITIAL;
  const lambdaFactor = options.lambdaFactor ?? DEFAULT_LAMBDA_FACTOR;
  const tolGradient = options.tolGradient ?? DEFAULT_TOL_GRADIENT;
  const tolStep = options.tolStep ?? DEFAULT_TOL_STEP;
  const tolResidual = options.tolResidual ?? DEFAULT_TOL_RESIDUAL;
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
    'constrainedLevenbergMarquardt'
  );

  const effectiveJacobianOptions: EffectiveJacobianOptions = {
    drdp: options.drdp,
    drdx: options.drdx,
    dcdp: options.dcdp,
    dcdx: options.dcdx,
    stepSizeP,
    stepSizeX
  };

  let currentParameters: Float64Array = new Float64Array(initialParameters);
  let currentStates: Float64Array = new Float64Array(initialStates);
  let currentLambda = lambdaInitial;
  let bestParameters: Float64Array = new Float64Array(initialParameters);
  let bestStates: Float64Array = new Float64Array(initialStates);
  let bestCost = Infinity;

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const iterationResult = performConstrainedLevenbergMarquardtIteration(
      currentParameters,
      currentStates,
      currentLambda,
      residualFunction,
      constraintFunction,
      effectiveJacobianOptions,
      tolGradient,
      tolStep,
      tolResidual,
      constraintTolerance,
      stepSizeP,
      stepSizeX,
      lambdaFactor,
      iteration,
      logger,
      onIteration,
      options.dcdp,
      options.dcdx
    );

    if (iterationResult.converged && iterationResult.result) {
      return iterationResult.result;
    }

    if (iterationResult.bestCost !== undefined && iterationResult.bestCost < bestCost) {
      bestCost = iterationResult.bestCost;
      if (iterationResult.bestParameters && iterationResult.bestStates) {
        bestParameters = new Float64Array(iterationResult.bestParameters);
        bestStates = new Float64Array(iterationResult.bestStates);
      }
    }

    if (iterationResult.newParameters && iterationResult.newStates) {
      currentParameters = iterationResult.newParameters as Float64Array;
      currentStates = iterationResult.newStates as Float64Array;
      if (iterationResult.newLambda !== undefined) {
        currentLambda = iterationResult.newLambda;
      }
    } else {
      break;
    }
  }

  // Maximum iterations reached - return best solution found
  const finalResidual = residualFunction(bestParameters, bestStates);
  const finalResidualNorm = vectorNorm(finalResidual);
  const finalConstraint = constraintFunction(bestParameters, bestStates);
  const finalConstraintNorm = vectorNorm(finalConstraint);

  logger.warn('constrainedLevenbergMarquardt', undefined, 'Maximum iterations reached', [
    { key: 'Iterations:', value: maxIterations },
    { key: 'Final cost:', value: bestCost },
    { key: 'Final gradient norm:', value: vectorNorm(matrixToFloat64Array(computeNormalEquationsMatrices(
      computeEffectiveJacobian(
        bestParameters,
        bestStates,
        residualFunction,
        constraintFunction,
        {
          drdp: options.drdp,
          drdx: options.drdx,
          dcdp: options.dcdp,
          dcdx: options.dcdx,
          stepSizeP,
          stepSizeX
        },
        logger,
        'constrainedLevenbergMarquardt'
      ),
      finalResidual
    ).jtr)) },
    { key: 'Final residual norm:', value: finalResidualNorm },
    { key: 'Final constraint norm:', value: finalConstraintNorm },
    { key: 'Final lambda:', value: currentLambda }
  ]);

  // Compute final gradient norm for result
  const finalEffectiveJacobian = computeEffectiveJacobian(
    bestParameters,
    bestStates,
    residualFunction,
    constraintFunction,
    {
      drdp: options.drdp,
      drdx: options.drdx,
      dcdp: options.dcdp,
      dcdx: options.dcdx,
      stepSizeP,
      stepSizeX
    },
    logger,
    'constrainedLevenbergMarquardt'
  );
  const { jtr: finalJtr } = computeNormalEquationsMatrices(finalEffectiveJacobian, finalResidual);
  const finalGradientNorm = vectorNorm(matrixToFloat64Array(finalJtr));

  return createConvergenceResultForLM(
    bestParameters,
    bestStates,
    maxIterations - 1,
    false,
    bestCost,
    finalGradientNorm,
    finalResidualNorm,
    finalConstraintNorm,
    currentLambda
  );
}

