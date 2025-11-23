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

import { Matrix, CholeskyDecomposition } from 'ml-matrix';
import type {
  ConstrainedResidualFn,
  ConstraintFn,
  ConstrainedLevenbergMarquardtOptions,
  ConstrainedLevenbergMarquardtResult
} from './types.js';
import { float64ArrayToMatrix, matrixToFloat64Array, vectorNorm, computeSumOfSquaredResiduals } from '../utils/matrix.js';
import { checkGradientConvergence, checkStepSizeConvergence, checkResidualConvergence } from './convergence.js';
import { computeEffectiveJacobian, type EffectiveJacobianOptions } from './effectiveJacobian.js';
import { updateStates, validateInitialConditions, projectStatesToConstraints } from './constrainedUtils.js';
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
 * Solves damped normal equations for Levenberg-Marquardt step.
 * Damping parameter lambda interpolates between Gauss-Newton (λ→0) and gradient descent (λ→∞).
 * Returns step vector or increases lambda if Cholesky decomposition fails.
 */
function solveDampedNormalEquations(
  jtj: Matrix,
  jtr: Matrix,
  currentLambda: number,
  lambdaFactor: number
): { step: Float64Array; stepNorm: number } | { newLambda: number } {
  const parameterCount = jtj.rows;
  const identity = Matrix.eye(parameterCount, parameterCount);
  const dampedHessian = jtj.add(identity.mul(currentLambda));

  const negativeJtr = jtr.mul(NEGATIVE_COEFFICIENT);
  let stepMatrix: Matrix;
  try {
    const cholesky = new CholeskyDecomposition(dampedHessian);
    if (cholesky.isPositiveDefinite()) {
      stepMatrix = cholesky.solve(negativeJtr);
    } else {
      const newLambda = currentLambda * lambdaFactor;
      return { newLambda };
    }
  } catch (choleskyError) {
    const newLambda = currentLambda * lambdaFactor;
    return { newLambda };
  }
  const step = matrixToFloat64Array(stepMatrix);
  const stepNorm = vectorNorm(step);
  return { step, stepNorm };
}

/**
 * Evaluates step quality by comparing new cost to current cost.
 * Returns acceptance result with updated lambda based on cost improvement.
 */
function evaluateStepQuality(
  newParameters: Float64Array,
  newStates: Float64Array,
  newCost: number,
  currentCost: number,
  currentLambda: number,
  lambdaFactor: number,
  iteration: number,
  logger: Logger
): { stepAccepted: boolean; newParameters: Float64Array; newStates: Float64Array; newLambda: number } {
  if (newCost < currentCost) {
    const newLambda = currentLambda / lambdaFactor;
    logger.debug('constrainedLevenbergMarquardt', iteration, 'Step accepted', [
      { key: 'Cost:', value: currentCost },
      { key: 'New cost:', value: newCost },
      { key: 'Lambda:', value: newLambda }
    ]);
    return { stepAccepted: true, newParameters, newStates, newLambda };
  }

  const newLambda = currentLambda * lambdaFactor;
  logger.debug('constrainedLevenbergMarquardt', iteration, 'Step rejected', [
    { key: 'Cost:', value: currentCost },
    { key: 'New cost:', value: newCost },
    { key: 'Lambda:', value: newLambda }
  ]);
  return { stepAccepted: false, newParameters, newStates, newLambda };
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
  constraintTolerance: number,
  logger: Logger,
  dcdp?: (parameters: Float64Array, states: Float64Array) => Matrix,
  dcdx?: (parameters: Float64Array, states: Float64Array) => Matrix
): {
  stepAccepted: boolean;
  newParameters?: Float64Array;
  newStates?: Float64Array;
  newLambda: number;
  stepNorm?: number;
  shouldStop?: boolean;
} {
  // Lambda threshold prevents infinite loops when matrix is severely ill-conditioned
  if (currentLambda >= MAXIMUM_LAMBDA_THRESHOLD) {
    logger.warn('constrainedLevenbergMarquardt', iteration, 'Lambda too large, stopping optimization', [
      { key: 'Lambda:', value: currentLambda },
      { key: 'Cost:', value: currentCost }
    ]);
    return { stepAccepted: false, newLambda: currentLambda, shouldStop: true };
  }

  try {
    const solveResult = solveDampedNormalEquations(jtj, jtr, currentLambda, lambdaFactor);
    
    if ('newLambda' in solveResult) {
      return { stepAccepted: false, newLambda: solveResult.newLambda };
    }

    const { step, stepNorm } = solveResult;

    if (checkStepSizeConvergence(stepNorm, tolStep, iteration)) {
      const newLambda = currentLambda * lambdaFactor;
      return { stepAccepted: false, newLambda, stepNorm };
    }

    const newParameters = new Float64Array(currentParameters.length);
    for (let i = 0; i < currentParameters.length; i++) {
      newParameters[i] = currentParameters[i] + step[i];
    }

    const c_x = dcdx
      ? dcdx(currentParameters, currentStates)
      : finiteDiffConstraintPartialX(currentParameters, currentStates, constraintFunction, { stepSize: stepSizeX });
    const c_p = dcdp
      ? dcdp(currentParameters, currentStates)
      : finiteDiffConstraintPartialP(currentParameters, currentStates, constraintFunction, { stepSize: stepSizeP });

    const newStates = updateStates(currentStates, c_x, c_p, step, logger, 'constrainedLevenbergMarquardt') as Float64Array;

    const projectedStates = projectStatesToConstraints(
      newParameters,
      newStates,
      constraintFunction,
      stepSizeX,
      constraintTolerance,
      logger,
      'constrainedLevenbergMarquardt'
    );

    const newResidual = residualFunction(newParameters, projectedStates);
    const newResidualNorm = vectorNorm(newResidual);
    const newCost = computeSumOfSquaredResiduals(newResidualNorm);

    const evaluationResult = evaluateStepQuality(
      newParameters,
      projectedStates,
      newCost,
      currentCost,
      currentLambda,
      lambdaFactor,
      iteration,
      logger
    );

    return {
      stepAccepted: evaluationResult.stepAccepted,
      newParameters,
      newStates: evaluationResult.newStates,
      newLambda: evaluationResult.newLambda
    };
  } catch (error) {
    // Numerical issues indicate ill-conditioning: increasing lambda improves conditioning
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
 * Checks convergence criteria for constrained Levenberg-Marquardt.
 * Returns convergence result if converged, null otherwise.
 */
function checkConvergenceForLM(
  gradientNorm: number,
  constraintSatisfied: boolean,
  tolGradient: number,
  iteration: number,
  currentParameters: Float64Array,
  currentStates: Float64Array,
  cost: number,
  residualNorm: number,
  constraintNorm: number,
  currentLambda: number,
  logger: Logger
): ConstrainedLevenbergMarquardtResult | null {
  if (constraintSatisfied && checkGradientConvergence(gradientNorm, tolGradient, iteration)) {
    logger.info('constrainedLevenbergMarquardt', iteration, 'Converged', [
      { key: 'Cost:', value: cost },
      { key: 'Gradient norm:', value: gradientNorm },
      { key: 'Residual norm:', value: residualNorm },
      { key: 'Constraint norm:', value: constraintNorm },
      { key: 'Lambda:', value: currentLambda }
    ]);
    return createConvergenceResultForLM(
      currentParameters,
      currentStates,
      iteration,
      true,
      cost,
      gradientNorm,
      residualNorm,
      constraintNorm,
      currentLambda
    );
  }
  return null;
}

/**
 * Tries steps with increasing lambda until an acceptable step is found.
 * Returns step result or indicates that optimization should stop.
 */
function tryStepWithLambda(
  jtj: Matrix,
  jtr: Matrix,
  currentParameters: Float64Array,
  currentStates: Float64Array,
  currentLambda: number,
  lambdaFactor: number,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  cost: number,
  tolStep: number,
  iteration: number,
  stepSizeP: number,
  stepSizeX: number,
  constraintTolerance: number,
  constraintSatisfied: boolean,
  gradientNorm: number,
  residualNorm: number,
  constraintNorm: number,
  logger: Logger,
  dcdp?: (parameters: Float64Array, states: Float64Array) => Matrix,
  dcdx?: (parameters: Float64Array, states: Float64Array) => Matrix
): {
  stepAccepted: boolean;
  updatedParameters?: Float64Array;
  updatedStates?: Float64Array;
  updatedLambda: number;
  shouldStop?: boolean;
  stepSizeConverged?: boolean;
} {
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
      constraintTolerance,
      logger,
      dcdp,
      dcdx
    );

    if (stepResult.shouldStop) {
      return { stepAccepted: false, updatedLambda, shouldStop: true };
    }

    if (
      stepResult.stepNorm !== undefined &&
      constraintSatisfied &&
      checkStepSizeConvergence(stepResult.stepNorm, tolStep, iteration)
    ) {
      logger.info('constrainedLevenbergMarquardt', iteration, 'Converged', [
        { key: 'Cost:', value: cost },
        { key: 'Gradient norm:', value: gradientNorm },
        { key: 'Residual norm:', value: residualNorm },
        { key: 'Step size:', value: stepResult.stepNorm },
        { key: 'Constraint norm:', value: constraintNorm },
        { key: 'Lambda:', value: updatedLambda }
      ]);
      return { stepAccepted: false, updatedLambda, stepSizeConverged: true };
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
    return { stepAccepted: false, updatedLambda, shouldStop: true };
  }

  return { stepAccepted, updatedParameters, updatedStates, updatedLambda };
}

/**
 * Processes step result and checks residual convergence.
 * Returns convergence result if converged, or best solution found.
 */
function processStepResult(
  updatedParameters: Float64Array,
  updatedStates: Float64Array,
  updatedLambda: number,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  constraintTolerance: number,
  tolResidual: number,
  iteration: number,
  gradientNorm: number,
  logger: Logger
): {
  converged: boolean;
  result?: ConstrainedLevenbergMarquardtResult;
  bestCost: number;
  bestParameters: Float64Array;
  bestStates: Float64Array;
} {
  const currentResidual = residualFunction(updatedParameters, updatedStates);
  const currentResidualNorm = vectorNorm(currentResidual);
  const currentCost = computeSumOfSquaredResiduals(currentResidualNorm);
  const currentConstraint = constraintFunction(updatedParameters, updatedStates);
  const currentConstraintNorm = vectorNorm(currentConstraint);

  if (currentConstraintNorm <= constraintTolerance && checkResidualConvergence(currentResidualNorm, tolResidual, iteration)) {
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
      ),
      bestCost: currentCost,
      bestParameters: updatedParameters,
      bestStates: updatedStates
    };
  }

  return {
    converged: false,
    bestCost: currentCost,
    bestParameters: updatedParameters,
    bestStates: updatedStates
  };
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
  shouldStop?: boolean;
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
  const constraintSatisfied = constraintNorm <= constraintTolerance;

  const gradientConvergenceResult = checkConvergenceForLM(
    gradientNorm,
    constraintSatisfied,
    tolGradient,
    iteration,
    currentParameters,
    currentStates,
    cost,
    residualNorm,
    constraintNorm,
    currentLambda,
    logger
  );

  if (gradientConvergenceResult) {
    return {
      converged: true,
      result: gradientConvergenceResult
    };
  }

  const stepResult = tryStepWithLambda(
    jtj,
    jtr,
    currentParameters,
    currentStates,
    currentLambda,
    lambdaFactor,
    residualFunction,
    constraintFunction,
    cost,
    tolStep,
    iteration,
    stepSizeP,
    stepSizeX,
    constraintTolerance,
    constraintSatisfied,
    gradientNorm,
    residualNorm,
    constraintNorm,
    logger,
    dcdp,
    dcdx
  );

  if (stepResult.shouldStop) {
    return { converged: false, shouldStop: true };
  }

  if (stepResult.stepSizeConverged) {
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
        stepResult.updatedLambda
      )
    };
  }

  if (stepResult.stepAccepted && stepResult.updatedParameters && stepResult.updatedStates) {
    const processResult = processStepResult(
      stepResult.updatedParameters,
      stepResult.updatedStates,
      stepResult.updatedLambda,
      residualFunction,
      constraintFunction,
      constraintTolerance,
      tolResidual,
      iteration,
      gradientNorm,
      logger
    );

    if (processResult.converged && processResult.result) {
      return {
        converged: true,
        result: processResult.result
      };
    }

    return {
      converged: false,
      newParameters: processResult.bestParameters,
      newStates: processResult.bestStates,
      newLambda: stepResult.updatedLambda,
      bestCost: processResult.bestCost,
      bestParameters: processResult.bestParameters,
      bestStates: processResult.bestStates
    };
  }

  return { converged: false, newLambda: stepResult.updatedLambda };
}

/**
 * Initializes state for constrained Levenberg-Marquardt optimization.
 * Tracks best solution found to return if max iterations reached.
 */
function initializeLMState(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  residualFunction: ConstrainedResidualFn
): {
  currentParameters: Float64Array;
  currentStates: Float64Array;
  bestParameters: Float64Array;
  bestStates: Float64Array;
  bestCost: number;
} {
  const currentParameters: Float64Array = new Float64Array(initialParameters);
  const currentStates: Float64Array = new Float64Array(initialStates);
  const bestParameters: Float64Array = new Float64Array(initialParameters);
  const bestStates: Float64Array = new Float64Array(initialStates);
  const initialResidual = residualFunction(initialParameters, initialStates);
  const initialResidualNorm = vectorNorm(initialResidual);
  const bestCost = computeSumOfSquaredResiduals(initialResidualNorm);
  return { currentParameters, currentStates, bestParameters, bestStates, bestCost };
}

/**
 * Runs main iteration loop for constrained Levenberg-Marquardt optimization.
 * Returns result if converged, or final state if max iterations reached.
 */
function runLMIterations(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  effectiveJacobianOptions: EffectiveJacobianOptions,
  tolGradient: number,
  tolStep: number,
  tolResidual: number,
  constraintTolerance: number,
  stepSizeP: number,
  stepSizeX: number,
  lambdaInitial: number,
  lambdaFactor: number,
  maxIterations: number,
  logger: Logger,
  onIteration?: (iteration: number, cost: number, parameters: Float64Array) => void,
  dcdp?: (parameters: Float64Array, states: Float64Array) => Matrix,
  dcdx?: (parameters: Float64Array, states: Float64Array) => Matrix
): {
  result: ConstrainedLevenbergMarquardtResult;
} | {
  bestParameters: Float64Array;
  bestStates: Float64Array;
  bestCost: number;
  currentLambda: number;
  actualIterations: number;
} {
  const state = initializeLMState(initialParameters, initialStates, residualFunction);
  let currentLambda = lambdaInitial;
  let actualIterations = 0;

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    actualIterations = iteration + 1;
    const iterationResult = performConstrainedLevenbergMarquardtIteration(
      state.currentParameters,
      state.currentStates,
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
      dcdp,
      dcdx
    );

    if (iterationResult.converged && iterationResult.result) {
      return { result: iterationResult.result };
    }

    if (iterationResult.shouldStop) {
      break;
    }

    if (iterationResult.bestCost !== undefined && iterationResult.bestCost < state.bestCost) {
      state.bestCost = iterationResult.bestCost;
      if (iterationResult.bestParameters && iterationResult.bestStates) {
        state.bestParameters = new Float64Array(iterationResult.bestParameters);
        state.bestStates = new Float64Array(iterationResult.bestStates);
      }
    }

    if (iterationResult.newLambda !== undefined) {
      currentLambda = iterationResult.newLambda;
    }

    if (iterationResult.newParameters && iterationResult.newStates) {
      state.currentParameters = iterationResult.newParameters as Float64Array;
      state.currentStates = iterationResult.newStates as Float64Array;
    }
  }

  return {
    bestParameters: state.bestParameters,
    bestStates: state.bestStates,
    bestCost: state.bestCost,
    currentLambda,
    actualIterations
  };
}

/**
 * Creates final result when max iterations reached or optimization stopped.
 * Computes final gradient norm for diagnostic purposes.
 */
function createFinalLMResult(
  bestParameters: Float64Array,
  bestStates: Float64Array,
  bestCost: number,
  currentLambda: number,
  actualIterations: number,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  effectiveJacobianOptions: EffectiveJacobianOptions,
  logger: Logger
): ConstrainedLevenbergMarquardtResult {
  const finalResidual = residualFunction(bestParameters, bestStates);
  const finalResidualNorm = vectorNorm(finalResidual);
  const finalConstraint = constraintFunction(bestParameters, bestStates);
  const finalConstraintNorm = vectorNorm(finalConstraint);

  const finalEffectiveJacobian = computeEffectiveJacobian(
    bestParameters,
    bestStates,
    residualFunction,
    constraintFunction,
    effectiveJacobianOptions,
    logger,
    'constrainedLevenbergMarquardt'
  );
  const { jtr: finalJtr } = computeNormalEquationsMatrices(finalEffectiveJacobian, finalResidual);
  const finalGradientNorm = vectorNorm(matrixToFloat64Array(finalJtr));

  return createConvergenceResultForLM(
    bestParameters,
    bestStates,
    actualIterations - 1,
    false,
    bestCost,
    finalGradientNorm,
    finalResidualNorm,
    finalConstraintNorm,
    currentLambda
  );
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

  const iterationResult = runLMIterations(
    initialParameters,
    initialStates,
    residualFunction,
    constraintFunction,
    effectiveJacobianOptions,
    tolGradient,
    tolStep,
    tolResidual,
    constraintTolerance,
    stepSizeP,
    stepSizeX,
    lambdaInitial,
    lambdaFactor,
    maxIterations,
    logger,
    onIteration,
    options.dcdp,
    options.dcdx
  );

  if ('result' in iterationResult) {
    return iterationResult.result;
  }

  const finalResidual = residualFunction(iterationResult.bestParameters, iterationResult.bestStates);
  const finalResidualNorm = vectorNorm(finalResidual);
  const finalConstraint = constraintFunction(iterationResult.bestParameters, iterationResult.bestStates);
  const finalConstraintNorm = vectorNorm(finalConstraint);

  logger.warn('constrainedLevenbergMarquardt', undefined, 'Maximum iterations reached', [
    { key: 'Iterations:', value: iterationResult.actualIterations },
    { key: 'Final cost:', value: iterationResult.bestCost },
    {
      key: 'Final gradient norm:', value: vectorNorm(matrixToFloat64Array(computeNormalEquationsMatrices(
        computeEffectiveJacobian(
          iterationResult.bestParameters,
          iterationResult.bestStates,
          residualFunction,
          constraintFunction,
          effectiveJacobianOptions,
          logger,
          'constrainedLevenbergMarquardt'
        ),
        finalResidual
      ).jtr))
    },
    { key: 'Final residual norm:', value: finalResidualNorm },
    { key: 'Final constraint norm:', value: finalConstraintNorm },
    { key: 'Final lambda:', value: iterationResult.currentLambda }
  ]);

  return createFinalLMResult(
    iterationResult.bestParameters,
    iterationResult.bestStates,
    iterationResult.bestCost,
    iterationResult.currentLambda,
    iterationResult.actualIterations,
    residualFunction,
    constraintFunction,
    effectiveJacobianOptions,
    logger
  );
}
