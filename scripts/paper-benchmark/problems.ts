/**
 * Classic problems and solver cases for the paper benchmark.
 * Known optima only. Shared solver tolerances; success is not `converged`.
 */

import { Matrix } from 'ml-matrix';
import {
  adjointBfgs,
  adjointGradientDescent,
  bfgs,
  cmaEs,
  constrainedGaussNewton,
  constrainedLevenbergMarquardt,
  gaussNewton,
  gradientDescent,
  lbfgs,
  levenbergMarquardt
} from '../../src/index';
import { vectorNorm } from '../../src/utils/matrix';
import {
  buildPenaltyResidual,
  computeParameterError,
  concatParameterAndState,
  splitParameterAndState,
  timePaperTrial,
  wrapConstrainedCostFunction,
  wrapConstrainedResidualFunction,
  wrapConstraintFunction,
  wrapCostFunction,
  wrapGradientFunction,
  wrapResidualFunction
} from '../benchmark-harness';
import {
  CHAIN_ADJOINT_DECISION_VARIABLE_COUNT,
  CHAIN_PENALTY_DECISION_VARIABLE_COUNT,
  CHAIN_STATE_COUNT,
  CMA_ES_INITIAL_STEP_SIZE,
  CMA_ES_MAX_ITERATIONS,
  CMA_ES_ROSENBROCK_MAX_ITERATIONS,
  CMA_ES_ROSENBROCK_SUCCESS_TOLERANCE,
  CMA_ES_ROSENBROCK_TARGET_COST,
  CMA_ES_SEED_COUNT,
  CMA_ES_WARMUP_SEED,
  DEFAULT_MAX_ITERATIONS,
  DETERMINISTIC_REPEAT_COUNT,
  EXPONENTIAL_SUCCESS_PARAMETER_TOLERANCE,
  METHOD_ADJOINT_BFGS,
  METHOD_ADJOINT_GD,
  METHOD_BFGS,
  METHOD_CMA_ES,
  METHOD_CONSTRAINED_GN,
  METHOD_CONSTRAINED_LM,
  METHOD_GAUSS_NEWTON,
  METHOD_GRADIENT_DESCENT,
  METHOD_LBFGS,
  METHOD_LEVENBERG_MARQUARDT,
  METHOD_PENALTY_GN,
  METHOD_PENALTY_LM,
  PROBLEM_CHAIN,
  PROBLEM_CIRCLE,
  PROBLEM_EXPONENTIAL_FIT,
  PROBLEM_LINEAR_LEAST_SQUARES,
  PROBLEM_ROSENBROCK,
  PROBLEM_SPHERE,
  ROSENBROCK_GRADIENT_DESCENT_MAX_ITERATIONS,
  SHARED_SOLVER_TOLERANCE,
  SUCCESS_CONSTRAINT_TOLERANCE,
  SUCCESS_PARAMETER_TOLERANCE
} from './constants';
import type { PaperCase } from './types';

const SPHERE_DIMENSION = 10;
const LEAST_SQUARES_DIMENSION = 2;
const ROSENBROCK_A = 1;
const ROSENBROCK_B = 100;
const CHAIN_TARGET_END = 1;
const CHAIN_EXPECTED_PARAMETER = CHAIN_TARGET_END / CHAIN_STATE_COUNT;
const CHAIN_PENALTY_WEIGHT = 1e4;
const CHAIN_INITIAL_PARAMETER = 0.2;
const CIRCLE_INITIAL_PARAMETER = 0.2;
const CIRCLE_RADIUS_SQUARED = 2;
const CIRCLE_EXPECTED_PARAMETER = 1;
const CIRCLE_EXPECTED_STATE = 1;
const CIRCLE_DECISION_VARIABLE_COUNT = 1;
const LINEAR_SLOPE = 2;
const LINEAR_INTERCEPT = 0;
const EXPONENTIAL_A = 1;
const EXPONENTIAL_B = 1;
const EXPONENTIAL_X = new Float64Array([0, 1, 2, 3, 4]);
const EXPONENTIAL_Y = new Float64Array([1.0, 2.7, 7.4, 20.1, 54.6]);
const LINEAR_X = new Float64Array([1, 2, 3, 4, 5]);
const LINEAR_Y = new Float64Array([2.0, 4.0, 6.0, 8.0, 10.0]);

function sphereInitial(): Float64Array {
  return new Float64Array(SPHERE_DIMENSION).fill(1);
}

function sphereExpected(): Float64Array {
  return new Float64Array(SPHERE_DIMENSION);
}

function sphereCost(parameters: Float64Array): number {
  let sum = 0;
  for (const value of parameters) {
    sum += value * value;
  }
  return sum;
}

function sphereGradient(parameters: Float64Array): Float64Array {
  const gradient = new Float64Array(parameters.length);
  for (let index = 0; index < parameters.length; index++) {
    gradient[index] = 2 * parameters[index];
  }
  return gradient;
}

function rosenbrockInitial(): Float64Array {
  return new Float64Array([-ROSENBROCK_A, ROSENBROCK_A]);
}

function rosenbrockExpected(): Float64Array {
  return new Float64Array([ROSENBROCK_A, ROSENBROCK_A]);
}

function rosenbrockCost(parameters: Float64Array): number {
  const x = parameters[0];
  const y = parameters[1];
  const term1 = ROSENBROCK_A - x;
  const term2 = y - x * x;
  return term1 * term1 + ROSENBROCK_B * term2 * term2;
}

function rosenbrockGradient(parameters: Float64Array): Float64Array {
  const x = parameters[0];
  const y = parameters[1];
  const valley = y - x * x;
  return new Float64Array([
    -2 * (ROSENBROCK_A - x) - 4 * ROSENBROCK_B * x * valley,
    2 * ROSENBROCK_B * valley
  ]);
}

function exponentialResidual(parameters: Float64Array): Float64Array {
  const residual = new Float64Array(EXPONENTIAL_X.length);
  for (let index = 0; index < EXPONENTIAL_X.length; index++) {
    residual[index] = parameters[0] * Math.exp(parameters[1] * EXPONENTIAL_X[index]) - EXPONENTIAL_Y[index];
  }
  return residual;
}

function linearResidual(parameters: Float64Array): Float64Array {
  const residual = new Float64Array(LINEAR_X.length);
  for (let index = 0; index < LINEAR_X.length; index++) {
    residual[index] = parameters[0] * LINEAR_X[index] + parameters[1] - LINEAR_Y[index];
  }
  return residual;
}

function circleResidual(parameters: Float64Array, states: Float64Array): Float64Array {
  return new Float64Array([parameters[0] - CIRCLE_EXPECTED_PARAMETER, states[0] - CIRCLE_EXPECTED_STATE]);
}

function circleConstraint(parameters: Float64Array, states: Float64Array): Float64Array {
  return new Float64Array([
    parameters[0] * parameters[0] + states[0] * states[0] - CIRCLE_RADIUS_SQUARED
  ]);
}

function circleInitialStates(): Float64Array {
  return new Float64Array([
    Math.sqrt(CIRCLE_RADIUS_SQUARED - CIRCLE_INITIAL_PARAMETER * CIRCLE_INITIAL_PARAMETER)
  ]);
}

function circleSolutionError(parameters: Float64Array, states: Float64Array): number {
  return computeParameterError(
    new Float64Array([parameters[0], states[0]]),
    new Float64Array([CIRCLE_EXPECTED_PARAMETER, CIRCLE_EXPECTED_STATE])
  );
}

function chainCost(_parameters: Float64Array, states: Float64Array): number {
  const last = states[CHAIN_STATE_COUNT - 1] - CHAIN_TARGET_END;
  return last * last;
}

function chainResidual(_parameters: Float64Array, states: Float64Array): Float64Array {
  return new Float64Array([states[CHAIN_STATE_COUNT - 1] - CHAIN_TARGET_END]);
}

function chainConstraint(parameters: Float64Array, states: Float64Array): Float64Array {
  const constraint = new Float64Array(CHAIN_STATE_COUNT);
  constraint[0] = states[0] - parameters[0];
  for (let index = 1; index < CHAIN_STATE_COUNT; index++) {
    constraint[index] = states[index] - states[index - 1] - parameters[0];
  }
  return constraint;
}

function chainInitialStates(): Float64Array {
  const states = new Float64Array(CHAIN_STATE_COUNT);
  for (let index = 0; index < CHAIN_STATE_COUNT; index++) {
    states[index] = (index + 1) * CHAIN_INITIAL_PARAMETER;
  }
  return states;
}

function circleAdjointPartials(parameters: Float64Array, states: Float64Array) {
  return {
    dfdp: new Float64Array([parameters[0] - CIRCLE_EXPECTED_PARAMETER]),
    dfdx: new Float64Array([states[0] - CIRCLE_EXPECTED_STATE]),
    dcdp: new Matrix([[2 * parameters[0]]]),
    dcdx: new Matrix([[2 * states[0]]])
  };
}

function chainAdjointPartials(_parameters: Float64Array, states: Float64Array) {
  const dcdpData = Array.from({ length: CHAIN_STATE_COUNT }, () => [-1]);
  const dcdxData = Array.from({ length: CHAIN_STATE_COUNT }, (_, row) => {
    const rowValues = new Array<number>(CHAIN_STATE_COUNT).fill(0);
    rowValues[row] = 1;
    if (row > 0) {
      rowValues[row - 1] = -1;
    }
    return rowValues;
  });
  const dfdx = new Float64Array(CHAIN_STATE_COUNT);
  dfdx[CHAIN_STATE_COUNT - 1] = 2 * (states[CHAIN_STATE_COUNT - 1] - CHAIN_TARGET_END);
  return {
    dfdp: new Float64Array([0]),
    dfdx,
    dcdp: new Matrix(dcdpData),
    dcdx: new Matrix(dcdxData)
  };
}

function sharedLeastSquaresOptions() {
  return {
    maxIterations: DEFAULT_MAX_ITERATIONS,
    tolerance: SHARED_SOLVER_TOLERANCE,
    tolGradient: SHARED_SOLVER_TOLERANCE,
    tolStep: SHARED_SOLVER_TOLERANCE,
    tolResidual: SHARED_SOLVER_TOLERANCE,
    useNumericJacobian: true
  };
}

function sharedConstrainedOptions() {
  return {
    ...sharedLeastSquaresOptions(),
    constraintTolerance: SHARED_SOLVER_TOLERANCE
  };
}

function analyticalAdjointDerivativeOptions(
  computePartials: (
    parameters: Float64Array,
    states: Float64Array
  ) => {
    dfdp: Float64Array;
    dfdx: Float64Array;
    dcdp: Matrix;
    dcdx: Matrix;
  }
) {
  return {
    ...sharedConstrainedOptions(),
    useLineSearch: true,
    dfdp: (parameters: Float64Array, states: Float64Array) => computePartials(parameters, states).dfdp,
    dfdx: (parameters: Float64Array, states: Float64Array) => computePartials(parameters, states).dfdx,
    dcdp: (parameters: Float64Array, states: Float64Array) => computePartials(parameters, states).dcdp,
    dcdx: (parameters: Float64Array, states: Float64Array) => computePartials(parameters, states).dcdx
  };
}

function createScalarSolverCase(
  problemName: string,
  methodName: string,
  maxIterations: number,
  initial: () => Float64Array,
  expected: () => Float64Array,
  cost: (parameters: Float64Array) => number,
  gradient: (parameters: Float64Array) => Float64Array,
  solve: typeof gradientDescent
): PaperCase {
  return {
    problemName,
    methodName,
    parameterTolerance: SUCCESS_PARAMETER_TOLERANCE,
    repeatCount: DETERMINISTIC_REPEAT_COUNT,
    runTrial: () =>
      timePaperTrial((counters) => {
        const start = initial();
        const result = solve(
          start,
          wrapCostFunction(cost, counters),
          wrapGradientFunction(gradient, counters),
          { maxIterations, tolerance: SHARED_SOLVER_TOLERANCE, useLineSearch: true }
        );
        return {
          iterations: result.iterations,
          finalCost: result.finalCost,
          parameterError: computeParameterError(result.finalParameters, expected()),
          decisionVariableCount: start.length
        };
      })
  };
}

function createCmaEsCase(
  problemName: string,
  parameterTolerance: number,
  initial: () => Float64Array,
  expected: () => Float64Array,
  cost: (parameters: Float64Array) => number,
  targetCost: number,
  extraOptions: { maxIterations?: number; restartStrategy?: 'none' | 'ipop' } = {}
): PaperCase {
  return {
    problemName,
    methodName: METHOD_CMA_ES,
    parameterTolerance,
    repeatCount: CMA_ES_SEED_COUNT,
    runTrial: (repeatIndex) =>
      timePaperTrial((counters) => {
        const start = initial();
        const result = cmaEs(start, wrapCostFunction(cost, counters), {
          maxIterations: CMA_ES_MAX_ITERATIONS,
          targetCost,
          randomSeed: repeatIndex < 0 ? CMA_ES_WARMUP_SEED : repeatIndex + 1,
          initialStepSize: CMA_ES_INITIAL_STEP_SIZE,
          ...extraOptions
        });
        return {
          iterations: result.iterations,
          finalCost: result.finalCost,
          parameterError: computeParameterError(result.finalParameters, expected()),
          decisionVariableCount: start.length
        };
      })
  };
}

function createLeastSquaresCase(
  problemName: string,
  methodName: string,
  parameterTolerance: number,
  initial: Float64Array,
  expected: Float64Array,
  residual: (parameters: Float64Array) => Float64Array,
  solve: typeof gaussNewton | typeof levenbergMarquardt
): PaperCase {
  return {
    problemName,
    methodName,
    parameterTolerance,
    repeatCount: DETERMINISTIC_REPEAT_COUNT,
    runTrial: () =>
      timePaperTrial((counters) => {
        const result = solve(
          new Float64Array(initial),
          wrapResidualFunction(residual, counters),
          sharedLeastSquaresOptions()
        );
        return {
          iterations: result.iterations,
          finalCost: result.finalCost,
          parameterError: computeParameterError(result.finalParameters, expected),
          decisionVariableCount: initial.length
        };
      })
  };
}

function createCircleAdjointCase(
  methodName: string,
  solve: typeof adjointGradientDescent | typeof adjointBfgs
): PaperCase {
  return {
    problemName: PROBLEM_CIRCLE,
    methodName,
    parameterTolerance: SUCCESS_PARAMETER_TOLERANCE,
    constraintTolerance: SUCCESS_CONSTRAINT_TOLERANCE,
    repeatCount: DETERMINISTIC_REPEAT_COUNT,
    runTrial: () =>
      timePaperTrial((counters) => {
        const initialParameters = new Float64Array([CIRCLE_INITIAL_PARAMETER]);
        const initialStates = circleInitialStates();
        const result = solve(
          initialParameters,
          initialStates,
          wrapConstrainedResidualFunction(circleResidual, counters),
          wrapConstraintFunction(circleConstraint, counters),
          analyticalAdjointDerivativeOptions(circleAdjointPartials)
        );
        return {
          iterations: result.iterations,
          finalCost: result.finalCost,
          parameterError: circleSolutionError(result.finalParameters, result.finalStates),
          constraintNorm:
            result.finalConstraintNorm ??
            vectorNorm(circleConstraint(result.finalParameters, result.finalStates)),
          decisionVariableCount: CIRCLE_DECISION_VARIABLE_COUNT
        };
      })
  };
}

function createCircleConstrainedCase(
  methodName: string,
  solve: typeof constrainedGaussNewton | typeof constrainedLevenbergMarquardt
): PaperCase {
  return {
    problemName: PROBLEM_CIRCLE,
    methodName,
    parameterTolerance: SUCCESS_PARAMETER_TOLERANCE,
    constraintTolerance: SUCCESS_CONSTRAINT_TOLERANCE,
    repeatCount: DETERMINISTIC_REPEAT_COUNT,
    runTrial: () =>
      timePaperTrial((counters) => {
        const result = solve(
          new Float64Array([CIRCLE_INITIAL_PARAMETER]),
          circleInitialStates(),
          wrapConstrainedResidualFunction(circleResidual, counters),
          wrapConstraintFunction(circleConstraint, counters),
          sharedConstrainedOptions()
        );
        return {
          iterations: result.iterations,
          finalCost: result.finalCost,
          parameterError: circleSolutionError(result.finalParameters, result.finalStates),
          constraintNorm:
            result.finalConstraintNorm ??
            vectorNorm(circleConstraint(result.finalParameters, result.finalStates)),
          decisionVariableCount: CIRCLE_DECISION_VARIABLE_COUNT
        };
      })
  };
}

function createChainAdjointCase(
  methodName: string,
  solve: typeof adjointGradientDescent | typeof adjointBfgs
): PaperCase {
  return {
    problemName: PROBLEM_CHAIN,
    methodName,
    parameterTolerance: SUCCESS_PARAMETER_TOLERANCE,
    constraintTolerance: SUCCESS_CONSTRAINT_TOLERANCE,
    repeatCount: DETERMINISTIC_REPEAT_COUNT,
    runTrial: () =>
      timePaperTrial((counters) => {
        const result = solve(
          new Float64Array([CHAIN_INITIAL_PARAMETER]),
          chainInitialStates(),
          wrapConstrainedCostFunction(chainCost, counters),
          wrapConstraintFunction(chainConstraint, counters),
          analyticalAdjointDerivativeOptions(chainAdjointPartials)
        );
        return {
          iterations: result.iterations,
          finalCost: result.finalCost,
          parameterError: Math.abs(result.finalParameters[0] - CHAIN_EXPECTED_PARAMETER),
          constraintNorm:
            result.finalConstraintNorm ??
            vectorNorm(chainConstraint(result.finalParameters, result.finalStates)),
          decisionVariableCount: CHAIN_ADJOINT_DECISION_VARIABLE_COUNT
        };
      })
  };
}

function createChainPenaltyCase(
  methodName: string,
  solve: typeof gaussNewton | typeof levenbergMarquardt
): PaperCase {
  return {
    problemName: PROBLEM_CHAIN,
    methodName,
    parameterTolerance: SUCCESS_PARAMETER_TOLERANCE,
    constraintTolerance: SUCCESS_CONSTRAINT_TOLERANCE,
    repeatCount: DETERMINISTIC_REPEAT_COUNT,
    runTrial: () =>
      timePaperTrial((counters) => {
        const countedResidual = wrapConstrainedResidualFunction(chainResidual, counters);
        const countedConstraint = wrapConstraintFunction(chainConstraint, counters);
        const penaltyResidual = buildPenaltyResidual({
          parameterCount: CHAIN_ADJOINT_DECISION_VARIABLE_COUNT,
          residual: countedResidual,
          constraint: countedConstraint,
          penaltyWeight: CHAIN_PENALTY_WEIGHT
        });
        const result = solve(
          concatParameterAndState(new Float64Array([CHAIN_INITIAL_PARAMETER]), chainInitialStates()),
          penaltyResidual,
          sharedLeastSquaresOptions()
        );
        const split = splitParameterAndState(result.finalParameters, CHAIN_ADJOINT_DECISION_VARIABLE_COUNT);
        return {
          iterations: result.iterations,
          finalCost: chainCost(split.parameters, split.states),
          parameterError: Math.abs(split.parameters[0] - CHAIN_EXPECTED_PARAMETER),
          constraintNorm: vectorNorm(chainConstraint(split.parameters, split.states)),
          decisionVariableCount: CHAIN_PENALTY_DECISION_VARIABLE_COUNT
        };
      })
  };
}

function createSphereCases(): PaperCase[] {
  return [
    createScalarSolverCase(
      PROBLEM_SPHERE,
      METHOD_GRADIENT_DESCENT,
      DEFAULT_MAX_ITERATIONS,
      sphereInitial,
      sphereExpected,
      sphereCost,
      sphereGradient,
      gradientDescent
    ),
    createScalarSolverCase(
      PROBLEM_SPHERE,
      METHOD_BFGS,
      DEFAULT_MAX_ITERATIONS,
      sphereInitial,
      sphereExpected,
      sphereCost,
      sphereGradient,
      bfgs
    ),
    createScalarSolverCase(
      PROBLEM_SPHERE,
      METHOD_LBFGS,
      DEFAULT_MAX_ITERATIONS,
      sphereInitial,
      sphereExpected,
      sphereCost,
      sphereGradient,
      lbfgs
    ),
    createCmaEsCase(
      PROBLEM_SPHERE,
      SUCCESS_PARAMETER_TOLERANCE,
      sphereInitial,
      sphereExpected,
      sphereCost,
      SHARED_SOLVER_TOLERANCE
    )
  ];
}

function createRosenbrockCases(): PaperCase[] {
  return [
    createScalarSolverCase(
      PROBLEM_ROSENBROCK,
      METHOD_GRADIENT_DESCENT,
      ROSENBROCK_GRADIENT_DESCENT_MAX_ITERATIONS,
      rosenbrockInitial,
      rosenbrockExpected,
      rosenbrockCost,
      rosenbrockGradient,
      gradientDescent
    ),
    createScalarSolverCase(
      PROBLEM_ROSENBROCK,
      METHOD_BFGS,
      DEFAULT_MAX_ITERATIONS,
      rosenbrockInitial,
      rosenbrockExpected,
      rosenbrockCost,
      rosenbrockGradient,
      bfgs
    ),
    createScalarSolverCase(
      PROBLEM_ROSENBROCK,
      METHOD_LBFGS,
      DEFAULT_MAX_ITERATIONS,
      rosenbrockInitial,
      rosenbrockExpected,
      rosenbrockCost,
      rosenbrockGradient,
      lbfgs
    ),
    createCmaEsCase(
      PROBLEM_ROSENBROCK,
      CMA_ES_ROSENBROCK_SUCCESS_TOLERANCE,
      rosenbrockInitial,
      rosenbrockExpected,
      rosenbrockCost,
      CMA_ES_ROSENBROCK_TARGET_COST,
      {
        maxIterations: CMA_ES_ROSENBROCK_MAX_ITERATIONS,
        restartStrategy: 'ipop'
      }
    )
  ];
}

function createLeastSquaresCases(): PaperCase[] {
  return [
    createLeastSquaresCase(
      PROBLEM_LINEAR_LEAST_SQUARES,
      METHOD_GAUSS_NEWTON,
      SUCCESS_PARAMETER_TOLERANCE,
      new Float64Array(LEAST_SQUARES_DIMENSION),
      new Float64Array([LINEAR_SLOPE, LINEAR_INTERCEPT]),
      linearResidual,
      gaussNewton
    ),
    createLeastSquaresCase(
      PROBLEM_LINEAR_LEAST_SQUARES,
      METHOD_LEVENBERG_MARQUARDT,
      SUCCESS_PARAMETER_TOLERANCE,
      new Float64Array(LEAST_SQUARES_DIMENSION),
      new Float64Array([LINEAR_SLOPE, LINEAR_INTERCEPT]),
      linearResidual,
      levenbergMarquardt
    ),
    createLeastSquaresCase(
      PROBLEM_EXPONENTIAL_FIT,
      METHOD_GAUSS_NEWTON,
      EXPONENTIAL_SUCCESS_PARAMETER_TOLERANCE,
      new Float64Array([EXPONENTIAL_A, EXPONENTIAL_B]),
      new Float64Array([EXPONENTIAL_A, EXPONENTIAL_B]),
      exponentialResidual,
      gaussNewton
    ),
    createLeastSquaresCase(
      PROBLEM_EXPONENTIAL_FIT,
      METHOD_LEVENBERG_MARQUARDT,
      EXPONENTIAL_SUCCESS_PARAMETER_TOLERANCE,
      new Float64Array([EXPONENTIAL_A, EXPONENTIAL_B]),
      new Float64Array([EXPONENTIAL_A, EXPONENTIAL_B]),
      exponentialResidual,
      levenbergMarquardt
    )
  ];
}

export function createAllCases(): PaperCase[] {
  return [
    ...createSphereCases(),
    ...createRosenbrockCases(),
    ...createLeastSquaresCases(),
    createCircleAdjointCase(METHOD_ADJOINT_GD, adjointGradientDescent),
    createCircleAdjointCase(METHOD_ADJOINT_BFGS, adjointBfgs),
    createCircleConstrainedCase(METHOD_CONSTRAINED_GN, constrainedGaussNewton),
    createCircleConstrainedCase(METHOD_CONSTRAINED_LM, constrainedLevenbergMarquardt),
    createChainAdjointCase(METHOD_ADJOINT_GD, adjointGradientDescent),
    createChainAdjointCase(METHOD_ADJOINT_BFGS, adjointBfgs),
    createChainPenaltyCase(METHOD_PENALTY_GN, gaussNewton),
    createChainPenaltyCase(METHOD_PENALTY_LM, levenbergMarquardt)
  ];
}
