/**
 * Shared labels and protocol constants for the paper benchmark suite.
 */

export const SHARED_SOLVER_TOLERANCE = 1e-8;
export const SUCCESS_PARAMETER_TOLERANCE = 1e-3;
export const SUCCESS_CONSTRAINT_TOLERANCE = 1e-6;
export const EXPONENTIAL_SUCCESS_PARAMETER_TOLERANCE = 0.2;
export const CMA_ES_ROSENBROCK_SUCCESS_TOLERANCE = 1e-2;
export const WARMUP_RUN_COUNT = 1;
export const DETERMINISTIC_REPEAT_COUNT = 7;
export const CMA_ES_SEED_COUNT = 5;
export const CMA_ES_WARMUP_SEED = 1;
export const DEFAULT_MAX_ITERATIONS = 500;
export const ROSENBROCK_GRADIENT_DESCENT_MAX_ITERATIONS = 10000;
export const CMA_ES_MAX_ITERATIONS = 400;
export const CMA_ES_ROSENBROCK_MAX_ITERATIONS = 800;
export const CMA_ES_INITIAL_STEP_SIZE = 1;
export const CMA_ES_ROSENBROCK_TARGET_COST = 1e-10;

export const PROBLEM_SPHERE = 'Sphere n=10';
export const PROBLEM_ROSENBROCK = 'Rosenbrock n=2';
export const PROBLEM_LINEAR_LEAST_SQUARES = 'Linear least squares';
export const PROBLEM_EXPONENTIAL_FIT = 'Exponential fit';
export const PROBLEM_CIRCLE = 'Circle simple residual';
export const PROBLEM_CHAIN = 'Reduced-space chain';

export const METHOD_GRADIENT_DESCENT = 'Gradient Descent';
export const METHOD_BFGS = 'BFGS';
export const METHOD_LBFGS = 'L-BFGS';
export const METHOD_CMA_ES = 'CMA-ES';
export const METHOD_GAUSS_NEWTON = 'Gauss-Newton';
export const METHOD_LEVENBERG_MARQUARDT = 'Levenberg-Marquardt';
export const METHOD_ADJOINT_GD = 'Adjoint GD';
export const METHOD_ADJOINT_BFGS = 'Adjoint BFGS';
export const METHOD_CONSTRAINED_GN = 'Constrained GN';
export const METHOD_CONSTRAINED_LM = 'Constrained LM';
export const METHOD_PENALTY_GN = 'Penalty GN';
export const METHOD_PENALTY_LM = 'Penalty LM';

export const CHAIN_STATE_COUNT = 20;
export const CHAIN_ADJOINT_DECISION_VARIABLE_COUNT = 1;
export const CHAIN_PENALTY_DECISION_VARIABLE_COUNT =
  CHAIN_ADJOINT_DECISION_VARIABLE_COUNT + CHAIN_STATE_COUNT;

export const HYPOTHESIS_PASS_LABEL = 'PASS';
export const HYPOTHESIS_FAIL_LABEL = 'FAIL';

export const OUTPUT_DIRECTORY_NAME = 'benchmark-results';
export const OUTPUT_BASENAME = 'paper-benchmark';
