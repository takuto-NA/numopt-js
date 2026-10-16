/** Shared public-LM fault injection and fatal-error checks for Node and Chromium. */
const KIBO_SUCCESS = 0;
const KIBO_CAPACITY_ERROR = 4;
const KIBO_NON_POSITIVE_PIVOT = 8;
const KIBO_ARITHMETIC_FAILURE = 10;
const INCOMPATIBLE_ABI = 2;
const TARGET_PARAMETER = 2;
const PARAMETER_TOLERANCE = 1e-6;
const MAXIMUM_ITERATIONS = 10;
const TINY_JACOBIAN = 1e-160;
const LARGE_RESIDUAL = 1e154;
const TINY_INITIAL_LAMBDA = 1e-320;

function ensure(condition, message) { if (!condition) throw new Error(message); }
function scalarMatrix(api, value) { return api.float64ArrayToMatrix(new Float64Array([value])); }
function scalarResidual(parameters) { return new Float64Array([parameters[0] - TARGET_PARAMETER]); }

function runScalarProblem(api, solver, residual = scalarResidual, options = {}) {
  return api.levenbergMarquardt(new Float64Array([0]), residual, {
    jacobian: () => scalarMatrix(api, 1), maxIterations: MAXIMUM_ITERATIONS, linearSolver: solver, ...options
  });
}

// Only external result/status faults are injected; normal fit scenarios require real WASM success.
function overrideWorkspace(workspace, overrides) {
  return {
    rows: workspace.rows, cols: workspace.cols, module: workspace.module,
    get allocatedNumericBytes() { return workspace.allocatedNumericBytes; },
    get disposed() { return workspace.disposed; },
    solveLlt: (...inputs) => workspace.solveLlt(...inputs),
    dispose: () => workspace.dispose(), ...overrides
  };
}

function runConfigurationCheck(api, wasmModule, LinearAlgebra, specification) {
  const workspace = new LinearAlgebra(wasmModule, 1, 1);
  let solver, rejected = false, statistics;
  try {
    solver = api.createKiboLinearSolver(overrideWorkspace(workspace, specification.overrides ?? {}));
    runScalarProblem(api, solver, specification.residual);
  } catch (error) {
    ensure(error instanceof api.LinearSolverConfigurationError, `${specification.name}: wrong error: ${error}`);
    rejected = true;
  } finally {
    statistics = solver?.statistics;
    solver?.dispose(); workspace.dispose();
  }
  ensure(rejected && workspace.disposed, `${specification.name}: configuration error or cleanup was hidden`);
  ensure(!statistics || (statistics.fallbackAttempts === 0 && statistics.solveAttempts <= 1),
    `${specification.name}: configuration error entered LU fallback or LM retries`);
  return { name: specification.name, rejected, disposed: workspace.disposed, statistics };
}

export function runConfigurationChecks(api, wasmModule, LinearAlgebra) {
  const badOutputs = [new Float64Array([NaN]), new Float64Array([Infinity]), new Float64Array([])];
  const specifications = [
    { name: 'non-finite-input', residual: () => new Float64Array([NaN]) },
    { name: 'shape-mismatch', overrides: { rows: 2, cols: 2 } },
    { name: 'capacity-error', overrides: { solveLlt: () => ({ status: KIBO_CAPACITY_ERROR }) } },
    { name: 'abi-mismatch', overrides: { module: { _kibo_abi_version: () => INCOMPATIBLE_ABI } } },
    { name: 'failed-wasm-call', overrides: { solveLlt() { throw new Error('WASM instance failed'); } } },
    ...badOutputs.map((output, index) => ({
      name: `invalid-output-${index}`, overrides: { solveLlt: () => ({ status: KIBO_SUCCESS, output }) }
    }))
  ];
  return specifications.map(specification => runConfigurationCheck(api, wasmModule, LinearAlgebra, specification));
}

function runFallbackProblem(api, wasmModule, LinearAlgebra, status, forceOverflow) {
  const workspace = new LinearAlgebra(wasmModule, 1, 1);
  const solver = api.createKiboLinearSolver(overrideWorkspace(workspace, { solveLlt: () => ({ status }) }));
  try {
    const result = forceOverflow
      ? runScalarProblem(api, solver, () => new Float64Array([LARGE_RESIDUAL]), {
        jacobian: () => scalarMatrix(api, TINY_JACOBIAN), lambdaInitial: TINY_INITIAL_LAMBDA,
        tolGradient: 0, tolStep: 0
      })
      : runScalarProblem(api, solver);
    const statistics = solver.statistics;
    ensure(statistics.fallbackAttempts > 0 && statistics.fallbackAttempts === statistics.solveAttempts &&
      statistics.failureReasons[`kibo-status-${status}`] === statistics.fallbackAttempts,
    'injected numeric status must pass through LU');
    if (forceOverflow) {
      ensure(!result.converged && result.finalParameters[0] === 0 &&
        result.finalLambda > TINY_INITIAL_LAMBDA && statistics.solveAttempts > statistics.solveSuccesses,
      'overflowing LU must retry with increased lambda and finish without convergence');
    } else {
      ensure(result.converged && Math.abs(result.finalParameters[0] - TARGET_PARAMETER) <= PARAMETER_TOLERANCE &&
        statistics.solveSuccesses > 0, 'public LM must continue after successful LU fallback');
    }
    return { injectedStatus: status, forceOverflow, converged: result.converged,
      finalParameter: result.finalParameters[0], finalLambda: result.finalLambda, statistics };
  } finally { solver.dispose(); }
}

export function runFallbackChecks(api, wasmModule, LinearAlgebra) {
  return [
    runFallbackProblem(api, wasmModule, LinearAlgebra, KIBO_NON_POSITIVE_PIVOT, false),
    runFallbackProblem(api, wasmModule, LinearAlgebra, KIBO_ARITHMETIC_FAILURE, false),
    runFallbackProblem(api, wasmModule, LinearAlgebra, KIBO_ARITHMETIC_FAILURE, true)
  ];
}
