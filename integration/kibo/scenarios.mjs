/** Shared real-consumer LM problems, acceptance checks and comparison records. */
const PARAMETER_TOLERANCE = 1e-6;
const COST_TOLERANCE = 1e-12;
const SOLUTION_TOLERANCE = 1e-14;
const GRADIENT_TOLERANCE = 1e-6;
const MAXIMUM_ITERATIONS = 100;
const MEDIUM_COLUMNS = 32;
const MEDIUM_ROWS = 128;
const FIXTURE_SEED = 0x6b69626f;
const UINT32_RANGE = 0x100000000;
const RANDOM_LEFT_SHIFT = 13;
const RANDOM_RIGHT_SHIFT = 17;
const RANDOM_FINAL_SHIFT = 5;
const RANDOM_UNIT_MIDPOINT = 0.5;
const PERTURBATION_NORM_BOUND = 0.5;
const DIAGONAL_STRENGTH = 2;
const MEMORY_GROWTH_BYTES = 20 * 1024 * 1024;

function ensure(condition, message) { if (!condition) throw new Error(message); }
function maximumError(actual, expected) {
  return Math.max(...Array.from(actual, (value, index) => Math.abs(value - expected[index])));
}

export function smallProblems(api) {
  const lineInputs = [1, 2, 3, 4, 5];
  const lineValues = [2.1, 3.9, 6.1, 8, 9.9];
  const exponentialInputs = [0, 1, 2, 3, 4];
  const exponentialValues = [1, 2.7, 7.4, 20.1, 54.6];
  return [
    {
      name: 'line', initial: [0, 0], expected: [1.97, 0.09], tolerance: PARAMETER_TOLERANCE,
      maximumCost: 0.1,
      residual: parameters => Float64Array.from(lineInputs, (value, index) =>
        parameters[0] * value + parameters[1] - lineValues[index])
    },
    {
      name: 'exponential', initial: [1, 1], expected: [1, 1], tolerance: 0.2,
      maximumCost: 0.1,
      residual: parameters => Float64Array.from(exponentialInputs, (value, index) =>
        parameters[0] * Math.exp(parameters[1] * value) - exponentialValues[index])
    },
    {
      name: 'analytic-square', initial: [3], expected: [2], tolerance: PARAMETER_TOLERANCE,
      maximumCost: 1e-6,
      residual: parameters => new Float64Array([parameters[0] * parameters[0] - 4]),
      jacobian: parameters => api.float64ArrayToMatrix(new Float64Array([2 * parameters[0]]))
    }
  ];
}

export function mediumProblem() {
  let state = FIXTURE_SEED;
  const matrix = Float64Array.from({ length: MEDIUM_ROWS * MEDIUM_COLUMNS }, () => {
    state ^= state << RANDOM_LEFT_SHIFT;
    state ^= state >>> RANDOM_RIGHT_SHIFT;
    state ^= state << RANDOM_FINAL_SHIFT;
    return ((state >>> 0) / UINT32_RANGE - RANDOM_UNIT_MIDPOINT) *
      (PERTURBATION_NORM_BOUND / RANDOM_UNIT_MIDPOINT) / Math.sqrt(MEDIUM_ROWS * MEDIUM_COLUMNS);
  });
  for (let index = 0; index < MEDIUM_COLUMNS; index++)
    matrix[index * MEDIUM_COLUMNS + index] += DIAGONAL_STRENGTH;
  // J is an embedded 2I plus a perturbation with Frobenius norm <= 0.5.
  // Singular values therefore lie in [1.5, 2.5], independently of the solver.
  return {
    name: 'dense-128x32', seed: FIXTURE_SEED,
    fixedJacobian: { rows: MEDIUM_ROWS, columns: MEDIUM_COLUMNS, values: matrix },
    conditionUpperBound: (DIAGONAL_STRENGTH + PERTURBATION_NORM_BOUND) /
      (DIAGONAL_STRENGTH - PERTURBATION_NORM_BOUND),
    initial: Array(MEDIUM_COLUMNS).fill(0), expected: Array(MEDIUM_COLUMNS).fill(1),
    tolerance: PARAMETER_TOLERANCE, maximumCost: 1e-10,
    residual(parameters) {
      const residual = new Float64Array(MEDIUM_ROWS);
      for (let row = 0; row < MEDIUM_ROWS; row++)
        for (let column = 0; column < MEDIUM_COLUMNS; column++)
          residual[row] += matrix[row * MEDIUM_COLUMNS + column] * (parameters[column] - 1);
      return residual;
    }
  };
}

export function runProblem(api, wasmModule, LinearAlgebra, problem, backend) {
  const parameterCount = problem.initial.length;
  const preparationStarted = performance.now();
  const solver = backend === 'kibo'
    ? api.createKiboLinearSolver(new LinearAlgebra(wasmModule, parameterCount, parameterCount))
    : api.createMlMatrixLinearSolver();
  const preparationMilliseconds = performance.now() - preparationStarted;
  let residualEvaluations = 0;
  let jacobianEvaluations = 0;
  try {
    const residual = parameters => { residualEvaluations++; return problem.residual(parameters); };
    const jacobian = problem.jacobian
      ? parameters => { jacobianEvaluations++; return problem.jacobian(parameters); }
      : undefined;
    const started = performance.now();
    const result = api.levenbergMarquardt(Float64Array.from(problem.initial), residual, {
      maxIterations: MAXIMUM_ITERATIONS, tolGradient: GRADIENT_TOLERANCE, jacobian, linearSolver: solver
    });
    const elapsedMilliseconds = performance.now() - started;
    const finalResidual = problem.residual(result.finalParameters);
    // numopt-js reports the sum of squares (without a factor of one half).
    const recomputedCost = Array.from(finalResidual).reduce((sum, value) => sum + value * value, 0);
    const parameterError = maximumError(result.finalParameters, problem.expected);
    const statistics = solver.statistics;
    ensure(result.converged && Number.isFinite(recomputedCost) && parameterError <= problem.tolerance &&
      recomputedCost < problem.maximumCost, `${problem.name}/${backend}: numerical acceptance failed`);
    ensure(Math.abs(result.finalCost - recomputedCost) <= COST_TOLERANCE,
      `${problem.name}/${backend}: reported cost differs from the residual`);
    if (backend === 'kibo') {
      ensure(statistics.wasmCalls > 0 && statistics.solveSuccesses > 0 && statistics.fallbackAttempts === 0,
        `${problem.name}: expected real WASM success without LU fallback`);
      ensure(wasmModule._kibo_allocation_count() === 0, 'prepared C++ solve allocated');
    }
    return {
      problem: problem.name, backend, parameters: Array.from(result.finalParameters),
      parameterError, cost: result.finalCost, gradientNorm: result.finalGradientNorm,
      residualNorm: result.finalResidualNorm, converged: result.converged, iterations: result.iterations,
      residualEvaluations, jacobianEvaluations, preparationMilliseconds, elapsedMilliseconds,
      allocatedNumericBytes: solver.allocatedNumericBytes ?? 0, statistics
    };
  } finally { solver.dispose?.(); }
}

export function runSmallComparisons(api, wasmModule, LinearAlgebra) {
  return smallProblems(api).flatMap(problem =>
    ['ml-matrix', 'kibo'].map(backend => runProblem(api, wasmModule, LinearAlgebra, problem, backend)));
}

export function runResourceChecks(api, wasmModule, LinearAlgebra) {
  const workspace = new LinearAlgebra(wasmModule, 1, 1);
  const solver = api.createKiboLinearSolver(workspace);
  const matrix = api.float64ArrayToMatrix(new Float64Array([2]));
  const rightHandSide = api.float64ArrayToMatrix(new Float64Array([4]));
  const originalMatrix = matrix.get(0, 0), originalRight = rightHandSide.get(0, 0);
  let growth;
  try {
    const bytes = solver.allocatedNumericBytes;
    const saved = solver.solve(matrix, rightHandSide);
    const originalSolution = saved[0];
    const originalBuffer = wasmModule.HEAPU8.buffer;
    growth = wasmModule._kibo_allocate(MEMORY_GROWTH_BYTES);
    ensure(growth && wasmModule.HEAPU8.buffer !== originalBuffer, 'expected WASM memory growth');
    const again = solver.solve(matrix, rightHandSide);
    ensure(Math.abs(originalSolution - 2) <= SOLUTION_TOLERANCE && saved[0] === originalSolution &&
      Math.abs(again[0] - 2) <= SOLUTION_TOLERANCE, 'solution copies must survive growth and next solve');
    ensure(bytes === solver.allocatedNumericBytes, 'workspace must be reused');
    ensure(matrix.get(0, 0) === originalMatrix && rightHandSide.get(0, 0) === originalRight, 'solver changed caller inputs');
    solver.dispose(); solver.dispose();
    ensure(saved[0] === originalSolution, 'copied solution must survive dispose');
    let rejected = false;
    try {
      api.levenbergMarquardt(new Float64Array([0]), parameters => new Float64Array([parameters[0] - 2]),
        { maxIterations: 2, linearSolver: solver });
    } catch (error) { rejected = error instanceof api.LinearSolverConfigurationError; }
    ensure(rejected, 'public LM must reject disposed solver');
    return { growth: true, independentCopy: true, inputsPreserved: true, workspaceReused: true, disposedRejected: true };
  } finally {
    if (growth) wasmModule._kibo_release(growth);
    solver.dispose();
  }
}
