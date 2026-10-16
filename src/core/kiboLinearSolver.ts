/** Own a prepared kibo facade and adapt its synchronous LLT solve to LM. */
import { solve, type Matrix } from 'ml-matrix';
import { matrixToFloat64Array } from '../utils/matrix.js';
import type { LevenbergMarquardtLinearSolver, LinearSolverStatistics } from './lmLinearSolver.js';
import { initialSolverStatistics, snapshotSolverStatistics, LinearSolverConfigurationError } from './lmLinearSolver.js';

const KIBO_ABI_VERSION = 1;
const KIBO_SUCCESS = 0;
const KIBO_NON_POSITIVE_PIVOT = 8;
const KIBO_ARITHMETIC_FAILURE = 10;

/** The existing kibo JavaScript facade, after module initialization and setup. */
export interface PreparedKiboLinearAlgebra {
  readonly rows: number;
  readonly cols: number;
  readonly disposed?: boolean;
  readonly module: { _kibo_abi_version(): number };
  readonly allocatedNumericBytes: number;
  solveLlt(matrix: Float64Array, rightHandSide: Float64Array): {
    status: number;
    output?: Float64Array;
  };
  dispose(): void;
}

export interface KiboLinearSolver extends LevenbergMarquardtLinearSolver {
  readonly statistics: LinearSolverStatistics;
  readonly allocatedNumericBytes: number;
  dispose(): void;
}

class PreparedKiboSolver implements KiboLinearSolver {
  private readonly observations = initialSolverStatistics();
  private readonly parameterCount: number;
  private disposed = false;

  constructor(private readonly workspace: PreparedKiboLinearAlgebra) {
    if (workspace.module._kibo_abi_version() !== KIBO_ABI_VERSION)
      throw new LinearSolverConfigurationError('Unsupported kibo ABI');
    this.parameterCount = workspace.cols;
    if (!Number.isSafeInteger(this.parameterCount) || this.parameterCount <= 0 || workspace.rows !== this.parameterCount)
      throw new LinearSolverConfigurationError('kibo workspace must be a nonempty square matrix');
  }

  get statistics(): LinearSolverStatistics { return snapshotSolverStatistics(this.observations); }
  get allocatedNumericBytes(): number { return this.workspace.allocatedNumericBytes; }

  private copyInputs(matrix: Matrix, rightHandSide: Matrix) {
    if (this.disposed || this.workspace.disposed) throw new LinearSolverConfigurationError('kibo solver disposed');
    if (matrix.rows !== this.parameterCount || matrix.columns !== this.parameterCount ||
        rightHandSide.rows !== this.parameterCount || rightHandSide.columns !== 1)
      throw new LinearSolverConfigurationError('kibo input shape differs from the prepared workspace');
    const started = performance.now();
    const input = Float64Array.from(matrix.to1DArray());
    const right = matrixToFloat64Array(rightHandSide);
    if (!input.every(Number.isFinite) || !right.every(Number.isFinite))
      throw new LinearSolverConfigurationError('kibo inputs must be finite');
    this.observations.conversionMilliseconds += performance.now() - started;
    this.observations.inputPayloadBytes += input.byteLength + right.byteLength;
    return { input, right };
  }

  private callWasm(input: Float64Array, right: Float64Array) {
    this.observations.wasmCalls++;
    const started = performance.now();
    try { return this.workspace.solveLlt(input, right); }
    catch (error) { throw new LinearSolverConfigurationError(`kibo call failed: ${String(error)}`); }
    finally { this.observations.boundaryMilliseconds += performance.now() - started; }
  }

  solve(matrix: Matrix, rightHandSide: Matrix): Float64Array {
    const { input, right } = this.copyInputs(matrix, rightHandSide);
    this.observations.solveAttempts++;
    const result = this.callWasm(input, right);
    if (!result || typeof result !== 'object' || !Number.isInteger(result.status))
      throw new LinearSolverConfigurationError('kibo returned an invalid result');
    if (result.status === KIBO_SUCCESS && result.output) {
      if (!(result.output instanceof Float64Array) || result.output.length !== this.parameterCount || !result.output.every(Number.isFinite))
        throw new LinearSolverConfigurationError('kibo output must have the prepared shape and finite values');
      this.observations.solveSuccesses++;
      this.observations.outputPayloadBytes += result.output.byteLength;
      return result.output;
    }
    if (result.status !== KIBO_NON_POSITIVE_PIVOT && result.status !== KIBO_ARITHMETIC_FAILURE)
      throw new LinearSolverConfigurationError(`Invalid kibo solve status: ${result.status}`);
    const solution = this.fallback(matrix, rightHandSide, result.status);
    this.observations.solveSuccesses++;
    return solution;
  }

  private fallback(matrix: Matrix, rightHandSide: Matrix, status: number) {
    this.observations.fallbackAttempts++;
    const reason = `kibo-status-${status}`;
    this.observations.failureReasons[reason] = (this.observations.failureReasons[reason] ?? 0) + 1;
    const started = performance.now();
    try {
      const solution = matrixToFloat64Array(solve(matrix, rightHandSide));
      if (!solution.every(Number.isFinite)) throw new Error('LU fallback produced a non-finite solution');
      return solution;
    } finally { this.observations.boundaryMilliseconds += performance.now() - started; }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.workspace.dispose();
  }
}

/** Ownership of a dedicated prepared facade transfers to this solver. */
export function createKiboLinearSolver(workspace: PreparedKiboLinearAlgebra): KiboLinearSolver {
  return new PreparedKiboSolver(workspace);
}
