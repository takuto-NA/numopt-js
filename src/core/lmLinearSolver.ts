/** Synchronous LM linear-system solvers and observable per-instance statistics. */
import { CholeskyDecomposition, Matrix, solve } from 'ml-matrix';
import { matrixToFloat64Array } from '../utils/matrix.js';

/** Inputs already include damping and the negative residual-gradient sign. */
export interface LevenbergMarquardtLinearSolver {
  solve(matrix: Matrix, rightHandSide: Matrix): Float64Array;
}

export interface LinearSolverStatistics {
  solveAttempts: number;
  solveSuccesses: number;
  fallbackAttempts: number;
  wasmCalls: number;
  conversionMilliseconds: number;
  boundaryMilliseconds: number;
  inputPayloadBytes: number;
  outputPayloadBytes: number;
  failureReasons: Record<string, number>;
}

export function initialSolverStatistics(): LinearSolverStatistics {
  return {
    solveAttempts: 0, solveSuccesses: 0, fallbackAttempts: 0, wasmCalls: 0,
    conversionMilliseconds: 0, boundaryMilliseconds: 0,
    inputPayloadBytes: 0, outputPayloadBytes: 0, failureReasons: {}
  };
}

export function snapshotSolverStatistics(statistics: LinearSolverStatistics): LinearSolverStatistics {
  return { ...statistics, failureReasons: { ...statistics.failureReasons } };
}

/** Configuration/lifetime errors cannot be repaired by changing LM damping. */
export class LinearSolverConfigurationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LinearSolverConfigurationError';
  }
}

/** Explicit selection of the existing Cholesky, then LU, policy. */
export function createMlMatrixLinearSolver(): LevenbergMarquardtLinearSolver & {
  readonly statistics: LinearSolverStatistics;
} {
  const statistics = initialSolverStatistics();
  return {
    get statistics() { return snapshotSolverStatistics(statistics); },
    solve(matrix, rightHandSide) {
      statistics.solveAttempts++;
      const started = performance.now();
      try {
        let cholesky: CholeskyDecomposition | undefined;
        try { cholesky = new CholeskyDecomposition(matrix); } catch { /* LU below. */ }
        if (cholesky?.isPositiveDefinite()) {
          try {
            const solution = matrixToFloat64Array(cholesky.solve(rightHandSide));
            statistics.solveSuccesses++;
            return solution;
          } catch { /* Preserve the original fallback on a Cholesky solve failure. */ }
        }
        statistics.fallbackAttempts++;
        statistics.failureReasons['ml-matrix-cholesky'] =
          (statistics.failureReasons['ml-matrix-cholesky'] ?? 0) + 1;
        const solution = matrixToFloat64Array(solve(matrix, rightHandSide));
        statistics.solveSuccesses++;
        return solution;
      } finally { statistics.boundaryMilliseconds += performance.now() - started; }
    }
  };
}
