/** Multiple-right-hand-side state solves; one LU/QR factorization on regular square/tall systems. */

import { LuDecomposition, Matrix, QrDecomposition } from 'ml-matrix';
import { solveLeastSquares } from './constrainedUtils.js';
import { Logger } from './logger.js';

function assertFinite(matrix: Matrix, label: string): void {
  if (!matrix.to1DArray().every(Number.isFinite)) throw new Error(`${label} contains NaN or Infinity`);
}

export function solveLeastSquaresMultiple(
  coefficients: Matrix, rightHandSides: Matrix, logger: Logger, algorithmName: string
): Matrix {
  if (coefficients.rows !== rightHandSides.rows) throw new Error('State solve requires matching right-hand-side rows');
  assertFinite(coefficients, 'Constraint state Jacobian');
  assertFinite(rightHandSides, 'Constraint parameter Jacobian');
  if (coefficients.rows >= coefficients.columns) {
    try {
      // QR avoids squaring the condition number of redundant constraints in normal equations.
      const decomposition = coefficients.isSquare()
        ? new LuDecomposition(coefficients) : new QrDecomposition(coefficients);
      const solution = decomposition.solve(rightHandSides);
      assertFinite(solution, 'State sensitivity');
      return solution;
    } catch {
      // Preserve the existing regularization policy for singular or numerically unusable systems.
    }
  }
  const solution = Matrix.zeros(coefficients.columns, rightHandSides.columns);
  for (let column = 0; column < rightHandSides.columns; column++) {
    const values = solveLeastSquares(coefficients, Matrix.columnVector(rightHandSides.getColumn(column)), logger, algorithmName);
    solution.setColumn(column, values);
  }
  assertFinite(solution, 'State sensitivity');
  return solution;
}
