/**
 * Dense BFGS inverse-Hessian update and descent-direction fallback.
 * Shared by unconstrained bfgs and reduced-space adjointBfgs. Not a public export.
 */

import { Matrix } from 'ml-matrix';
import { Logger } from './logger.js';
import { dotProduct, scaleVector } from '../utils/matrix.js';

const NEGATIVE_GRADIENT_DIRECTION = -1.0;
const MINIMUM_CURVATURE_THRESHOLD = 1e-10;

export function createIdentityInverseHessian(dimension: number): Matrix {
  // NOTE: Provide both dimensions to stay compatible with our (older) local typing history
  // and with ml-matrix's API where `columns` is optional.
  return Matrix.eye(dimension, dimension);
}

export function multiplyMatrixVector(matrix: Matrix, vector: Float64Array): Float64Array {
  const result = new Float64Array(vector.length);
  for (let rowIndex = 0; rowIndex < matrix.rows; rowIndex++) {
    let sum = 0.0;
    for (let columnIndex = 0; columnIndex < matrix.columns; columnIndex++) {
      sum += matrix.get(rowIndex, columnIndex) * vector[columnIndex];
    }
    result[rowIndex] = sum;
  }
  return result;
}

export function computeBfgsSearchDirection(
  inverseHessianApproximation: Matrix,
  currentGradient: Float64Array
): Float64Array {
  const approximateNewtonDirection = multiplyMatrixVector(inverseHessianApproximation, currentGradient);
  return scaleVector(approximateNewtonDirection, NEGATIVE_GRADIENT_DIRECTION);
}

export function ensureDescentDirectionOrFallback(
  currentGradient: Float64Array,
  proposedSearchDirection: Float64Array,
  currentInverseHessianApproximation: Matrix,
  logger: Logger,
  iteration: number,
  currentCost: number,
  algorithmName: string
): { searchDirection: Float64Array; inverseHessianApproximation: Matrix } {
  const directionalDerivative = dotProduct(currentGradient, proposedSearchDirection);
  const isDescentDirection = directionalDerivative < 0.0;
  if (isDescentDirection) {
    return {
      searchDirection: proposedSearchDirection,
      inverseHessianApproximation: currentInverseHessianApproximation
    };
  }

  // WHY: If numerical issues yield a non-descent direction, reset H to identity and fall back to -g.
  logger.warn(
    algorithmName,
    iteration,
    'Non-descent direction detected; resetting inverse Hessian and using negative gradient.',
    [
      { key: 'Cost:', value: currentCost },
      { key: 'Directional derivative:', value: directionalDerivative }
    ]
  );
  return {
    searchDirection: scaleVector(currentGradient, NEGATIVE_GRADIENT_DIRECTION),
    inverseHessianApproximation: createIdentityInverseHessian(currentGradient.length)
  };
}

export function updateInverseHessianApproximation(
  inverseHessianApproximation: Matrix,
  stepVector: Float64Array,
  gradientChangeVector: Float64Array,
  logger: Logger,
  iteration: number,
  currentCost: number,
  algorithmName: string
): Matrix {
  const stepDotGradientChange = dotProduct(stepVector, gradientChangeVector);
  const curvatureIsTooWeak = stepDotGradientChange <= MINIMUM_CURVATURE_THRESHOLD;
  if (curvatureIsTooWeak) {
    // WHY: If curvature is weak/negative, the BFGS update can break positive definiteness.
    logger.warn(algorithmName, iteration, 'Curvature condition too weak; resetting inverse Hessian approximation.', [
      { key: 'Cost:', value: currentCost },
      { key: 'stepDotGradientChange:', value: stepDotGradientChange }
    ]);
    return createIdentityInverseHessian(stepVector.length);
  }

  const curvatureScaling = 1.0 / stepDotGradientChange;
  const stepMatrix = Matrix.columnVector(Array.from(stepVector));
  const gradientChangeMatrix = Matrix.columnVector(Array.from(gradientChangeVector));

  const stepGradientOuterProduct = stepMatrix.mmul(gradientChangeMatrix.transpose()).mul(curvatureScaling);
  const gradientStepOuterProduct = gradientChangeMatrix.mmul(stepMatrix.transpose()).mul(curvatureScaling);

  // WHY: ml-matrix `sub` mutates the receiver. Reusing one identity for both
  // factors makes left === right === I-ρsy^T-ρys^T, so the rank-two part
  // collapses to H in 1-D and the inverse Hessian never leaves I+ρss^T.
  const leftFactor = createIdentityInverseHessian(stepVector.length).sub(stepGradientOuterProduct);
  const rightFactor = createIdentityInverseHessian(stepVector.length).sub(gradientStepOuterProduct);

  const rankTwoPart = leftFactor.mmul(inverseHessianApproximation).mmul(rightFactor);
  const rankOnePart = stepMatrix.mmul(stepMatrix.transpose()).mul(curvatureScaling);

  return rankTwoPart.add(rankOnePart);
}
