/**
 * Section-12 Hessian approximation f_pp + S^T f_xx S and safeguarded Newton direction.
 * Numerical state curvature uses fixed-p directional differences, not restored states.
 */

import { CholeskyDecomposition, Matrix, solve } from 'ml-matrix';
import type { AdjointDerivativeOptions, AdjointNewtonOptions, CostFn } from './types.js';
import type { ReducedPartials } from './adjointReducedSpace.js';
import { dotProduct, scaleVector } from '../utils/matrix.js';

const DEFAULT_HESSIAN_STEP_SIZE = 1e-4;
const SYMMETRY_TOLERANCE = 1e-8;
const MINIMUM_DAMPING_SCALE = 1e-8;
const DAMPING_GROWTH_FACTOR = 10;
const MAX_DAMPING_ATTEMPTS = 8;
const CENTRAL_SECOND_DIFFERENCE_FACTOR = 2;
const MIXED_SECOND_DIFFERENCE_FACTOR = 4;
const NEGATIVE_DIRECTION = -1;

export function checkedSymmetricMatrix(matrix: Matrix, dimension: number, name: string): Matrix {
  if (matrix.rows !== dimension || matrix.columns !== dimension) {
    throw new Error(`${name} must be ${dimension} × ${dimension}`);
  }
  const symmetric = matrix.clone();
  for (let row = 0; row < dimension; row++) {
    for (let column = row; column < dimension; column++) {
      const left = matrix.get(row, column);
      const right = matrix.get(column, row);
      if (!Number.isFinite(left) || !Number.isFinite(right)) {
        throw new Error(`${name} must contain finite values`);
      }
      if (Math.abs(left - right) > SYMMETRY_TOLERANCE * Math.max(1, Math.abs(left), Math.abs(right))) {
        throw new Error(`${name} must be symmetric`);
      }
      const average = left / CENTRAL_SECOND_DIFFERENCE_FACTOR + right / CENTRAL_SECOND_DIFFERENCE_FACTOR;
      symmetric.set(row, column, average);
      symmetric.set(column, row, average);
    }
  }
  return symmetric;
}

function finiteCost(cost: CostFn, values: Float64Array): number {
  const value = cost(values);
  if (!Number.isFinite(value)) {
    throw new Error('Non-finite partial Hessian sample; provide analytical partial Hessians or reduce the Hessian step');
  }
  return value;
}

function shiftedPoint(
  base: Float64Array, directions: Matrix, first: number, firstStep: number,
  second: number = first, secondStep: number = 0
): Float64Array {
  return Float64Array.from(base, (value, row) =>
    value + directions.get(row, first) * firstStep + directions.get(row, second) * secondStep);
}

/** Computes D^T ∇²f D with O(parameterCount²) samples, even when states are numerous. */
export function directionalHessian(cost: CostFn, base: Float64Array, directions: Matrix, relativeStep: number): Matrix {
  const dimension = directions.columns;
  const hessian = Matrix.zeros(dimension, dimension);
  const baseCost = finiteCost(cost, base);
  const steps = Array.from({ length: dimension }, (_, column) => {
    let coordinateScale = Number.POSITIVE_INFINITY;
    for (let row = 0; row < base.length; row++) {
      const magnitude = Math.abs(directions.get(row, column));
      if (magnitude > 0) coordinateScale = Math.min(coordinateScale, Math.max(1, Math.abs(base[row])) / magnitude);
    }
    return Number.isFinite(coordinateScale) ? relativeStep * coordinateScale : 0;
  });
  for (let first = 0; first < dimension; first++) {
    const firstStep = steps[first];
    if (firstStep === 0) continue;
    const forward = finiteCost(cost, shiftedPoint(base, directions, first, firstStep));
    const backward = finiteCost(cost, shiftedPoint(base, directions, first, -firstStep));
    hessian.set(first, first,
      (forward - CENTRAL_SECOND_DIFFERENCE_FACTOR * baseCost + backward) / (firstStep * firstStep));
    for (let second = first + 1; second < dimension; second++) {
      const secondStep = steps[second];
      if (secondStep === 0) continue;
      const sample = (firstSign: number, secondSign: number): number => finiteCost(cost,
        shiftedPoint(base, directions, first, firstSign * firstStep, second, secondSign * secondStep));
      const value = (sample(1, 1) - sample(1, -1) - sample(-1, 1) + sample(-1, -1)) /
        (MIXED_SECOND_DIFFERENCE_FACTOR * firstStep * secondStep);
      hessian.set(first, second, value);
      hessian.set(second, first, value);
    }
  }
  return checkedSymmetricMatrix(hessian, dimension, 'Numerical partial Hessian');
}

export function computeStateSensitivities(partials: ReducedPartials, options: AdjointDerivativeOptions): Matrix {
  // WHY: A matrix right-hand side reuses one factorization across all parameter sensitivities.
  const stateJacobian = partials.dcdx.clone();
  const regularization = options.regularization ?? 0;
  for (let index = 0; index < stateJacobian.rows; index++) {
    stateJacobian.set(index, index, stateJacobian.get(index, index) + regularization);
  }
  let sensitivities: Matrix;
  try {
    sensitivities = solve(stateJacobian, partials.dcdp.clone().mul(NEGATIVE_DIRECTION));
  } catch {
    throw new Error('adjointNewton requires a nonsingular implicit-state Jacobian for state sensitivities');
  }
  if (sensitivities.to1DArray().some(value => !Number.isFinite(value))) {
    throw new Error('Non-finite state sensitivity; the implicit-state Jacobian must be nonsingular');
  }
  return sensitivities;
}

export function computeApproximateAdjointHessian(
  parameters: Float64Array,
  states: Float64Array,
  cost: (parameters: Float64Array, states: Float64Array) => number,
  partials: ReducedPartials,
  options: AdjointNewtonOptions
): Matrix {
  const sensitivities = computeStateSensitivities(partials, options);
  const parameterHessian = options.d2fdp2
    ? checkedSymmetricMatrix(options.d2fdp2(parameters, states), parameters.length, 'd2fdp2')
    : directionalHessian(values => cost(values, states), parameters,
      Matrix.eye(parameters.length), options.hessianStepSizeP ?? DEFAULT_HESSIAN_STEP_SIZE);
  const stateContribution = options.d2fdx2
    ? sensitivities.transpose().mmul(
      checkedSymmetricMatrix(options.d2fdx2(parameters, states), states.length, 'd2fdx2')
    ).mmul(sensitivities)
    : directionalHessian(values => cost(parameters, values), states,
      sensitivities, options.hessianStepSizeX ?? DEFAULT_HESSIAN_STEP_SIZE);
  return checkedSymmetricMatrix(parameterHessian.add(stateContribution), parameters.length, 'Approximate Hessian');
}

export type NewtonDirection = { direction: Float64Array; damping: number; usedGradientFallback: boolean };

function hessianBounds(hessian: Matrix): { scale: number; lowerBound: number } {
  let scale = 1;
  let lowerBound = Number.POSITIVE_INFINITY;
  for (let row = 0; row < hessian.rows; row++) {
    let radius = 0;
    for (let column = 0; column < hessian.columns; column++) {
      if (column !== row) radius += Math.abs(hessian.get(row, column));
    }
    const diagonal = hessian.get(row, row);
    scale = Math.max(scale, Math.abs(diagonal) + radius);
    lowerBound = Math.min(lowerBound, diagonal - radius);
  }
  return { scale, lowerBound };
}

export function computeSafeguardedNewtonDirection(
  hessian: Matrix, gradient: Float64Array, minimumDamping: number
): NewtonDirection {
  const { scale, lowerBound } = hessianBounds(hessian);
  const dampingFloor = MINIMUM_DAMPING_SCALE * scale;
  let damping = minimumDamping;
  for (let attempt = 0; attempt < MAX_DAMPING_ATTEMPTS; attempt++) {
    const shifted = hessian.clone();
    for (let index = 0; index < shifted.rows; index++) {
      shifted.set(index, index, shifted.get(index, index) + damping);
    }
    try {
      const decomposition = new CholeskyDecomposition(shifted);
      if (decomposition.isPositiveDefinite()) {
        const direction = Float64Array.from(decomposition.solve(Matrix.columnVector(Array.from(gradient))).getColumn(0),
          value => -value);
        const directionalDerivative = dotProduct(gradient, direction);
        if (direction.every(Number.isFinite) && Number.isFinite(directionalDerivative) && directionalDerivative < 0) {
          return { direction, damping, usedGradientFallback: false };
        }
      }
    } catch {
      // WHY: Indefinite/singular curvature is legitimate for this incomplete Hessian model.
    }
    damping = Math.max(damping * DAMPING_GROWTH_FACTOR, dampingFloor, -lowerBound + dampingFloor);
  }
  return { direction: scaleVector(gradient, NEGATIVE_DIRECTION), damping, usedGradientFallback: true };
}
