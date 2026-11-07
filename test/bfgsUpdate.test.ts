/** Regression checks for small positive curvature, scale changes, and invalid secant pairs. */

import { Matrix } from 'ml-matrix';
import { describe, expect, it } from 'vitest';
import { updateInverseHessianApproximation } from '../src/core/bfgsUpdate';
import { Logger } from '../src/core/logger';

const SILENT_LOGGER = new Logger(undefined, undefined);
const SMALL_STEP = 1e-7;
const QUADRATIC_CURVATURE = 4;
const OBJECTIVE_SCALES = [1e-8, 1, 1e8];

function update(inverse: Matrix, step: Float64Array, change: Float64Array): Matrix {
  return updateInverseHessianApproximation(inverse, step, change, SILENT_LOGGER, 0, 0, 'test');
}

describe('BFGS secant updates', () => {
  it('keeps valid curvature when the absolute step-gradient product is tiny', () => {
    const result = update(Matrix.eye(1), new Float64Array([SMALL_STEP]),
      new Float64Array([QUADRATIC_CURVATURE * SMALL_STEP]));
    expect(result.get(0, 0)).toBeCloseTo(1 / QUADRATIC_CURVATURE, 12);
  });

  it.each(OBJECTIVE_SCALES)('respects the secant equation when the cost is scaled by %g', scale => {
    const initial = Matrix.diag([1 / scale, 2 / scale]);
    const step = new Float64Array([SMALL_STEP, -SMALL_STEP]);
    const change = new Float64Array([2 * SMALL_STEP * scale, -5 * SMALL_STEP * scale]);
    const result = update(initial, step, change);
    const recovered = result.mmul(Matrix.columnVector(change)).to1DArray();
    recovered.forEach((value, index) => expect(value / SMALL_STEP).toBeCloseTo(step[index] / SMALL_STEP, 10));
    expect(initial.to2DArray()).toEqual([[1 / scale, 0], [0, 2 / scale]]);
  });

  it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY])('skips unusable curvature (%s) without destroying a valid model', value => {
    const initial = Matrix.diag([2, 3]);
    const result = update(initial, new Float64Array([1, 0]), new Float64Array([value, 0]));
    expect(result.to2DArray()).toEqual(initial.to2DArray());
  });

  it('rejects nearly orthogonal pairs even when the reciprocal curvature is finite', () => {
    const initial = Matrix.diag([2, 3]);
    const result = update(initial, new Float64Array([1, 0]), new Float64Array([1e-12, 1]));
    expect(result.to2DArray()).toEqual(initial.to2DArray());
  });
});
