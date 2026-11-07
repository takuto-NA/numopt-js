/** Checks two-loop recursion against explicitly updated inverse Hessians. */
import { Matrix } from 'ml-matrix';
import { Logger } from '../src/core/logger.js';
import {
  createEmptyHistory, computeLbfgsSearchDirection, updateHistoryIfCurvatureIsValid,
  ensureDescentDirectionOrFallback
} from '../src/core/lbfgsUpdate.js';

describe('shared L-BFGS updates', () => {
  it('retains a valid tiny positive-curvature pair instead of clearing history', () => {
    const history = createEmptyHistory();
    const smallStep = 1e-7;
    updateHistoryIfCurvatureIsValid(history, 2, new Float64Array([1]), new Float64Array([2]), new Logger(undefined, undefined), 0, 0);
    updateHistoryIfCurvatureIsValid(history, 2, new Float64Array([smallStep]), new Float64Array([4 * smallStep]),
      new Logger(undefined, undefined), 1, 0);
    expect(history.stepVectorHistory).toHaveLength(2);
    const direction = computeLbfgsSearchDirection(new Float64Array([1]), history);
    expect(direction[0]).toBeCloseTo(-0.25, 12);
  });

  it('matches dense BFGS applied to retained pairs with the same initial scaling', () => {
    const history = createEmptyHistory();
    const pairs = [
      [new Float64Array([1, 2, 1]), new Float64Array([2, 3, 2])],
      [new Float64Array([2, 1, -1]), new Float64Array([3, 2, -1])],
      [new Float64Array([1, -1, 2]), new Float64Array([2, -1, 3])]
    ];
    for (const [step, change] of pairs) updateHistoryIfCurvatureIsValid(history, 2, step, change, new Logger(undefined, undefined), 0, 0);
    expect(history.stepVectorHistory).toHaveLength(2);
    const lastStep = Matrix.columnVector(Array.from(pairs[2][0]));
    const lastChange = Matrix.columnVector(Array.from(pairs[2][1]));
    const scaling = lastStep.transpose().mmul(lastChange).get(0, 0) /
      lastChange.transpose().mmul(lastChange).get(0, 0);
    let inverse = Matrix.eye(3).mul(scaling);
    for (let index = 0; index < history.stepVectorHistory.length; index++) {
      const step = Matrix.columnVector(Array.from(history.stepVectorHistory[index]));
      const change = Matrix.columnVector(Array.from(history.gradientChangeVectorHistory[index]));
      const reciprocal = history.reciprocalCurvatureHistory[index];
      const left = Matrix.sub(Matrix.eye(3), step.mmul(change.transpose()).mul(reciprocal));
      inverse = left.mmul(inverse).mmul(left.transpose()).add(step.mmul(step.transpose()).mul(reciprocal));
    }
    const gradient = new Float64Array([2, -3, 4]);
    const expected = inverse.mmul(Matrix.columnVector(Array.from(gradient))).mul(-1).to1DArray();
    const actual = computeLbfgsSearchDirection(gradient, history);
    actual.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 12));
  });

  it.each([new Float64Array([-1]), new Float64Array([NaN])])('clears invalid curvature and recovers descent', change => {
    const history = createEmptyHistory();
    updateHistoryIfCurvatureIsValid(history, 2, new Float64Array([1]), new Float64Array([2]), new Logger(undefined, undefined), 0, 0);
    updateHistoryIfCurvatureIsValid(history, 2, new Float64Array([1]), change, new Logger(undefined, undefined), 0, 0);
    expect(history.stepVectorHistory).toHaveLength(0);
    const direction = ensureDescentDirectionOrFallback(new Float64Array([2]), new Float64Array([NaN]), history,
      new Logger(undefined, undefined), 0, 0);
    expect(Array.from(direction)).toEqual([-2]);
  });
});
