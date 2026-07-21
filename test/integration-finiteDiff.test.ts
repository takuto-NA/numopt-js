/**
 * Integration: numerical gradient helpers work with gradientDescent.
 * Also guards the documented finiteDiffGradient argument-order foot-gun.
 */

import { gradientDescent } from '../src/core/gradientDescent';
import { finiteDiffGradient } from '../src/core/finiteDiff';
import { createFiniteDiffGradient } from '../src/core/createGradientFunction';
import type { CostFn } from '../src/core/types';

const TARGET_X = 3;
const TARGET_Y = 2;
const PARAMETER_TOLERANCE = 1e-3;
const MAX_ITERATIONS = 100;
const OPTIMIZER_TOLERANCE = 1e-6;

describe('Integration: finiteDiffGradient with gradientDescent', () => {
  const costFn: CostFn = (params: Float64Array) => {
    return (params[0] - TARGET_X) ** 2 + (params[1] - TARGET_Y) ** 2;
  };

  it('should converge with finiteDiffGradient and createFiniteDiffGradient', () => {
    const inlineNumericGradientResult = gradientDescent(
      new Float64Array([0, 0]),
      costFn,
      (params) => finiteDiffGradient(params, costFn),
      { maxIterations: MAX_ITERATIONS, tolerance: OPTIMIZER_TOLERANCE }
    );

    const helperGradientResult = gradientDescent(
      new Float64Array([0, 0]),
      costFn,
      createFiniteDiffGradient(costFn),
      { maxIterations: MAX_ITERATIONS, tolerance: OPTIMIZER_TOLERANCE }
    );

    expect(inlineNumericGradientResult.converged).toBe(true);
    expect(helperGradientResult.converged).toBe(true);
    expect(Math.abs(inlineNumericGradientResult.finalParameters[0] - TARGET_X)).toBeLessThan(PARAMETER_TOLERANCE);
    expect(Math.abs(inlineNumericGradientResult.finalParameters[1] - TARGET_Y)).toBeLessThan(PARAMETER_TOLERANCE);
    expect(Math.abs(helperGradientResult.finalParameters[0] - TARGET_X)).toBeLessThan(PARAMETER_TOLERANCE);
    expect(Math.abs(helperGradientResult.finalParameters[1] - TARGET_Y)).toBeLessThan(PARAMETER_TOLERANCE);
  });

  it('should throw when finiteDiffGradient arguments are reversed', () => {
    expect(() => {
      gradientDescent(
        new Float64Array([0, 0]),
        costFn,
        (params) => finiteDiffGradient(costFn as unknown as Float64Array, params as unknown as CostFn),
        { maxIterations: MAX_ITERATIONS, tolerance: OPTIMIZER_TOLERANCE }
      );
    }).toThrow();
  });
});
