import { bfgs } from '../src/core/bfgs';
import type { CostFn, GradientFn } from '../src/core/types';

describe('BFGS', () => {
  /**
   * Simple quadratic: f(x) = x^2
   * Minimum at x = 0
   */
  const quadraticCost: CostFn = (parameters: Float64Array) => {
    return parameters[0] * parameters[0];
  };

  const quadraticGradient: GradientFn = (parameters: Float64Array) => {
    return new Float64Array([2 * parameters[0]]);
  };

  it('should converge on a simple quadratic', () => {
    const initialParameters = new Float64Array([5.0]);
    const result = bfgs(initialParameters, quadraticCost, quadraticGradient, {
      maxIterations: 200,
      tolerance: 1e-10
    });

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0])).toBeLessThan(1e-6);
    expect(result.finalCost).toBeLessThan(1e-12);
  });

  /**
   * Rosenbrock function: f(x, y) = (1-x)^2 + 100*(y-x^2)^2
   * Minimum at (1, 1)
   */
  const rosenbrockCost: CostFn = (parameters: Float64Array) => {
    const x = parameters[0];
    const y = parameters[1];
    const term1 = 1 - x;
    const term2 = y - x * x;
    return term1 * term1 + 100 * term2 * term2;
  };

  const rosenbrockGradient: GradientFn = (parameters: Float64Array) => {
    const x = parameters[0];
    const y = parameters[1];
    const gradientX = -2 * (1 - x) - 400 * x * (y - x * x);
    const gradientY = 200 * (y - x * x);
    return new Float64Array([gradientX, gradientY]);
  };

  it('should minimize Rosenbrock function', () => {
    const initialParameters = new Float64Array([-1.0, 1.0]);
    const result = bfgs(initialParameters, rosenbrockCost, rosenbrockGradient, {
      maxIterations: 5000,
      tolerance: 1e-6
    });

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - 1.0)).toBeLessThan(1e-3);
    expect(Math.abs(result.finalParameters[1] - 1.0)).toBeLessThan(1e-3);
  });
});

