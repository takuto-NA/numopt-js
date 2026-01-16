import { gradientDescent } from '../src/core/gradientDescent';
import type { CostFn, GradientFn } from '../src/core/types';

describe('Gradient Descent', () => {
  /**
   * Simple quadratic function: f(x) = x^2
   * Minimum at x = 0
   * Gradient: f'(x) = 2x
   */
  const quadraticCost: CostFn = (params: Float64Array) => {
    return params[0] * params[0];
  };

  const quadraticGradient: GradientFn = (params: Float64Array) => {
    return new Float64Array([2 * params[0]]);
  };

  it('should converge to minimum for simple quadratic function', () => {
    const initialParams = new Float64Array([5.0]);
    const result = gradientDescent(initialParams, quadraticCost, quadraticGradient, {
      maxIterations: 100,
      tolerance: 1e-6
    });

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0])).toBeLessThan(1e-3);
    expect(result.finalCost).toBeLessThan(1e-6);
  });

  it('should work with fixed step size', () => {
    const initialParams = new Float64Array([10.0]);
    const result = gradientDescent(initialParams, quadraticCost, quadraticGradient, {
      stepSize: 0.1,
      useLineSearch: false,
      maxIterations: 1000,
      tolerance: 1e-6
    });

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0])).toBeLessThan(1e-3);
  });

  it('should use line search when enabled', () => {
    const initialParams = new Float64Array([3.0]);
    const result = gradientDescent(initialParams, quadraticCost, quadraticGradient, {
      useLineSearch: true,
      maxIterations: 100,
      tolerance: 1e-6
    });

    expect(result.converged).toBe(true);
    expect(result.usedLineSearch).toBe(true);
  });

  /**
   * 2D quadratic function: f(x, y) = x^2 + y^2
   * Minimum at (0, 0)
   * Gradient: [2x, 2y]
   */
  const quadratic2DCost: CostFn = (params: Float64Array) => {
    return params[0] * params[0] + params[1] * params[1];
  };

  const quadratic2DGradient: GradientFn = (params: Float64Array) => {
    return new Float64Array([2 * params[0], 2 * params[1]]);
  };

  it('should converge for 2D quadratic function', () => {
    const initialParams = new Float64Array([5.0, -3.0]);
    const result = gradientDescent(initialParams, quadratic2DCost, quadratic2DGradient, {
      maxIterations: 1000,
      tolerance: 1e-6
    });

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0])).toBeLessThan(1e-3);
    expect(Math.abs(result.finalParameters[1])).toBeLessThan(1e-3);
    expect(result.finalCost).toBeLessThan(1e-6);
  });

  it('should call onIteration callback if provided', () => {
    const initialParams = new Float64Array([2.0]);
    let callbackCalled = false;
    let iterationCount = 0;

    gradientDescent(initialParams, quadraticCost, quadraticGradient, {
      maxIterations: 10,
      tolerance: 1e-6,
      onIteration: (iteration, cost, params) => {
        callbackCalled = true;
        iterationCount = iteration;
        expect(typeof cost).toBe('number');
        expect(params).toBeInstanceOf(Float64Array);
      }
    });

    expect(callbackCalled).toBe(true);
    expect(iterationCount).toBeGreaterThan(0);
  });

  it('should handle maximum iterations gracefully', () => {
    // Use a problem that won't converge quickly: f(x) = x^4 (flatter near minimum)
    const flatCost: CostFn = (params: Float64Array) => {
      const x = params[0];
      return x * x * x * x;
    };

    const flatGradient: GradientFn = (params: Float64Array) => {
      const x = params[0];
      return new Float64Array([4 * x * x * x]);
    };

    const initialParams = new Float64Array([10.0]);
    const result = gradientDescent(initialParams, flatCost, flatGradient, {
      maxIterations: 5,
      tolerance: 1e-10,
      stepSize: 0.01,
      useLineSearch: false
    });

    expect(result.converged).toBe(false);
    expect(result.iterations).toBe(5);
    expect(result.finalParameters).toBeInstanceOf(Float64Array);
  });
});

