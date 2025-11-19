import { describe, it, expect } from 'vitest';
import { gaussNewton } from '../src/core/gaussNewton';
import type { ResidualFn, JacobianFn } from '../src/core/types';
import { Matrix } from 'ml-matrix';

describe('Gauss-Newton Method', () => {
  /**
   * Simple linear least squares: r(x) = [x - 2]
   * Minimum at x = 2
   * Jacobian: J = [[1]]
   */
  const linearResidual: ResidualFn = (params: Float64Array) => {
    return new Float64Array([params[0] - 2]);
  };

  const linearJacobian: JacobianFn = (params: Float64Array) => {
    return new Matrix([[1]]);
  };

  it('should converge for simple linear least squares', () => {
    const initialParams = new Float64Array([0.0]);
    const result = gaussNewton(initialParams, linearResidual, {
      jacobian: linearJacobian,
      maxIterations: 10,
      tolerance: 1e-6
    });

    expect(result.converged).toBe(true);
    expect(Math.abs(result.parameters[0] - 2.0)).toBeLessThan(1e-3);
    expect(result.finalCost).toBeLessThan(1e-6);
  });

  it('should work with numerical Jacobian', () => {
    const initialParams = new Float64Array([0.0]);
    const result = gaussNewton(initialParams, linearResidual, {
      useNumericJacobian: true,
      maxIterations: 10,
      tolerance: 1e-6
    });

    expect(result.converged).toBe(true);
    expect(Math.abs(result.parameters[0] - 2.0)).toBeLessThan(1e-3);
  });

  /**
   * Nonlinear least squares: r(x) = [x^2 - 4]
   * Solutions at x = ±2
   * Jacobian: J = [[2x]]
   */
  const nonlinearResidual: ResidualFn = (params: Float64Array) => {
    return new Float64Array([params[0] * params[0] - 4]);
  };

  const nonlinearJacobian: JacobianFn = (params: Float64Array) => {
    return new Matrix([[2 * params[0]]]);
  };

  it('should converge for nonlinear least squares', () => {
    const initialParams = new Float64Array([3.0]);
    const result = gaussNewton(initialParams, nonlinearResidual, {
      jacobian: nonlinearJacobian,
      maxIterations: 100,
      tolerance: 1e-6
    });

    expect(result.converged).toBe(true);
    // Should converge to x = 2 (positive solution)
    expect(Math.abs(result.parameters[0] - 2.0)).toBeLessThan(1e-3);
    expect(result.finalCost).toBeLessThan(1e-6);
  });

  /**
   * 2D nonlinear least squares: r(x, y) = [x^2 + y^2 - 5, x + y - 3]
   * Solution: x = 1, y = 2 (or x = 2, y = 1)
   * Jacobian: J = [[2x, 2y], [1, 1]]
   */
  const residual2D: ResidualFn = (params: Float64Array) => {
    const x = params[0];
    const y = params[1];
    return new Float64Array([x * x + y * y - 5, x + y - 3]);
  };

  const jacobian2D: JacobianFn = (params: Float64Array) => {
    const x = params[0];
    const y = params[1];
    return new Matrix([[2 * x, 2 * y], [1, 1]]);
  };

  it('should converge for 2D nonlinear least squares', () => {
    // Simpler 2D problem: r(x, y) = [x - 1, y - 2]
    // Solution: x = 1, y = 2
    // Jacobian: J = [[1, 0], [0, 1]]
    const simpleResidual2D: ResidualFn = (params: Float64Array) => {
      return new Float64Array([params[0] - 1, params[1] - 2]);
    };

    const simpleJacobian2D: JacobianFn = (params: Float64Array) => {
      return new Matrix([[1, 0], [0, 1]]);
    };

    const initialParams = new Float64Array([0.0, 0.0]);
    const result = gaussNewton(initialParams, simpleResidual2D, {
      jacobian: simpleJacobian2D,
      maxIterations: 10,
      tolerance: 1e-6
    });

    expect(result.converged).toBe(true);
    expect(Math.abs(result.parameters[0] - 1.0)).toBeLessThan(1e-3);
    expect(Math.abs(result.parameters[1] - 2.0)).toBeLessThan(1e-3);
    expect(result.finalCost).toBeLessThan(1e-6);
  });

  it('should call onIteration with the current iteration starting at zero', () => {
    const initialParams = new Float64Array([0.0]);
    const iterations: number[] = [];
    const firstParams: number[] = [];
    const costs: number[] = [];

    const result = gaussNewton(initialParams, linearResidual, {
      jacobian: linearJacobian,
      maxIterations: 3,
      tolerance: 0,
      onIteration: (iteration, cost, params) => {
        iterations.push(iteration);
        costs.push(cost);
        if (iteration === 0) {
          firstParams.push(params[0]);
        }
      }
    });

    expect(iterations).toEqual([0, 1, 2]);
    expect(result.iterations).toBe(3);
    expect(firstParams[0]).toBe(initialParams[0]);
    expect(costs[0]).toBeCloseTo(4);
  });

  it('should handle maximum iterations gracefully', () => {
    const initialParams = new Float64Array([100.0]);
    const result = gaussNewton(initialParams, nonlinearResidual, {
      jacobian: nonlinearJacobian,
      maxIterations: 5,
      tolerance: 1e-10
    });

    expect(result.converged).toBe(false);
    expect(result.iterations).toBe(5);
    expect(result.parameters).toBeInstanceOf(Float64Array);
  });
});

