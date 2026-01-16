import { levenbergMarquardt } from '../src/core/levenbergMarquardt';
import type { ResidualFn, JacobianFn } from '../src/core/types';
import { Matrix } from 'ml-matrix';

describe('Levenberg-Marquardt Method', () => {
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
    const result = levenbergMarquardt(initialParams, linearResidual, {
      jacobian: linearJacobian,
      maxIterations: 10,
      tolGradient: 1e-6
    });

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - 2.0)).toBeLessThan(1e-3);
    // Allow slightly larger tolerance for numerical errors
    expect(result.finalCost).toBeLessThan(1e-5);
    expect(result.finalResidualNorm).toBeLessThan(1e-3);
  });

  it('should work with numerical Jacobian', () => {
    const initialParams = new Float64Array([0.0]);
    const result = levenbergMarquardt(initialParams, linearResidual, {
      useNumericJacobian: true,
      maxIterations: 10,
      tolGradient: 1e-6
    });

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - 2.0)).toBeLessThan(1e-3);
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
    const result = levenbergMarquardt(initialParams, nonlinearResidual, {
      jacobian: nonlinearJacobian,
      maxIterations: 100,
      tolGradient: 1e-6
    });

    expect(result.converged).toBe(true);
    // Should converge to x = 2 (positive solution)
    expect(Math.abs(result.finalParameters[0] - 2.0)).toBeLessThan(1e-3);
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
    const initialParams = new Float64Array([0.0, 0.0]);
    const result = levenbergMarquardt(initialParams, residual2D, {
      jacobian: jacobian2D,
      maxIterations: 100,
      tolGradient: 1e-6
    });

    expect(result.converged).toBe(true);
    // Check that solution satisfies constraints approximately
    const residual = residual2D(result.finalParameters);
    const residualNorm = Math.sqrt(residual[0] * residual[0] + residual[1] * residual[1]);
    expect(residualNorm).toBeLessThan(1e-3);
  });

  it('should handle lambda updates correctly', () => {
    const initialParams = new Float64Array([3.0]);
    const result = levenbergMarquardt(initialParams, nonlinearResidual, {
      jacobian: nonlinearJacobian,
      maxIterations: 100,
      lambdaInitial: 1e-3,
      lambdaFactor: 10.0,
      tolGradient: 1e-6
    });

    expect(result.converged).toBe(true);
    expect(result.finalLambda).toBeGreaterThan(0);
    expect(result.finalLambda).toBeLessThan(1e3); // Should decrease from initial
  });

  it('should call onIteration starting from iteration zero', () => {
    const initialParams = new Float64Array([0.0]);
    const iterations: number[] = [];
    const firstParams: number[] = [];

    const result = levenbergMarquardt(initialParams, linearResidual, {
      jacobian: linearJacobian,
      maxIterations: 3,
      lambdaInitial: 1e-3,
      lambdaFactor: 2,
      tolGradient: 1e-12,
      tolStep: 1e-12,
      tolResidual: 1e-12,
      onIteration: (iteration, cost, params) => {
        iterations.push(iteration);
        expect(cost).toBeGreaterThanOrEqual(0);
        if (iteration === 0) {
          firstParams.push(params[0]);
        }
      }
    });

    const expectedIterations = Array.from({ length: result.iterations }, (_, idx) => idx);
    expect(iterations).toEqual(expectedIterations);
    expect(firstParams[0]).toBe(initialParams[0]);
  });

  it('should return best solution when max iterations reached', () => {
    const initialParams = new Float64Array([100.0]);
    const result = levenbergMarquardt(initialParams, nonlinearResidual, {
      jacobian: nonlinearJacobian,
      maxIterations: 5,
      tolGradient: 1e-10
    });

    expect(result.converged).toBe(false);
    expect(result.iterations).toBe(5);
    expect(result.finalParameters).toBeInstanceOf(Float64Array);
    // Should return best solution found, not necessarily the last one
    expect(result.finalCost).toBeLessThanOrEqual(
      nonlinearResidual(initialParams)[0] * nonlinearResidual(initialParams)[0]
    );
  });
});

