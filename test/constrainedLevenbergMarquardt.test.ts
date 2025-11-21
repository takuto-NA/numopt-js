import { describe, it, expect } from 'vitest';
import { constrainedLevenbergMarquardt } from '../src/core/constrainedLevenbergMarquardt';
import type { ConstrainedResidualFn, ConstraintFn } from '../src/core/types';
import { Matrix } from 'ml-matrix';
import { vectorNorm } from '../src/utils/matrix';

describe('Constrained Levenberg-Marquardt Method', () => {
  /**
   * Simple constrained least squares problem:
   * Minimize: f(p, x) = 1/2 ((p - 0.5)² + (x - 0.5)²)
   * Subject to: c(p, x) = p + x - 1 = 0
   * 
   * Residual: r(p, x) = [p - 0.5, x - 0.5]
   * Solution: p = 0.5, x = 0.5, f = 0
   */
  const simpleResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
    return new Float64Array([p[0] - 0.5, x[0] - 0.5]);
  };

  const simpleConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
    return new Float64Array([p[0] + x[0] - 1.0]);
  };

  it('should converge for simple constrained least squares', () => {
    const initialP = new Float64Array([2.0]);
    const initialX = new Float64Array([-1.0]); // p + x - 1 = 0 => x = -1

    const result = constrainedLevenbergMarquardt(
      initialP,
      initialX,
      simpleResidual,
      simpleConstraint,
      {
        maxIterations: 100,
        tolGradient: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.parameters[0] - 0.5)).toBeLessThan(1e-3);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-3);
    expect(result.finalCost).toBeLessThan(1e-5);
    expect(result.finalLambda).toBeGreaterThan(0);
    
    // Check constraint satisfaction
    const constraint = simpleConstraint(result.parameters, result.finalStates);
    expect(vectorNorm(constraint)).toBeLessThan(1e-3);
  });

  it('should work with numerical derivatives', () => {
    const initialP = new Float64Array([1.0]);
    const initialX = new Float64Array([0.0]);

    const result = constrainedLevenbergMarquardt(
      initialP,
      initialX,
      simpleResidual,
      simpleConstraint,
      {
        maxIterations: 100,
        tolGradient: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.parameters[0] - 0.5)).toBeLessThan(1e-2);
  });

  it('should work with analytical derivatives', () => {
    const initialP = new Float64Array([1.0]);
    const initialX = new Float64Array([0.0]);

    const result = constrainedLevenbergMarquardt(
      initialP,
      initialX,
      simpleResidual,
      simpleConstraint,
      {
        maxIterations: 100,
        tolGradient: 1e-6,
        // r_p: (2×1) - derivative of [p-0.5, x-0.5] w.r.t. p
        drdp: (p: Float64Array, x: Float64Array) => new Matrix([[1], [0]]),
        // r_x: (2×1) - derivative of [p-0.5, x-0.5] w.r.t. x
        drdx: (p: Float64Array, x: Float64Array) => new Matrix([[0], [1]]),
        dcdp: (p: Float64Array, x: Float64Array) => new Matrix([[1]]),
        dcdx: (p: Float64Array, x: Float64Array) => new Matrix([[1]])
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.parameters[0] - 0.5)).toBeLessThan(1e-2);
  });

  it('should handle lambda updates correctly', () => {
    const initialP = new Float64Array([3.0]);
    const initialX = new Float64Array([-2.0]);

    const result = constrainedLevenbergMarquardt(
      initialP,
      initialX,
      simpleResidual,
      simpleConstraint,
      {
        maxIterations: 100,
        lambdaInitial: 1e-3,
        lambdaFactor: 10.0,
        tolGradient: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(result.finalLambda).toBeGreaterThan(0);
    expect(result.finalLambda).toBeLessThan(1e3); // Should decrease from initial
  });

  it('should handle constraint violation warning', () => {
    const initialP = new Float64Array([1.0]);
    const initialX = new Float64Array([1.0]); // Doesn't satisfy constraint: 1 + 1 - 1 = 1 ≠ 0

    const result = constrainedLevenbergMarquardt(
      initialP,
      initialX,
      simpleResidual,
      simpleConstraint,
      {
        maxIterations: 200,
        tolGradient: 1e-4,
        constraintTolerance: 1e-6
      }
    );

    // The algorithm should handle initial constraint violation gracefully
    expect(result.finalConstraintNorm).toBeDefined();
    expect(result.parameters).toBeInstanceOf(Float64Array);
    expect(result.finalStates).toBeInstanceOf(Float64Array);
  });

  it('should work with non-square constraint Jacobian', () => {
    // Non-square constraint Jacobian is now supported
    const nonSquareConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      // Returns 2 constraints but only 1 state (overdetermined)
      return new Float64Array([p[0] + x[0] - 1.0, 2.0 * p[0] + x[0] - 1.5]);
    };

    const initialP = new Float64Array([0.5]);
    const initialX = new Float64Array([0.5]);

    const result = constrainedLevenbergMarquardt(
      initialP,
      initialX,
      simpleResidual,
      nonSquareConstraint,
      { maxIterations: 100, tolerance: 1e-4, constraintTolerance: 1e-2 }
    );

    // Should run without errors (even if constraints are not perfectly satisfied due to overdetermined nature)
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.finalCost).toBeDefined();
  });

  it('should call onIteration starting from iteration zero', () => {
    const initialP = new Float64Array([2.0]);
    const initialX = new Float64Array([-1.0]);
    const iterations: number[] = [];
    const firstParams: number[] = [];

    const result = constrainedLevenbergMarquardt(
      initialP,
      initialX,
      simpleResidual,
      simpleConstraint,
      {
        maxIterations: 10,
        tolGradient: 1e-6,
        onIteration: (iteration, cost, params) => {
          iterations.push(iteration);
          expect(cost).toBeGreaterThanOrEqual(0);
          if (iteration === 0) {
            firstParams.push(params[0]);
          }
        }
      }
    );

    const expectedIterations = Array.from({ length: result.iterations }, (_, idx) => idx);
    expect(iterations).toEqual(expectedIterations);
    expect(firstParams[0]).toBe(initialP[0]);
  });

  it('should return best solution when max iterations reached', () => {
    const initialP = new Float64Array([10.0]);
    const initialX = new Float64Array([-9.0]);

    const result = constrainedLevenbergMarquardt(
      initialP,
      initialX,
      simpleResidual,
      simpleConstraint,
      {
        maxIterations: 3,
        tolGradient: 1e-12, // Very strict tolerance to prevent convergence
        tolStep: 1e-12,
        tolResidual: 1e-12
      }
    );

    // May converge or not depending on initial conditions
    expect(result.iterations).toBeLessThanOrEqual(3);
    expect(result.parameters).toBeInstanceOf(Float64Array);
    expect(result.finalStates).toBeInstanceOf(Float64Array);
    // Should return best solution found, not necessarily the last one
    expect(result.finalCost).toBeLessThanOrEqual(
      simpleResidual(initialP, initialX).reduce((sum, r) => sum + r * r, 0) / 2
    );
  });
});

