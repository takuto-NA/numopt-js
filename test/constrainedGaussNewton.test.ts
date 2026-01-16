import { constrainedGaussNewton } from '../src/core/constrainedGaussNewton';
import type { ConstrainedResidualFn, ConstraintFn } from '../src/core/types';
import { Matrix } from 'ml-matrix';
import { vectorNorm } from '../src/utils/matrix';

describe('Constrained Gauss-Newton Method', () => {
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

    const result = constrainedGaussNewton(
      initialP,
      initialX,
      simpleResidual,
      simpleConstraint,
      {
        maxIterations: 100,
        tolerance: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(1e-3);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-3);
    expect(result.finalCost).toBeLessThan(1e-5);
    
    // Check constraint satisfaction
    const constraint = simpleConstraint(result.finalParameters, result.finalStates);
    expect(vectorNorm(constraint)).toBeLessThan(1e-3);
  });

  it('should work with analytical derivatives', () => {
    const initialP = new Float64Array([1.0]);
    const initialX = new Float64Array([0.0]);

    const result = constrainedGaussNewton(
      initialP,
      initialX,
      simpleResidual,
      simpleConstraint,
      {
        maxIterations: 100,
        tolerance: 1e-6,
        // r_p: (2×1) - derivative of [p-0.5, x-0.5] w.r.t. p
        drdp: (p: Float64Array, x: Float64Array) => new Matrix([[1], [0]]),
        // r_x: (2×1) - derivative of [p-0.5, x-0.5] w.r.t. x
        drdx: (p: Float64Array, x: Float64Array) => new Matrix([[0], [1]]),
        dcdp: (p: Float64Array, x: Float64Array) => new Matrix([[1]]),
        dcdx: (p: Float64Array, x: Float64Array) => new Matrix([[1]])
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(1e-2);
  });

  it('should handle constraint violation warning', () => {
    const initialP = new Float64Array([1.0]);
    const initialX = new Float64Array([1.0]); // Doesn't satisfy constraint: 1 + 1 - 1 = 1 ≠ 0

    const result = constrainedGaussNewton(
      initialP,
      initialX,
      simpleResidual,
      simpleConstraint,
      {
        maxIterations: 200,
        tolerance: 1e-4,
        constraintTolerance: 1e-6
      }
    );

    // The algorithm should handle initial constraint violation gracefully
    expect(result.finalConstraintNorm).toBeDefined();
    expect(result.finalParameters).toBeInstanceOf(Float64Array);
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

    const result = constrainedGaussNewton(
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

  it('should call onIteration callback if provided', () => {
    const initialP = new Float64Array([2.0]);
    const initialX = new Float64Array([-1.0]);
    let callbackCalled = false;
    let iterationCount = 0;

    constrainedGaussNewton(
      initialP,
      initialX,
      simpleResidual,
      simpleConstraint,
      {
        maxIterations: 10,
        tolerance: 1e-6,
        onIteration: (iteration, cost, params) => {
          callbackCalled = true;
          iterationCount = iteration;
          expect(typeof cost).toBe('number');
          expect(params).toBeInstanceOf(Float64Array);
        }
      }
    );

    expect(callbackCalled).toBe(true);
    expect(iterationCount).toBeGreaterThanOrEqual(0);
  });

  it('should handle maximum iterations gracefully', () => {
    const initialP = new Float64Array([10.0]);
    const initialX = new Float64Array([-9.0]);

    const result = constrainedGaussNewton(
      initialP,
      initialX,
      simpleResidual,
      simpleConstraint,
      {
        maxIterations: 3,
        tolerance: 1e-12 // Very strict tolerance to prevent convergence
      }
    );

    // May converge or not depending on initial conditions
    expect(result.iterations).toBeLessThanOrEqual(3);
    expect(result.finalParameters).toBeInstanceOf(Float64Array);
    expect(result.finalStates).toBeInstanceOf(Float64Array);
  });

  /**
   * 2D problem:
   * Residual: r(p, x) = [p₁ - 0.5, p₂ - 0.5, x₁ - 0.5, x₂ - 0.5]
   * Subject to: c₁(p, x) = p₁ + x₁ - 1 = 0
   *             c₂(p, x) = p₂ + x₂ - 1 = 0
   * 
   * Solution: p = [0.5, 0.5], x = [0.5, 0.5]
   */
  const residual2D: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
    return new Float64Array([
      p[0] - 0.5,
      p[1] - 0.5,
      x[0] - 0.5,
      x[1] - 0.5
    ]);
  };

  const constraint2D: ConstraintFn = (p: Float64Array, x: Float64Array) => {
    return new Float64Array([
      p[0] + x[0] - 1.0,
      p[1] + x[1] - 1.0
    ]);
  };

  it('should converge for 2D constrained least squares', () => {
    const initialP = new Float64Array([2.0, 2.0]);
    const initialX = new Float64Array([-1.0, -1.0]);

    const result = constrainedGaussNewton(
      initialP,
      initialX,
      residual2D,
      constraint2D,
      {
        maxIterations: 200,
        tolerance: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalParameters[1] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalStates[1] - 0.5)).toBeLessThan(1e-2);
    
    // Check constraint satisfaction
    const constraint = constraint2D(result.finalParameters, result.finalStates);
    expect(vectorNorm(constraint)).toBeLessThan(1e-3);
  });
});

