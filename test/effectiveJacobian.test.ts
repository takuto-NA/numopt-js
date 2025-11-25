import { describe, it, expect } from 'vitest';
import { computeEffectiveJacobian } from '../src/core/effectiveJacobian';
import type { ConstrainedResidualFn, ConstraintFn } from '../src/core/types';
import { Matrix } from 'ml-matrix';
import { Logger } from '../src/core/logger';

describe('Effective Jacobian Computation', () => {
  /**
   * Simple test case:
   * Residual: r(p, x) = [p - 0.5, x - 0.5]
   * Constraint: c(p, x) = p + x - 1 = 0
   * 
   * Partial derivatives:
   * r_p = [[1, 0], [0, 0]] (2×1 matrix, but actually 2×1 for 1 parameter)
   * Actually: r_p is (2×1) for 1 parameter
   * r_x = [[0, 0], [0, 1]] (2×1 matrix)
   * Actually: r_x is (2×1) for 1 state
   * c_p = [[1]] (1×1)
   * c_x = [[1]] (1×1)
   * 
   * Effective Jacobian: J_eff = r_p - r_x c_x^-1 c_p
   * = [[1], [0]] - [[0], [1]] * 1 * [[1]]
   * = [[1], [0]] - [[0], [1]]
   * = [[1], [-1]]
   * 
   * Wait, let me recalculate:
   * r_p: (2×1) = [[1], [0]] (derivative of [p-0.5, x-0.5] w.r.t. p)
   * r_x: (2×1) = [[0], [1]] (derivative of [p-0.5, x-0.5] w.r.t. x)
   * c_p: (1×1) = [[1]]
   * c_x: (1×1) = [[1]]
   * 
   * J_eff = r_p - r_x c_x^-1 c_p
   * = [[1], [0]] - [[0], [1]] * 1 * [[1]]
   * = [[1], [0]] - [[0], [1]]
   * = [[1], [-1]]
   */
  const simpleResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
    return new Float64Array([p[0] - 0.5, x[0] - 0.5]);
  };

  const simpleConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
    return new Float64Array([p[0] + x[0] - 1.0]);
  };

  it('should compute effective Jacobian correctly', () => {
    const parameters = new Float64Array([1.0]);
    const states = new Float64Array([0.0]);
    const logger = new Logger();

    const effectiveJacobian = computeEffectiveJacobian(
      parameters,
      states,
      simpleResidual,
      simpleConstraint,
      {},
      logger,
      'test'
    );

    // Effective Jacobian should be (2×1) matrix
    expect(effectiveJacobian.rows).toBe(2);
    expect(effectiveJacobian.columns).toBe(1);
    
    // Check approximate values (numerical differentiation has some error)
    // J_eff[0, 0] should be approximately 1 (r_p[0, 0] = 1, r_x[0, 0] = 0)
    expect(Math.abs(effectiveJacobian.get(0, 0) - 1.0)).toBeLessThan(1e-3);
    // J_eff[1, 0] should be approximately -1 (r_p[1, 0] = 0, r_x[1, 0] = 1, c_x^-1 c_p = 1)
    expect(Math.abs(effectiveJacobian.get(1, 0) + 1.0)).toBeLessThan(1e-3);
  });

  it('should work with analytical derivatives', () => {
    const parameters = new Float64Array([1.0]);
    const states = new Float64Array([0.0]);
    const logger = new Logger();

    const effectiveJacobian = computeEffectiveJacobian(
      parameters,
      states,
      simpleResidual,
      simpleConstraint,
      {
        drdp: (p: Float64Array, x: Float64Array) => new Matrix([[1], [0]]),
        drdx: (p: Float64Array, x: Float64Array) => new Matrix([[0], [1]]),
        dcdp: (p: Float64Array, x: Float64Array) => new Matrix([[1]]),
        dcdx: (p: Float64Array, x: Float64Array) => new Matrix([[1]])
      },
      logger,
      'test'
    );

    expect(effectiveJacobian.rows).toBe(2);
    expect(effectiveJacobian.columns).toBe(1);
    // With analytical derivatives, should be exact
    // Allow small floating error from matrix inversion
    expect(Math.abs(effectiveJacobian.get(0, 0) - 1.0)).toBeLessThan(1e-8);
    expect(Math.abs(effectiveJacobian.get(1, 0) + 1.0)).toBeLessThan(1e-8);
  });

  it('should work with non-square constraint Jacobian', () => {
    // Non-square constraint Jacobian is now supported
    const nonSquareConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] + x[0] - 1.0, 2.0 * p[0] + x[0] - 1.5]);
    };

    const parameters = new Float64Array([0.5]);
    const states = new Float64Array([0.5]);
    const logger = new Logger();

    // Should compute effective Jacobian without errors
    const effectiveJacobian = computeEffectiveJacobian(
      parameters,
      states,
      simpleResidual,
      nonSquareConstraint,
      {},
      logger,
      'test'
    );

    expect(effectiveJacobian).toBeDefined();
    expect(effectiveJacobian.rows).toBeGreaterThan(0);
    expect(effectiveJacobian.columns).toBeGreaterThan(0);
  });

  it('should handle 2D case', () => {
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

    const parameters = new Float64Array([1.0, 1.0]);
    const states = new Float64Array([0.0, 0.0]);
    const logger = new Logger();

    const effectiveJacobian = computeEffectiveJacobian(
      parameters,
      states,
      residual2D,
      constraint2D,
      {},
      logger,
      'test'
    );

    // Effective Jacobian should be (4×2) matrix (4 residuals, 2 parameters)
    expect(effectiveJacobian.rows).toBe(4);
    expect(effectiveJacobian.columns).toBe(2);
  });
});

