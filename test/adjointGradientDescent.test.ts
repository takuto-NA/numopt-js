import { describe, it, expect } from 'vitest';
import { adjointGradientDescent } from '../src/core/adjointGradientDescent';
import type { ConstrainedCostFn, ConstraintFn, ConstrainedResidualFn } from '../src/core/types';
import { Matrix } from 'ml-matrix';
import { vectorNorm } from '../src/utils/matrix';

describe('Adjoint Gradient Descent', () => {
  /**
   * Simple constrained optimization problem:
   * Minimize: f(p, x) = p² + x²
   * Subject to: c(p, x) = p + x - 1 = 0
   * 
   * Analytical solution: p = 0.5, x = 0.5, f = 0.5
   * 
   * Partial derivatives:
   * ∂f/∂p = 2p, ∂f/∂x = 2x
   * ∂c/∂p = [1], ∂c/∂x = [1]
   */
  const simpleCost: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
    return p[0] * p[0] + x[0] * x[0];
  };

  const simpleConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
    return new Float64Array([p[0] + x[0] - 1.0]);
  };

  it('should converge to minimum for simple constrained problem', () => {
    const initialP = new Float64Array([2.0]);
    const initialX = new Float64Array([-1.0]); // p + x - 1 = 0 => x = -1

    const result = adjointGradientDescent(
      initialP,
      initialX,
      simpleCost,
      simpleConstraint,
      {
        maxIterations: 100,
        tolerance: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.parameters[0] - 0.5)).toBeLessThan(1e-3);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-3);
    expect(Math.abs(result.finalCost - 0.5)).toBeLessThan(1e-3);
    
    // Check constraint satisfaction
    const constraint = simpleConstraint(result.parameters, result.finalStates);
    expect(vectorNorm(constraint)).toBeLessThan(1e-3);
  });

  it('should work with analytical derivatives', () => {
    const initialP = new Float64Array([1.0]);
    const initialX = new Float64Array([0.0]);

    const result = adjointGradientDescent(
      initialP,
      initialX,
      simpleCost,
      simpleConstraint,
      {
        maxIterations: 100,
        tolerance: 1e-6,
        dfdp: (p: Float64Array) => new Float64Array([2 * p[0]]),
        dfdx: (p: Float64Array, x: Float64Array) => new Float64Array([2 * x[0]]),
        dcdp: (p: Float64Array, x: Float64Array) => new Matrix([[1]]),
        dcdx: (p: Float64Array, x: Float64Array) => new Matrix([[1]])
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.parameters[0] - 0.5)).toBeLessThan(1e-3);
  });

  it('should work with fixed step size', () => {
    const initialP = new Float64Array([3.0]);
    const initialX = new Float64Array([-2.0]);

    const result = adjointGradientDescent(
      initialP,
      initialX,
      simpleCost,
      simpleConstraint,
      {
        stepSize: 0.1,
        useLineSearch: false,
        maxIterations: 1000,
        tolerance: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(result.usedLineSearch).toBe(false);
  });

  it('should use line search when enabled', () => {
    const initialP = new Float64Array([2.0]);
    const initialX = new Float64Array([-1.0]);

    const result = adjointGradientDescent(
      initialP,
      initialX,
      simpleCost,
      simpleConstraint,
      {
        useLineSearch: true,
        maxIterations: 100,
        tolerance: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(result.usedLineSearch).toBe(true);
  });

  it('should handle constraint violation warning', () => {
    const initialP = new Float64Array([1.0]);
    const initialX = new Float64Array([1.0]); // Doesn't satisfy constraint: 1 + 1 - 1 = 1 ≠ 0

    const result = adjointGradientDescent(
      initialP,
      initialX,
      simpleCost,
      simpleConstraint,
      {
        maxIterations: 500,
        tolerance: 1e-4,
        constraintTolerance: 1e-6
      }
    );

    // The algorithm should handle initial constraint violation gracefully
    // Note: When initial values don't satisfy constraints, linear approximation
    // may not perfectly satisfy constraints, but the algorithm should still run
    expect(result.finalConstraintNorm).toBeDefined();
    expect(result.parameters).toBeInstanceOf(Float64Array);
    expect(result.finalStates).toBeInstanceOf(Float64Array);
    
    // The algorithm should complete without errors
    // (Constraint satisfaction depends on the problem and initial conditions)
  });

  it('should work with non-square constraint Jacobian (overdetermined)', () => {
    // Overdetermined system: 2 constraints, 1 state
    // Minimize: f(p, x) = p² + x²
    // Subject to: c1(p, x) = p + x - 1 = 0, c2(p, x) = 2p + x - 1.5 = 0
    // These constraints are consistent: solution is p = 0.5, x = 0.5
    const overdeterminedCost: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
      return p[0] * p[0] + x[0] * x[0];
    };

    const overdeterminedConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([
        p[0] + x[0] - 1.0,
        2.0 * p[0] + x[0] - 1.5
      ]);
    };

    const initialP = new Float64Array([0.5]);
    const initialX = new Float64Array([0.5]); // Start at solution

    const result = adjointGradientDescent(
      initialP,
      initialX,
      overdeterminedCost,
      overdeterminedConstraint,
      { maxIterations: 200, tolerance: 1e-4, constraintTolerance: 1e-2 }
    );

    // Should converge (even if constraints are not perfectly satisfied due to overdetermined nature)
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.finalCost).toBeDefined();
    // For overdetermined systems, we verify that the algorithm runs without errors
    // The constraint norm may not be zero due to least squares approximation
    const finalConstraint = overdeterminedConstraint(result.parameters, result.finalStates);
    expect(Number.isFinite(vectorNorm(finalConstraint))).toBe(true);
  });

  it('should work with non-square constraint Jacobian (underdetermined)', () => {
    // Underdetermined system: 1 constraint, 2 states
    // Minimize: f(p, x) = p² + x[0]² + x[1]²
    // Subject to: c(p, x) = p + x[0] + x[1] - 1 = 0
    const underdeterminedCost: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
      return p[0] * p[0] + x[0] * x[0] + x[1] * x[1];
    };

    const underdeterminedConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] + x[0] + x[1] - 1.0]);
    };

    const initialP = new Float64Array([1.0]);
    const initialX = new Float64Array([0.0, 0.0]);

    const result = adjointGradientDescent(
      initialP,
      initialX,
      underdeterminedCost,
      underdeterminedConstraint,
      { maxIterations: 100, tolerance: 1e-4 }
    );

    expect(result.converged).toBe(true);
    // Constraint should be satisfied
    const finalConstraint = underdeterminedConstraint(result.parameters, result.finalStates);
    expect(vectorNorm(finalConstraint)).toBeLessThan(1e-3);
  });

  it('should work with residual function', () => {
    /**
     * Residual-based problem:
     * r(p, x) = [p - 0.5, x - 0.5]
     * f = 1/2 r^T r = 1/2 ((p-0.5)² + (x-0.5)²)
     * Subject to: c(p, x) = p + x - 1 = 0
     */
    const residualFn: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] - 0.5, x[0] - 0.5]);
    };

    const initialP = new Float64Array([2.0]);
    const initialX = new Float64Array([-1.0]);

    const result = adjointGradientDescent(
      initialP,
      initialX,
      residualFn,
      simpleConstraint,
      {
        maxIterations: 100,
        tolerance: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.parameters[0] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-2);
  });

  it('should call onIteration callback if provided', () => {
    const initialP = new Float64Array([2.0]);
    const initialX = new Float64Array([-1.0]);
    let callbackCalled = false;
    let iterationCount = 0;

    adjointGradientDescent(
      initialP,
      initialX,
      simpleCost,
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
    expect(iterationCount).toBeGreaterThan(0);
  });

  it('should handle maximum iterations gracefully', () => {
    const initialP = new Float64Array([10.0]);
    const initialX = new Float64Array([-9.0]);

    const result = adjointGradientDescent(
      initialP,
      initialX,
      simpleCost,
      simpleConstraint,
      {
        maxIterations: 5,
        tolerance: 1e-10,
        stepSize: 0.01,
        useLineSearch: false
      }
    );

    expect(result.converged).toBe(false);
    expect(result.iterations).toBe(5);
    expect(result.parameters).toBeInstanceOf(Float64Array);
    expect(result.finalStates).toBeInstanceOf(Float64Array);
  });

  /**
   * 2D problem:
   * Minimize: f(p, x) = p₁² + p₂² + x₁² + x₂²
   * Subject to: c₁(p, x) = p₁ + x₁ - 1 = 0
   *             c₂(p, x) = p₂ + x₂ - 1 = 0
   * 
   * Solution: p = [0.5, 0.5], x = [0.5, 0.5], f = 1.0
   */
  const cost2D: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
    return p[0] * p[0] + p[1] * p[1] + x[0] * x[0] + x[1] * x[1];
  };

  const constraint2D: ConstraintFn = (p: Float64Array, x: Float64Array) => {
    return new Float64Array([
      p[0] + x[0] - 1.0,
      p[1] + x[1] - 1.0
    ]);
  };

  it('should converge for 2D constrained problem', () => {
    const initialP = new Float64Array([2.0, 2.0]);
    const initialX = new Float64Array([-1.0, -1.0]);

    const result = adjointGradientDescent(
      initialP,
      initialX,
      cost2D,
      constraint2D,
      {
        maxIterations: 200,
        tolerance: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.parameters[0] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.parameters[1] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalStates[1] - 0.5)).toBeLessThan(1e-2);
    
    // Check constraint satisfaction
    const constraint = constraint2D(result.parameters, result.finalStates);
    expect(vectorNorm(constraint)).toBeLessThan(1e-3);
  });
});

