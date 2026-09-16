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
    expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(1e-3);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-3);
    expect(Math.abs(result.finalCost - 0.5)).toBeLessThan(1e-3);
    
    // Check constraint satisfaction
    const constraint = simpleConstraint(result.finalParameters, result.finalStates);
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
    expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(1e-3);
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

  it('should reduce constraint violation from a mildly infeasible start', () => {
    const initialP = new Float64Array([0.8]);
    const initialX = new Float64Array([0.8]);
    const initialConstraintNorm = vectorNorm(simpleConstraint(initialP, initialX));

    const result = adjointGradientDescent(
      initialP,
      initialX,
      simpleCost,
      simpleConstraint,
      {
        maxIterations: 500,
        tolerance: 1e-4,
        constraintTolerance: 1e-6,
        useLineSearch: true
      }
    );

    expect(result.finalConstraintNorm).toBeLessThan(initialConstraintNorm);
    expect(Number.isFinite(result.finalCost)).toBe(true);
  });

  it('rejects an overdetermined constraint Jacobian', () => {
    const overdeterminedCost: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
      return p[0] * p[0] + x[0] * x[0];
    };

    const overdeterminedConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([
        p[0] + x[0] - 1.0,
        2.0 * p[0] + x[0] - 1.5
      ]);
    };

    expect(() =>
      adjointGradientDescent(
        new Float64Array([0.5]),
        new Float64Array([0.5]),
        overdeterminedCost,
        overdeterminedConstraint
      )
    ).toThrow(/square implicit-state system/);
  });

  it('rejects a non-square analytical ∂c/∂x before projecting', () => {
    expect(() =>
      adjointGradientDescent(
        new Float64Array([0.5]),
        new Float64Array([0.5]),
        simpleCost,
        simpleConstraint,
        {
          dcdx: () => new Matrix([[1, 0]])
        }
      )
    ).toThrow(/square constraint Jacobian/);
  });

  it('rejects an underdetermined constraint Jacobian', () => {
    const underdeterminedCost: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
      return p[0] * p[0] + x[0] * x[0] + x[1] * x[1];
    };

    const underdeterminedConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] + x[0] + x[1] - 1.0]);
    };

    expect(() =>
      adjointGradientDescent(
        new Float64Array([1.0]),
        new Float64Array([0.0, 0.0]),
        underdeterminedCost,
        underdeterminedConstraint
      )
    ).toThrow(/square implicit-state system/);
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
    expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-2);
  });

  it('should accept options.regularization without breaking the simple problem', () => {
    // WHY: Locks the public regularization knob (and that square Tikhonov is not double-applied).
    const result = adjointGradientDescent(
      new Float64Array([2.0]),
      new Float64Array([-1.0]),
      simpleCost,
      simpleConstraint,
      {
        maxIterations: 200,
        tolerance: 1e-4,
        regularization: 1e-6
      }
    );

    expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-2);
    expect(result.finalConstraintNorm).toBeLessThan(1e-3);
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
    expect(result.finalParameters).toBeInstanceOf(Float64Array);
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
    expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalParameters[1] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalStates[1] - 0.5)).toBeLessThan(1e-2);
    
    // Check constraint satisfaction
    const constraint = constraint2D(result.finalParameters, result.finalStates);
    expect(vectorNorm(constraint)).toBeLessThan(1e-3);
  });

  it('takes one reduced-gradient step on the linear problem', () => {
    const result = adjointGradientDescent(
      new Float64Array([2.0]),
      new Float64Array([-1.0]),
      simpleCost,
      simpleConstraint,
      {
        maxIterations: 1,
        tolerance: 0,
        useLineSearch: false,
        stepSize: 0.01,
        dfdp: (p: Float64Array) => new Float64Array([2 * p[0]]),
        dfdx: (_p: Float64Array, x: Float64Array) => new Float64Array([2 * x[0]]),
        dcdp: () => new Matrix([[1]]),
        dcdx: () => new Matrix([[1]])
      }
    );

    // Reduced gradient at p=2 is 6, so p <- 2 - 0.01 * 6 = 1.94, x = 1 - p.
    expect(result.finalParameters[0]).toBeCloseTo(1.94, 8);
    expect(result.finalStates[0]).toBeCloseTo(-0.94, 8);
    expect(result.finalConstraintNorm).toBeLessThan(1e-8);
  });

  it('converges on the nonlinear circle from a feasible start', () => {
    const circleCost: ConstrainedCostFn = (p, x) => (p[0] - 1) ** 2 + (x[0] - 1) ** 2;
    const circleConstraint: ConstraintFn = (p, x) =>
      new Float64Array([p[0] * p[0] + x[0] * x[0] - 2.0]);
    const initialP = 0.2;
    const initialX = Math.sqrt(2 - initialP * initialP);

    const result = adjointGradientDescent(
      new Float64Array([initialP]),
      new Float64Array([initialX]),
      circleCost,
      circleConstraint,
      {
        maxIterations: 300,
        tolerance: 1e-6,
        constraintTolerance: 1e-6,
        useLineSearch: true
      }
    );

    expect(result.converged).toBe(true);
    expect(result.finalParameters[0]).toBeCloseTo(1, 2);
    expect(result.finalStates[0]).toBeCloseTo(1, 2);
    expect(result.finalConstraintNorm).toBeLessThan(1e-6);
  });

  it('projects a modest off-manifold guess onto the circle before optimizing', () => {
    const circleCost: ConstrainedCostFn = (p, x) => (p[0] - 1) ** 2 + (x[0] - 1) ** 2;
    const circleConstraint: ConstraintFn = (p, x) =>
      new Float64Array([p[0] * p[0] + x[0] * x[0] - 2.0]);

    const result = adjointGradientDescent(
      new Float64Array([0.2]),
      new Float64Array([1.0]),
      circleCost,
      circleConstraint,
      {
        maxIterations: 300,
        tolerance: 1e-6,
        constraintTolerance: 1e-6,
        useLineSearch: true
      }
    );

    expect(result.converged).toBe(true);
    expect(result.finalParameters[0]).toBeCloseTo(1, 2);
    expect(result.finalStates[0]).toBeCloseTo(1, 2);
    expect(result.finalConstraintNorm).toBeLessThan(1e-6);
  });

  it('rejects an initial point with no real implicit state', () => {
    const circleCost: ConstrainedCostFn = (p, x) => (p[0] - 1) ** 2 + (x[0] - 1) ** 2;
    const circleConstraint: ConstraintFn = (p, x) =>
      new Float64Array([p[0] * p[0] + x[0] * x[0] - 2.0]);

    let thrown: unknown;
    try {
      adjointGradientDescent(
        new Float64Array([1.5]),
        new Float64Array([0.5]),
        circleCost,
        circleConstraint,
        { constraintTolerance: 1e-6 }
      );
    } catch (error) {
      thrown = error;
    }

    expect(thrown).toBeInstanceOf(Error);
    const message = (thrown as Error).message;
    const originalInfeasibleCircleNorm = vectorNorm(
      circleConstraint(new Float64Array([1.5]), new Float64Array([0.5]))
    );
    expect(message).toMatch(/Failed to restore feasible states/);
    const residualMatch = message.match(/\|\|c\(p,x\)\|\|=([0-9.eE+-]+)/);
    expect(residualMatch).not.toBeNull();
    expect(Number(residualMatch?.[1])).toBeGreaterThan(originalInfeasibleCircleNorm);
  });

  it('optimizes a short implicit chain with one parameter and many states', () => {
    const stateCount = 8;
    const targetEnd = 1.0;
    const expectedParameter = targetEnd / stateCount;
    const chainCost: ConstrainedCostFn = (_p, x) => (x[stateCount - 1] - targetEnd) ** 2;
    const chainConstraint: ConstraintFn = (p, x) => {
      const constraint = new Float64Array(stateCount);
      constraint[0] = x[0] - p[0];
      for (let index = 1; index < stateCount; index++) {
        constraint[index] = x[index] - x[index - 1] - p[0];
      }
      return constraint;
    };

    const initialParameter = 0.4;
    const initialStates = new Float64Array(stateCount);
    for (let index = 0; index < stateCount; index++) {
      initialStates[index] = (index + 1) * initialParameter;
    }

    const result = adjointGradientDescent(
      new Float64Array([initialParameter]),
      initialStates,
      chainCost,
      chainConstraint,
      {
        maxIterations: 200,
        tolerance: 1e-8,
        constraintTolerance: 1e-8,
        useLineSearch: true
      }
    );

    expect(result.converged).toBe(true);
    expect(result.finalParameters[0]).toBeCloseTo(expectedParameter, 3);
    expect(result.finalParameters.length).toBe(1);
    expect(result.finalStates.length).toBe(stateCount);
    expect(result.finalConstraintNorm).toBeLessThan(1e-6);
  });
});

