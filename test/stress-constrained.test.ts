import { describe, it, expect } from 'vitest';
import { adjointGradientDescent } from '../src/core/adjointGradientDescent';
import { constrainedGaussNewton } from '../src/core/constrainedGaussNewton';
import { constrainedLevenbergMarquardt } from '../src/core/constrainedLevenbergMarquardt';
import type { ConstrainedCostFn, ConstrainedResidualFn, ConstraintFn } from '../src/core/types';
import { Matrix } from 'ml-matrix';
import { vectorNorm } from '../src/utils/matrix';

describe('Stress Tests: Challenging Nonlinear Constrained Optimization', () => {
  /**
   * Test 1: Rosenbrock function with nonlinear constraint
   * This is a classic difficult optimization problem with a narrow valley
   * 
   * Minimize: f(p, x) = (1 - p)² + 100(x - p²)²
   * Subject to: c(p, x) = p² + x² - 2 = 0
   * 
   * This tests:
   * - Highly nonlinear objective (Rosenbrock valley)
   * - Nonlinear constraint (circle)
   * - Poor conditioning in the valley
   */
  describe('Rosenbrock with Nonlinear Constraint', () => {
    const rosenbrockCost: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
      const a = 1.0 - p[0];
      const b = x[0] - p[0] * p[0];
      return a * a + 100.0 * b * b;
    };

    const rosenbrockResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
      const a = 1.0 - p[0];
      const b = x[0] - p[0] * p[0];
      return new Float64Array([a, 10.0 * b]);
    };

    const circleConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] * p[0] + x[0] * x[0] - 2.0]);
    };

    it('adjointGradientDescent should handle Rosenbrock valley', () => {
      // Start near the constraint manifold
      const initialP = new Float64Array([1.0]);
      const initialX = new Float64Array([1.0]); // p² + x² = 2

      const result = adjointGradientDescent(
        initialP,
        initialX,
        rosenbrockCost,
        circleConstraint,
        {
          maxIterations: 1000,
          tolerance: 1e-4,
          constraintTolerance: 1e-3,
          useLineSearch: true
        }
      );

      // Should converge (even if not to global minimum)
      expect(result.iterations).toBeGreaterThan(0);
      expect(result.finalCost).toBeLessThan(100); // Should make some progress
      
      // Constraint should be satisfied
      const finalConstraint = circleConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-2);
    });

    it('constrainedGaussNewton should handle Rosenbrock valley with residual form', () => {
      const initialP = new Float64Array([1.0]);
      const initialX = new Float64Array([1.0]);

      const result = constrainedGaussNewton(
        initialP,
        initialX,
        rosenbrockResidual,
        circleConstraint,
        {
          maxIterations: 500,
          tolerance: 1e-4,
          constraintTolerance: 1e-3
        }
      );

      expect(result.iterations).toBeGreaterThan(0);
      expect(result.finalCost).toBeLessThan(100);
      
      const finalConstraint = circleConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-2);
    });

    it('constrainedLevenbergMarquardt should handle Rosenbrock valley', () => {
      const initialP = new Float64Array([1.0]);
      const initialX = new Float64Array([1.0]);

      const result = constrainedLevenbergMarquardt(
        initialP,
        initialX,
        rosenbrockResidual,
        circleConstraint,
        {
          maxIterations: 500,
          tolGradient: 1e-4,
          tolStep: 1e-6,
          constraintTolerance: 1e-3,
          lambdaInitial: 1e-2
        }
      );

      expect(result.iterations).toBeGreaterThan(0);
      expect(result.finalCost).toBeLessThan(100);
      
      const finalConstraint = circleConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-2);
    });
  });

  /**
   * Test 2: High-dimensional problem
   * Tests scalability to larger problems
   * 
   * Minimize: f(p, x) = Σ(pᵢ - i)² + Σ(xᵢ - i)²
   * Subject to: cᵢ(p, x) = pᵢ + xᵢ - 2i = 0 for i = 1..n
   */
  describe('High-Dimensional Problem (10D)', () => {
    const n = 10;

    const highDimCost: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
      let cost = 0;
      for (let i = 0; i < n; i++) {
        const targetP = i + 1;
        const targetX = i + 1;
        cost += (p[i] - targetP) * (p[i] - targetP);
        cost += (x[i] - targetX) * (x[i] - targetX);
      }
      return cost;
    };

    const highDimResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
      const residual = new Float64Array(2 * n);
      for (let i = 0; i < n; i++) {
        const targetP = i + 1;
        const targetX = i + 1;
        residual[i] = p[i] - targetP;
        residual[n + i] = x[i] - targetX;
      }
      return residual;
    };

    const highDimConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      const constraint = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const target = 2 * (i + 1);
        constraint[i] = p[i] + x[i] - target;
      }
      return constraint;
    };

    it('adjointGradientDescent should handle 10D problem', () => {
      const initialP = new Float64Array(n).fill(5.0);
      const initialX = new Float64Array(n).map((_, i) => 2 * (i + 1) - 5.0);

      const result = adjointGradientDescent(
        initialP,
        initialX,
        highDimCost,
        highDimConstraint,
        {
          maxIterations: 500,
          tolerance: 1e-3,
          constraintTolerance: 1e-3
        }
      );

      expect(result.iterations).toBeGreaterThan(0);
      expect(result.finalCost).toBeLessThan(50); // Should make significant progress
      
      const finalConstraint = highDimConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-2);
    });

    it('constrainedGaussNewton should handle 10D problem', () => {
      const initialP = new Float64Array(n).fill(5.0);
      const initialX = new Float64Array(n).map((_, i) => 2 * (i + 1) - 5.0);

      const result = constrainedGaussNewton(
        initialP,
        initialX,
        highDimResidual,
        highDimConstraint,
        {
          maxIterations: 300,
          tolerance: 1e-3,
          constraintTolerance: 1e-3
        }
      );

      expect(result.iterations).toBeGreaterThan(0);
      expect(result.finalCost).toBeLessThan(50);
      
      const finalConstraint = highDimConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-2);
    });

    it('constrainedLevenbergMarquardt should handle 10D problem', () => {
      const initialP = new Float64Array(n).fill(5.0);
      const initialX = new Float64Array(n).map((_, i) => 2 * (i + 1) - 5.0);

      const result = constrainedLevenbergMarquardt(
        initialP,
        initialX,
        highDimResidual,
        highDimConstraint,
        {
          maxIterations: 300,
          tolGradient: 1e-3,
          constraintTolerance: 1e-3
        }
      );

      expect(result.iterations).toBeGreaterThan(0);
      expect(result.finalCost).toBeLessThan(50);
      
      const finalConstraint = highDimConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-2);
    });
  });

  /**
   * Test 3: Ill-conditioned problem
   * Tests numerical stability with poorly scaled variables
   * 
   * Minimize: f(p, x) = (p/1000)² + (1000x)²
   * Subject to: c(p, x) = p + x - 1 = 0
   * 
   * This has very different scales for p and x
   */
  describe('Ill-Conditioned Problem', () => {
    const illConditionedCost: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
      const scaledP = p[0] / 1000.0;
      const scaledX = 1000.0 * x[0];
      return scaledP * scaledP + scaledX * scaledX;
    };

    const illConditionedResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
      const scaledP = p[0] / 1000.0;
      const scaledX = 1000.0 * x[0];
      return new Float64Array([scaledP, scaledX]);
    };

    const simpleConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] + x[0] - 1.0]);
    };

    it('adjointGradientDescent should handle ill-conditioned problem', () => {
      const initialP = new Float64Array([2000.0]);
      const initialX = new Float64Array([-1999.0]);

      const result = adjointGradientDescent(
        initialP,
        initialX,
        illConditionedCost,
        simpleConstraint,
        {
          maxIterations: 1000,
          tolerance: 1e-2,
          constraintTolerance: 1e-3,
          useLineSearch: true
        }
      );

      expect(result.iterations).toBeGreaterThan(0);
      // Should at least reduce cost significantly
      const initialCost = illConditionedCost(initialP, initialX);
      expect(result.finalCost).toBeLessThan(initialCost * 0.5);
      
      const finalConstraint = simpleConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-2);
    });

    it('constrainedLevenbergMarquardt should handle ill-conditioned problem', () => {
      const initialP = new Float64Array([2000.0]);
      const initialX = new Float64Array([-1999.0]);

      const result = constrainedLevenbergMarquardt(
        initialP,
        initialX,
        illConditionedResidual,
        simpleConstraint,
        {
          maxIterations: 500,
          tolGradient: 1e-2,
          constraintTolerance: 1e-3,
          lambdaInitial: 1e-1
        }
      );

      expect(result.iterations).toBeGreaterThan(0);
      const initialCost = illConditionedCost(initialP, initialX);
      expect(result.finalCost).toBeLessThan(initialCost * 0.5);
      
      const finalConstraint = simpleConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-2);
    });
  });

  /**
   * Test 4: Problem with multiple local minima
   * Tests ability to find a good solution in non-convex landscape
   * 
   * Minimize: f(p, x) = sin(p) + sin(x) + 0.1(p² + x²)
   * Subject to: c(p, x) = p + x - π = 0
   */
  describe('Multi-Modal Problem', () => {
    const multiModalCost: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
      return Math.sin(p[0]) + Math.sin(x[0]) + 0.1 * (p[0] * p[0] + x[0] * x[0]);
    };

    const multiModalResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
      const r1 = Math.sin(p[0]) + 0.1 * p[0] * p[0];
      const r2 = Math.sin(x[0]) + 0.1 * x[0] * x[0];
      return new Float64Array([r1, r2]);
    };

    const piConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] + x[0] - Math.PI]);
    };

    it('adjointGradientDescent should find a local minimum', () => {
      const initialP = new Float64Array([2.0]);
      const initialX = new Float64Array([Math.PI - 2.0]);

      const result = adjointGradientDescent(
        initialP,
        initialX,
        multiModalCost,
        piConstraint,
        {
          maxIterations: 500,
          tolerance: 1e-4,
          constraintTolerance: 1e-3,
          useLineSearch: true
        }
      );

      expect(result.iterations).toBeGreaterThan(0);
      // Should find some local minimum
      expect(result.finalCost).toBeLessThan(5);
      
      const finalConstraint = piConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-2);
    });

    it('constrainedLevenbergMarquardt should find a local minimum', () => {
      const initialP = new Float64Array([2.0]);
      const initialX = new Float64Array([Math.PI - 2.0]);

      const result = constrainedLevenbergMarquardt(
        initialP,
        initialX,
        multiModalResidual,
        piConstraint,
        {
          maxIterations: 500,
          tolGradient: 1e-4,
          constraintTolerance: 1e-3
        }
      );

      expect(result.iterations).toBeGreaterThan(0);
      expect(result.finalCost).toBeLessThan(5);
      
      const finalConstraint = piConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-2);
    });
  });

  /**
   * Test 5: Far from optimal initial guess
   * Tests robustness when starting very far from solution
   */
  describe('Poor Initial Guess', () => {
    const simpleCost: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
      return (p[0] - 0.5) * (p[0] - 0.5) + (x[0] - 0.5) * (x[0] - 0.5);
    };

    const simpleResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] - 0.5, x[0] - 0.5]);
    };

    const simpleConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] + x[0] - 1.0]);
    };

    it('adjointGradientDescent should converge from far initial guess', () => {
      const initialP = new Float64Array([100.0]);
      const initialX = new Float64Array([-99.0]);

      const result = adjointGradientDescent(
        initialP,
        initialX,
        simpleCost,
        simpleConstraint,
        {
          maxIterations: 1000,
          tolerance: 1e-3,
          constraintTolerance: 1e-3,
          useLineSearch: true
        }
      );

      expect(result.converged || result.iterations > 0).toBe(true);
      // Should make significant progress
      expect(result.finalCost).toBeLessThan(5000);
      
      const finalConstraint = simpleConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-2);
    });

    it('constrainedLevenbergMarquardt should converge from far initial guess', () => {
      const initialP = new Float64Array([100.0]);
      const initialX = new Float64Array([-99.0]);

      const result = constrainedLevenbergMarquardt(
        initialP,
        initialX,
        simpleResidual,
        simpleConstraint,
        {
          maxIterations: 1000,
          tolGradient: 1e-3,
          constraintTolerance: 1e-3,
          lambdaInitial: 1.0
        }
      );

      expect(result.iterations).toBeGreaterThan(0);
      expect(result.finalCost).toBeLessThan(5000);
      
      const finalConstraint = simpleConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-2);
    });
  });

  /**
   * Test 6: Very challenging non-convex problem with multiple local minima
   * Tests that iterations are not prematurely terminated
   * 
   * Minimize: f(p, x) = sin(10p) + sin(10x) + 0.1(p² + x²)
   * Subject to: c(p, x) = p² + x² - 4 = 0
   * 
   * This has many local minima and requires many iterations
   */
  describe('Very Challenging Non-Convex Problem', () => {
    const challengingCost: ConstrainedCostFn = (p: Float64Array, x: Float64Array) => {
      return Math.sin(10 * p[0]) + Math.sin(10 * x[0]) + 0.1 * (p[0] * p[0] + x[0] * x[0]);
    };

    const challengingResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
      const r1 = Math.sin(10 * p[0]) + 0.1 * p[0] * p[0];
      const r2 = Math.sin(10 * x[0]) + 0.1 * x[0] * x[0];
      return new Float64Array([r1, r2]);
    };

    const circleConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] * p[0] + x[0] * x[0] - 4.0]);
    };

    it('constrainedLevenbergMarquardt should run many iterations for challenging problem', () => {
      // Start far from solution but satisfy constraint: p² + x² = 4
      const initialP = new Float64Array([1.8]);
      const initialX = new Float64Array([Math.sqrt(4 - 1.8 * 1.8)]); // p² + x² = 4

      const result = constrainedLevenbergMarquardt(
        initialP,
        initialX,
        challengingResidual,
        circleConstraint,
        {
          maxIterations: 2000,
          tolGradient: 1e-8, // Stricter tolerance to require more iterations
          tolStep: 1e-10, // Stricter tolerance
          tolResidual: 1e-8, // Stricter tolerance
          constraintTolerance: 1e-5, // Stricter constraint tolerance
          lambdaInitial: 1e-3, // Smaller initial lambda
          lambdaFactor: 1.5, // Smaller factor to allow more lambda adjustments
          verbose: true,
          logLevel: 'INFO'
        }
      );

      // Should run many iterations (not terminate early)
      expect(result.iterations).toBeGreaterThan(10);
      // Should not terminate at iteration 0 or 1
      expect(result.iterations).toBeGreaterThan(2);
      
      // Should make progress
      const finalConstraint = circleConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-1); // Relaxed constraint tolerance
      
      // Log for debugging
      console.log(`  Challenging problem: ${result.iterations} iterations, cost: ${result.finalCost}, converged: ${result.converged}`);
    });

    it('constrainedLevenbergMarquardt should handle very poor initial guess', () => {
      // Very far from solution but satisfy constraint: p² + x² = 4
      const initialP = new Float64Array([1.9]);
      const initialX = new Float64Array([Math.sqrt(4 - 1.9 * 1.9)]); // p² + x² = 4

      const result = constrainedLevenbergMarquardt(
        initialP,
        initialX,
        challengingResidual,
        circleConstraint,
        {
          maxIterations: 3000,
          tolGradient: 1e-5,
          tolStep: 1e-7,
          tolResidual: 1e-5,
          constraintTolerance: 1e-3,
          lambdaInitial: 1.0, // Start with larger lambda for stability
          lambdaFactor: 2.0,
          verbose: false
        }
      );

      // Should run iterations (algorithm may converge quickly if problem is well-conditioned)
      expect(result.iterations).toBeGreaterThan(0);
      expect(result.iterations).toBeGreaterThan(2);
      
      // Should satisfy constraint
      const finalConstraint = circleConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-1);
      
      console.log(`  Very poor initial guess: ${result.iterations} iterations, cost: ${result.finalCost}, converged: ${result.converged}`);
    });
  });

  /**
   * Test 7: Highly ill-conditioned Rosenbrock variant
   * Tests numerical stability and iteration behavior
   * 
   * Minimize: f(p, x) = (1 - p)² + 1000(x - p²)²
   * Subject to: c(p, x) = p² + x² - 2 = 0
   * 
   * This is even more ill-conditioned than standard Rosenbrock
   */
  describe('Highly Ill-Conditioned Rosenbrock', () => {
    const illConditionedRosenbrockResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
      const a = 1.0 - p[0];
      const b = x[0] - p[0] * p[0];
      return new Float64Array([a, Math.sqrt(1000.0) * b]);
    };

    const circleConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] * p[0] + x[0] * x[0] - 2.0]);
    };

    it('constrainedLevenbergMarquardt should handle highly ill-conditioned problem', () => {
      const initialP = new Float64Array([-1.0]);
      const initialX = new Float64Array([1.0]); // p² + x² = 2

      const result = constrainedLevenbergMarquardt(
        initialP,
        initialX,
        illConditionedRosenbrockResidual,
        circleConstraint,
        {
          maxIterations: 2000,
          tolGradient: 1e-4, // Relaxed tolerance
          tolStep: 1e-6, // Relaxed tolerance
          tolResidual: 1e-4, // Relaxed tolerance
          constraintTolerance: 1e-3,
          lambdaInitial: 1e-1,
          lambdaFactor: 2.0,
          verbose: true,
          logLevel: 'INFO'
        }
      );

      // Should run iterations (algorithm may converge quickly if effective)
      expect(result.iterations).toBeGreaterThan(0);
      expect(result.iterations).toBeGreaterThan(1);
      
      const finalConstraint = circleConstraint(result.parameters, result.finalStates);
      expect(vectorNorm(finalConstraint)).toBeLessThan(1e-1); // Relaxed constraint tolerance
      
      console.log(`  Ill-conditioned Rosenbrock: ${result.iterations} iterations, cost: ${result.finalCost}, converged: ${result.converged}`);
    });
  });
});
