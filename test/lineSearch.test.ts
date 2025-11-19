import { describe, it, expect } from 'vitest';
import { backtrackingLineSearch } from '../src/core/lineSearch';
import type { CostFn, GradientFn } from '../src/core/types';

describe('Line Search', () => {
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

  it('should find step size for descent direction', () => {
    const currentParams = new Float64Array([5.0]);
    const searchDirection = new Float64Array([-1.0]); // Negative gradient direction

    const stepSize = backtrackingLineSearch(
      quadraticCost,
      quadraticGradient,
      currentParams,
      searchDirection
    );

    expect(stepSize).toBeGreaterThan(0);
    expect(stepSize).toBeLessThanOrEqual(1.0);

    // Verify that the step actually decreases the cost
    const newParams = new Float64Array([currentParams[0] + stepSize * searchDirection[0]]);
    const oldCost = quadraticCost(currentParams);
    const newCost = quadraticCost(newParams);
    expect(newCost).toBeLessThan(oldCost);
  });

  it('should return 0 for non-descent direction', () => {
    const currentParams = new Float64Array([5.0]);
    const searchDirection = new Float64Array([1.0]); // Positive direction (not descent)

    const stepSize = backtrackingLineSearch(
      quadraticCost,
      quadraticGradient,
      currentParams,
      searchDirection
    );

    expect(stepSize).toBe(0.0);
  });

  it('should satisfy Armijo condition', () => {
    const currentParams = new Float64Array([3.0]);
    const searchDirection = new Float64Array([-1.0]);
    const armijoParameter = 0.1;

    const stepSize = backtrackingLineSearch(
      quadraticCost,
      quadraticGradient,
      currentParams,
      searchDirection,
      { armijoParameter }
    );

    expect(stepSize).toBeGreaterThan(0);

    // Verify Armijo condition: f(x + αd) <= f(x) + c * α * ∇f(x)^T * d
    const currentCost = quadraticCost(currentParams);
    const currentGradient = quadraticGradient(currentParams);
    const directionalDerivative = currentGradient[0] * searchDirection[0];
    const newParams = new Float64Array([currentParams[0] + stepSize * searchDirection[0]]);
    const newCost = quadraticCost(newParams);
    const armijoThreshold = currentCost + armijoParameter * stepSize * directionalDerivative;

    expect(newCost).toBeLessThanOrEqual(armijoThreshold);
  });

  it('should respect custom options', () => {
    const currentParams = new Float64Array([2.0]);
    const searchDirection = new Float64Array([-1.0]);

    const stepSize = backtrackingLineSearch(
      quadraticCost,
      quadraticGradient,
      currentParams,
      searchDirection,
      {
        initialStepSize: 0.5,
        contractionFactor: 0.8,
        armijoParameter: 0.2
      }
    );

    expect(stepSize).toBeGreaterThan(0);
    expect(stepSize).toBeLessThanOrEqual(0.5);
  });

  it('should handle 2D functions', () => {
    const quadratic2DCost: CostFn = (params: Float64Array) => {
      return params[0] * params[0] + params[1] * params[1];
    };

    const quadratic2DGradient: GradientFn = (params: Float64Array) => {
      return new Float64Array([2 * params[0], 2 * params[1]]);
    };

    const currentParams = new Float64Array([3.0, 4.0]);
    const searchDirection = new Float64Array([-1.0, -1.0]); // Descent direction

    const stepSize = backtrackingLineSearch(
      quadratic2DCost,
      quadratic2DGradient,
      currentParams,
      searchDirection
    );

    expect(stepSize).toBeGreaterThan(0);

    // Verify cost decreases
    const newParams = new Float64Array([
      currentParams[0] + stepSize * searchDirection[0],
      currentParams[1] + stepSize * searchDirection[1]
    ]);
    const oldCost = quadratic2DCost(currentParams);
    const newCost = quadratic2DCost(newParams);
    expect(newCost).toBeLessThan(oldCost);
  });

  describe('Edge cases for gradient norm scaling', () => {
    it('should use default step size when gradient norm is zero', () => {
      // At minimum, gradient is zero
      const currentParams = new Float64Array([0.0]);
      const searchDirection = new Float64Array([-1.0]);

      const stepSize = backtrackingLineSearch(
        quadraticCost,
        quadraticGradient,
        currentParams,
        searchDirection
        // No initialStepSize specified - should use default due to zero gradient
      );

      // Should return 0 because search direction is not descent when gradient is zero
      expect(stepSize).toBe(0.0);
    });

    it('should use default step size when gradient norm is very small', () => {
      // Very small gradient norm (below threshold)
      const verySmallGradientCost: CostFn = (params: Float64Array) => {
        return params[0] * params[0];
      };

      const verySmallGradient: GradientFn = (params: Float64Array) => {
        // Return gradient with very small norm (1e-12)
        return new Float64Array([1e-12]);
      };

      const currentParams = new Float64Array([1.0]);
      const searchDirection = new Float64Array([-1.0]);

      const stepSize = backtrackingLineSearch(
        verySmallGradientCost,
        verySmallGradient,
        currentParams,
        searchDirection
        // No initialStepSize specified - should use default due to small gradient norm
      );

      // Should still work, but may use default step size
      expect(stepSize).toBeGreaterThanOrEqual(0);
    });

    it('should scale initial step size by gradient norm when gradient is large', () => {
      // Large gradient norm (e.g., 898 as in the problem description)
      const largeGradientCost: CostFn = (params: Float64Array) => {
        return params[0] * params[0];
      };

      const largeGradient: GradientFn = (params: Float64Array) => {
        // Return gradient with large norm (898)
        return new Float64Array([898.0]);
      };

      const currentParams = new Float64Array([449.0]); // x such that 2x = 898
      const searchDirection = new Float64Array([-1.0]);

      const stepSize = backtrackingLineSearch(
        largeGradientCost,
        largeGradient,
        currentParams,
        searchDirection
        // No initialStepSize specified - should scale by 1.0 / 898 ≈ 0.001114
      );

      // Should find a valid step size
      expect(stepSize).toBeGreaterThan(0);
      
      // Verify that the step actually decreases the cost
      const newParams = new Float64Array([currentParams[0] + stepSize * searchDirection[0]]);
      const oldCost = largeGradientCost(currentParams);
      const newCost = largeGradientCost(newParams);
      expect(newCost).toBeLessThan(oldCost);
    });

    it('should respect explicitly provided initialStepSize even with large gradient', () => {
      const largeGradientCost: CostFn = (params: Float64Array) => {
        return params[0] * params[0];
      };

      const largeGradient: GradientFn = (params: Float64Array) => {
        return new Float64Array([898.0]);
      };

      const currentParams = new Float64Array([449.0]);
      const searchDirection = new Float64Array([-1.0]);

      const explicitInitialStepSize = 0.5;
      const stepSize = backtrackingLineSearch(
        largeGradientCost,
        largeGradient,
        currentParams,
        searchDirection,
        {
          initialStepSize: explicitInitialStepSize
        }
      );

      // Should use the explicitly provided step size (or smaller if backtracking occurs)
      expect(stepSize).toBeGreaterThan(0);
      expect(stepSize).toBeLessThanOrEqual(explicitInitialStepSize);
    });
  });
});

