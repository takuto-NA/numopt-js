import { describe, it, expect } from 'vitest';
import { finiteDiffGradient, finiteDiffJacobian } from '../src/core/finiteDiff';
import type { CostFn, ResidualFn } from '../src/core/types';

describe('Numerical Differentiation', () => {
  describe('finiteDiffGradient', () => {
    /**
     * Simple quadratic function: f(x) = x^2
     * Analytical gradient: f'(x) = 2x
     */
    const quadraticCost: CostFn = (params: Float64Array) => {
      return params[0] * params[0];
    };

    it('should approximate gradient accurately for quadratic function', () => {
      const params = new Float64Array([3.0]);
      const numericalGradient = finiteDiffGradient(params, quadraticCost);
      const analyticalGradient = 2 * params[0]; // 2 * 3 = 6

      expect(Math.abs(numericalGradient[0] - analyticalGradient)).toBeLessThan(1e-4);
    });

    /**
     * 2D function: f(x, y) = x^2 + 2y^2
     * Analytical gradient: [2x, 4y]
     */
    const quadratic2DCost: CostFn = (params: Float64Array) => {
      return params[0] * params[0] + 2 * params[1] * params[1];
    };

    it('should approximate gradient for 2D function', () => {
      const params = new Float64Array([2.0, 3.0]);
      const numericalGradient = finiteDiffGradient(params, quadratic2DCost);
      const analyticalGradient = [2 * params[0], 4 * params[1]]; // [4, 12]

      expect(Math.abs(numericalGradient[0] - analyticalGradient[0])).toBeLessThan(1e-4);
      expect(Math.abs(numericalGradient[1] - analyticalGradient[1])).toBeLessThan(1e-4);
    });

    it('should respect custom step size', () => {
      const params = new Float64Array([1.0]);
      const gradientSmallStep = finiteDiffGradient(params, quadraticCost, { stepSize: 1e-8 });
      const gradientDefaultStep = finiteDiffGradient(params, quadraticCost);

      // Both should be close to analytical value (2.0)
      expect(Math.abs(gradientSmallStep[0] - 2.0)).toBeLessThan(1e-3);
      expect(Math.abs(gradientDefaultStep[0] - 2.0)).toBeLessThan(1e-3);
    });
  });

  describe('finiteDiffJacobian', () => {
    /**
     * Simple residual function: r(x) = [x^2 - 4, x - 2]
     * Analytical Jacobian: J = [[2x, 1]]
     * Actually, for 1D parameter: J = [[2x], [1]] (2×1 matrix)
     */
    const simpleResidual: ResidualFn = (params: Float64Array) => {
      const x = params[0];
      return new Float64Array([x * x - 4, x - 2]);
    };

    it('should approximate Jacobian for simple residual function', () => {
      const params = new Float64Array([3.0]);
      const numericalJacobian = finiteDiffJacobian(simpleResidual, params);

      expect(numericalJacobian.rows).toBe(2); // 2 residuals
      expect(numericalJacobian.columns).toBe(1); // 1 parameter

      // Analytical Jacobian at x=3: [[6], [1]]
      const analyticalJ00 = 2 * params[0]; // 6
      const analyticalJ10 = 1.0;

      expect(Math.abs(numericalJacobian.get(0, 0) - analyticalJ00)).toBeLessThan(1e-4);
      expect(Math.abs(numericalJacobian.get(1, 0) - analyticalJ10)).toBeLessThan(1e-4);
    });

    /**
     * 2D residual function: r(x, y) = [x^2 + y^2 - 5, x + y - 3]
     * Analytical Jacobian: J = [[2x, 2y], [1, 1]]
     */
    const residual2D: ResidualFn = (params: Float64Array) => {
      const x = params[0];
      const y = params[1];
      return new Float64Array([x * x + y * y - 5, x + y - 3]);
    };

    it('should approximate Jacobian for 2D residual function', () => {
      const params = new Float64Array([2.0, 1.0]);
      const numericalJacobian = finiteDiffJacobian(residual2D, params);

      expect(numericalJacobian.rows).toBe(2); // 2 residuals
      expect(numericalJacobian.columns).toBe(2); // 2 parameters

      // Analytical Jacobian at (2, 1): [[4, 2], [1, 1]]
      const analyticalJ00 = 2 * params[0]; // 4
      const analyticalJ01 = 2 * params[1]; // 2
      const analyticalJ10 = 1.0;
      const analyticalJ11 = 1.0;

      expect(Math.abs(numericalJacobian.get(0, 0) - analyticalJ00)).toBeLessThan(1e-4);
      expect(Math.abs(numericalJacobian.get(0, 1) - analyticalJ01)).toBeLessThan(1e-4);
      expect(Math.abs(numericalJacobian.get(1, 0) - analyticalJ10)).toBeLessThan(1e-4);
      expect(Math.abs(numericalJacobian.get(1, 1) - analyticalJ11)).toBeLessThan(1e-4);
    });

    it('should respect custom step size', () => {
      const params = new Float64Array([1.0]);
      const jacobianSmallStep = finiteDiffJacobian(simpleResidual, params, { stepSize: 1e-8 });
      const jacobianDefaultStep = finiteDiffJacobian(simpleResidual, params);

      // Both should approximate analytical values
      expect(Math.abs(jacobianSmallStep.get(0, 0) - 2.0)).toBeLessThan(1e-3);
      expect(Math.abs(jacobianDefaultStep.get(0, 0) - 2.0)).toBeLessThan(1e-3);
    });
  });
});

