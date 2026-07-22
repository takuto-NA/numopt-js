import { createFiniteDiffGradient, createFiniteDiffJacobian } from '../src/core/createGradientFunction';
import { gradientDescent } from '../src/core/gradientDescent';
import type { CostFn, ResidualFn } from '../src/core/types';

describe('createFiniteDiffGradient', () => {
    /**
     * Simple quadratic function: f(x) = x^2
     * Analytical gradient: f'(x) = 2x
     */
    const quadraticCost: CostFn = (params: Float64Array) => {
        return params[0] * params[0];
    };

    it('should create a working gradient function', () => {
        const gradientFn = createFiniteDiffGradient(quadraticCost);
        const params = new Float64Array([3.0]);
        const gradient = gradientFn(params);

        const analyticalGradient = 2 * params[0]; // 6
        expect(Math.abs(gradient[0] - analyticalGradient)).toBeLessThan(1e-4);
    });

    it('should work with gradientDescent', () => {
        const gradientFn = createFiniteDiffGradient(quadraticCost);
        const initialParams = new Float64Array([5.0]);

        const result = gradientDescent(initialParams, quadraticCost, gradientFn, {
            maxIterations: 100,
            tolerance: 1e-6,

        });

        expect(result.converged).toBe(true);
        expect(Math.abs(result.finalParameters[0])).toBeLessThan(1e-3);
        expect(result.finalCost).toBeLessThan(1e-6);
    });

    it('should respect custom step size', () => {
        const gradientFnSmallStep = createFiniteDiffGradient(quadraticCost, { stepSize: 1e-8 });
        const gradientFnDefaultStep = createFiniteDiffGradient(quadraticCost);

        const params = new Float64Array([1.0]);
        const gradientSmall = gradientFnSmallStep(params);
        const gradientDefault = gradientFnDefaultStep(params);

        // Both should be close to analytical value (2.0)
        expect(Math.abs(gradientSmall[0] - 2.0)).toBeLessThan(1e-3);
        expect(Math.abs(gradientDefault[0] - 2.0)).toBeLessThan(1e-3);
    });

    /**
     * 2D function: f(x, y) = x^2 + 2y^2
     * Analytical gradient: [2x, 4y]
     */
    const quadratic2DCost: CostFn = (params: Float64Array) => {
        return params[0] * params[0] + 2 * params[1] * params[1];
    };

    it('should work for 2D functions', () => {
        const gradientFn = createFiniteDiffGradient(quadratic2DCost);
        const params = new Float64Array([2.0, 3.0]);
        const gradient = gradientFn(params);

        const analyticalGradient = [2 * params[0], 4 * params[1]]; // [4, 12]
        expect(Math.abs(gradient[0] - analyticalGradient[0])).toBeLessThan(1e-4);
        expect(Math.abs(gradient[1] - analyticalGradient[1])).toBeLessThan(1e-4);
    });
});

describe('createFiniteDiffJacobian', () => {
    /**
     * Simple residual function: r(x) = [x^2 - 4, x - 2]
     * Analytical Jacobian: J = [[2x], [1]] (2×1 matrix)
     */
    const simpleResidual: ResidualFn = (params: Float64Array) => {
        const x = params[0];
        return new Float64Array([x * x - 4, x - 2]);
    };

    it('should create a working Jacobian function', () => {
        const jacobianFn = createFiniteDiffJacobian(simpleResidual);
        const params = new Float64Array([3.0]);
        const jacobian = jacobianFn(params);

        expect(jacobian.rows).toBe(2); // 2 residuals
        expect(jacobian.columns).toBe(1); // 1 parameter

        // Analytical Jacobian at x=3: [[6], [1]]
        const analyticalJ00 = 2 * params[0]; // 6
        const analyticalJ10 = 1.0;

        expect(Math.abs(jacobian.get(0, 0) - analyticalJ00)).toBeLessThan(1e-4);
        expect(Math.abs(jacobian.get(1, 0) - analyticalJ10)).toBeLessThan(1e-4);
    });

    it('should respect custom step size', () => {
        const jacobianFnSmallStep = createFiniteDiffJacobian(simpleResidual, { stepSize: 1e-8 });
        const jacobianFnDefaultStep = createFiniteDiffJacobian(simpleResidual);

        const params = new Float64Array([1.0]);
        const jacobianSmall = jacobianFnSmallStep(params);
        const jacobianDefault = jacobianFnDefaultStep(params);

        // Both should approximate analytical values
        expect(Math.abs(jacobianSmall.get(0, 0) - 2.0)).toBeLessThan(1e-3);
        expect(Math.abs(jacobianDefault.get(0, 0) - 2.0)).toBeLessThan(1e-3);
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

    it('should work for 2D residual functions', () => {
        const jacobianFn = createFiniteDiffJacobian(residual2D);
        const params = new Float64Array([2.0, 1.0]);
        const jacobian = jacobianFn(params);

        expect(jacobian.rows).toBe(2); // 2 residuals
        expect(jacobian.columns).toBe(2); // 2 parameters

        // Analytical Jacobian at (2, 1): [[4, 2], [1, 1]]
        const analyticalJ00 = 2 * params[0]; // 4
        const analyticalJ01 = 2 * params[1]; // 2
        const analyticalJ10 = 1.0;
        const analyticalJ11 = 1.0;

        expect(Math.abs(jacobian.get(0, 0) - analyticalJ00)).toBeLessThan(1e-4);
        expect(Math.abs(jacobian.get(0, 1) - analyticalJ01)).toBeLessThan(1e-4);
        expect(Math.abs(jacobian.get(1, 0) - analyticalJ10)).toBeLessThan(1e-4);
        expect(Math.abs(jacobian.get(1, 1) - analyticalJ11)).toBeLessThan(1e-4);
    });
});
