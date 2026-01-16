import { gradientDescent } from '../src/core/gradientDescent';
import { finiteDiffGradient } from '../src/core/finiteDiff';
import { createFiniteDiffGradient } from '../src/core/createGradientFunction';
import type { CostFn } from '../src/core/types';

describe('Integration: finiteDiffGradient with gradientDescent', () => {
    /**
     * Test case from user report:
     * f(x, y) = (x - 3)^2 + (y - 2)^2
     * Minimum at (3, 2)
     */
    const costFn: CostFn = (params: Float64Array) => {
        return Math.pow(params[0] - 3, 2) + Math.pow(params[1] - 2, 2);
    };

    it('should work with finiteDiffGradient (correct parameter order)', () => {
        const result = gradientDescent(
            new Float64Array([0, 0]),
            costFn,
            (params) => finiteDiffGradient(params, costFn), // ✅ CORRECT ORDER
            { maxIterations: 100, tolerance: 1e-6 }
        );

        expect(result.converged).toBe(true);
        expect(Math.abs(result.finalParameters[0] - 3)).toBeLessThan(1e-3);
        expect(Math.abs(result.finalParameters[1] - 2)).toBeLessThan(1e-3);
    });

    it('should work with createFiniteDiffGradient helper (recommended approach)', () => {
        // This is the recommended approach - no parameter order confusion!
        const gradientFn = createFiniteDiffGradient(costFn);

        const result = gradientDescent(
            new Float64Array([0, 0]),
            costFn,
            gradientFn,
            { maxIterations: 100, tolerance: 1e-6 }
        );

        expect(result.converged).toBe(true);
        expect(Math.abs(result.finalParameters[0] - 3)).toBeLessThan(1e-3);
        expect(Math.abs(result.finalParameters[1] - 2)).toBeLessThan(1e-3);
    });

    it('should reproduce exact user scenario from bug report', () => {
        // This is the EXACT pattern from the user's working manual gradient code
        const manualGradient = (params: Float64Array) =>
            new Float64Array([2 * (params[0] - 3), 2 * (params[1] - 2)]);

        const resultManual = gradientDescent(
            new Float64Array([0, 0]),
            costFn,
            manualGradient,
            { maxIterations: 100, tolerance: 1e-6 }
        );

        // Now with finiteDiffGradient (correct order)
        const resultFiniteDiff = gradientDescent(
            new Float64Array([0, 0]),
            costFn,
            (params) => finiteDiffGradient(params, costFn),
            { maxIterations: 100, tolerance: 1e-6 }
        );

        // Now with helper function
        const resultHelper = gradientDescent(
            new Float64Array([0, 0]),
            costFn,
            createFiniteDiffGradient(costFn),
            { maxIterations: 100, tolerance: 1e-6 }
        );

        // All three approaches should converge to the same solution
        expect(resultManual.converged).toBe(true);
        expect(resultFiniteDiff.converged).toBe(true);
        expect(resultHelper.converged).toBe(true);

        // All should find the minimum at (3, 2)
        expect(Math.abs(resultManual.finalParameters[0] - 3)).toBeLessThan(1e-3);
        expect(Math.abs(resultFiniteDiff.finalParameters[0] - 3)).toBeLessThan(1e-3);
        expect(Math.abs(resultHelper.finalParameters[0] - 3)).toBeLessThan(1e-3);

        expect(Math.abs(resultManual.finalParameters[1] - 2)).toBeLessThan(1e-3);
        expect(Math.abs(resultFiniteDiff.finalParameters[1] - 2)).toBeLessThan(1e-3);
        expect(Math.abs(resultHelper.finalParameters[1] - 2)).toBeLessThan(1e-3);
    });

    it('should fail with incorrect parameter order (reproducing user bug)', () => {
        // This test demonstrates the user's mistake
        expect(() => {
            gradientDescent(
                new Float64Array([0, 0]),
                costFn,
                (params) => finiteDiffGradient(costFn as any, params as any), // ❌ WRONG ORDER
                { maxIterations: 100, tolerance: 1e-6 }
            );
        }).toThrow();
    });
});
