import { describe, it, expect } from 'vitest';
import { gradientDescent } from '../src/core/gradientDescent';
import { finiteDiffGradient } from '../src/core/finiteDiff';
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
            { maxIterations: 100, tolerance: 1e-6, logLevel: 'none' }
        );

        expect(result.converged).toBe(true);
        expect(Math.abs(result.parameters[0] - 3)).toBeLessThan(1e-3);
        expect(Math.abs(result.parameters[1] - 2)).toBeLessThan(1e-3);
    });

    it('should fail with incorrect parameter order (reproducing user bug)', () => {
        // This test demonstrates the user's mistake
        expect(() => {
            gradientDescent(
                new Float64Array([0, 0]),
                costFn,
                (params) => finiteDiffGradient(costFn as any, params as any), // ❌ WRONG ORDER
                { maxIterations: 100, tolerance: 1e-6, logLevel: 'none' }
            );
        }).toThrow();
    });
});
