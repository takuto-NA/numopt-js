/**
 * Tests for result formatting utilities.
 */

import {
  formatOptimizationResult,
  formatGradientDescentResult,
  formatLevenbergMarquardtResult,
  formatCmaEsResult,
  formatConstrainedGaussNewtonResult,
  formatConstrainedLevenbergMarquardtResult,
  formatAdjointGradientDescentResult,
  formatResult,
  printResult
} from '../src/utils/resultFormatter.js';
import type {
  OptimizationResult,
  GradientDescentResult,
  LevenbergMarquardtResult,
  CmaEsResult,
  ConstrainedGaussNewtonResult,
  ConstrainedLevenbergMarquardtResult,
  AdjointGradientDescentResult
} from '../src/core/types.js';

describe('ResultFormatter', () => {
  describe('formatOptimizationResult', () => {
    it('should format basic optimization result', () => {
      const result: OptimizationResult = {
        finalParameters: new Float64Array([1.0, 2.0]),
        parameters: new Float64Array([1.0, 2.0]),
        iterations: 10,
        converged: true,
        finalCost: 0.001,
        finalGradientNorm: 1e-6,
        finalResidualNorm: 0.01
      };

      const formatted = formatOptimizationResult(result);
      expect(formatted).toContain('=== Optimization Results ===');
      expect(formatted).toContain('p = 1');
      expect(formatted).toContain('x = 2');
      expect(formatted).toContain('Converged: true');
      expect(formatted).toContain('Iterations: 10');
      expect(formatted).toContain('0.001');
    });

    it('should format result without optional fields', () => {
      const result: OptimizationResult = {
        finalParameters: new Float64Array([0.5]),
        parameters: new Float64Array([0.5]),
        iterations: 5,
        converged: false,
        finalCost: 1.5
      };

      const formatted = formatOptimizationResult(result);
      expect(formatted).toContain('p = 0.5');
      expect(formatted).toContain('Converged: false');
      expect(formatted).not.toContain('Final gradient norm');
    });

    it('should respect options', () => {
      const result: OptimizationResult = {
        finalParameters: new Float64Array([1.0]),
        parameters: new Float64Array([1.0]),
        iterations: 10,
        converged: true,
        finalCost: 0.001
      };

      const formatted = formatOptimizationResult(result, {
        showSectionHeaders: false,
        showExecutionTime: true,
        elapsedTimeMs: 100
      });
      expect(formatted).not.toContain('=== Optimization Results ===');
      expect(formatted).toContain('Execution time');
    });
  });

  describe('formatGradientDescentResult', () => {
    it('should format gradient descent result with line search info', () => {
      const result: GradientDescentResult = {
        finalParameters: new Float64Array([0.0, 0.0]),
        parameters: new Float64Array([0.0, 0.0]),
        iterations: 20,
        converged: true,
        finalCost: 1e-8,
        finalGradientNorm: 1e-7,
        usedLineSearch: true
      };

      const formatted = formatGradientDescentResult(result);
      expect(formatted).toContain('Used line search: true');
    });
  });

  describe('formatLevenbergMarquardtResult', () => {
    it('should format Levenberg-Marquardt result with lambda', () => {
      const result: LevenbergMarquardtResult = {
        finalParameters: new Float64Array([2.0]),
        parameters: new Float64Array([2.0]),
        iterations: 15,
        converged: true,
        finalCost: 0.0001,
        finalResidualNorm: 0.01,
        finalLambda: 0.001
      };

      const formatted = formatLevenbergMarquardtResult(result);
      expect(formatted).toContain('Final lambda');
      expect(formatted).toContain('0.001');
    });
  });

  describe('formatCmaEsResult', () => {
    it('should format CMA-ES result with evaluation and step-size fields', () => {
      const result: CmaEsResult = {
        finalParameters: new Float64Array([0.1, -0.2]),
        parameters: new Float64Array([0.1, -0.2]),
        iterations: 40,
        converged: true,
        finalCost: 0.05,
        functionEvaluations: 800,
        finalStepSize: 0.01,
        finalMaxStdDev: 0.02,
        stopReason: 'targetCost'
      };

      const formatted = formatCmaEsResult(result);
      expect(formatted).toContain('Function evaluations');
      expect(formatted).toContain('800');
      expect(formatted).toContain('Final step size');
    });
  });

  describe('printResult', () => {
    it('should print a formatted result via console.log', () => {
      const result: OptimizationResult = {
        finalParameters: new Float64Array([1.0]),
        parameters: new Float64Array([1.0]),
        iterations: 3,
        converged: true,
        finalCost: 0.0
      };

      const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      printResult(result);
      expect(logSpy).toHaveBeenCalledTimes(1);
      expect(String(logSpy.mock.calls[0][0])).toContain('Optimization Results');
      logSpy.mockRestore();
    });
  });

  describe('formatConstrainedGaussNewtonResult', () => {
    it('should format constrained Gauss-Newton result with states', () => {
      const result: ConstrainedGaussNewtonResult = {
        finalParameters: new Float64Array([0.5]),
        parameters: new Float64Array([0.5]),
        finalStates: new Float64Array([0.5]),
        iterations: 8,
        converged: true,
        finalCost: 0.0,
        finalResidualNorm: 1e-6,
        finalConstraintNorm: 1e-8
      };

      const formatted = formatConstrainedGaussNewtonResult(result);
      expect(formatted).toContain('Optimized states');
      expect(formatted).toContain('||c(p, x)||');
    });
  });

  describe('formatConstrainedLevenbergMarquardtResult', () => {
    it('should format constrained Levenberg-Marquardt result', () => {
      const result: ConstrainedLevenbergMarquardtResult = {
        finalParameters: new Float64Array([0.5]),
        parameters: new Float64Array([0.5]),
        finalStates: new Float64Array([0.5]),
        iterations: 12,
        converged: true,
        finalCost: 0.0,
        finalResidualNorm: 1e-6,
        finalConstraintNorm: 1e-8,
        finalLambda: 0.0001
      };

      const formatted = formatConstrainedLevenbergMarquardtResult(result);
      expect(formatted).toContain('Final lambda');
      expect(formatted).toContain('||c(p, x)||');
    });
  });

  describe('formatAdjointGradientDescentResult', () => {
    it('should format adjoint gradient descent result', () => {
      const result: AdjointGradientDescentResult = {
        finalParameters: new Float64Array([0.5]),
        parameters: new Float64Array([0.5]),
        finalStates: new Float64Array([0.5]),
        iterations: 25,
        converged: true,
        finalCost: 0.5,
        finalGradientNorm: 1e-6,
        usedLineSearch: true,
        finalConstraintNorm: 1e-8
      };

      const formatted = formatAdjointGradientDescentResult(result);
      expect(formatted).toContain('Used line search: true');
      expect(formatted).toContain('||c(p, x)||');
    });
  });

  describe('formatResult (overloaded)', () => {
    it('should format OptimizationResult', () => {
      const result: OptimizationResult = {
        finalParameters: new Float64Array([1.0]),
        parameters: new Float64Array([1.0]),
        iterations: 10,
        converged: true,
        finalCost: 0.001
      };

      const formatted = formatResult(result);
      expect(formatted).toContain('Optimization Results');
    });

    it('should format GradientDescentResult', () => {
      const result: GradientDescentResult = {
        finalParameters: new Float64Array([1.0]),
        parameters: new Float64Array([1.0]),
        iterations: 10,
        converged: true,
        finalCost: 0.001,
        usedLineSearch: true
      };

      const formatted = formatResult(result);
      expect(formatted).toContain('Used line search');
    });

    it('should format LevenbergMarquardtResult', () => {
      const result: LevenbergMarquardtResult = {
        finalParameters: new Float64Array([1.0]),
        parameters: new Float64Array([1.0]),
        iterations: 10,
        converged: true,
        finalCost: 0.001,
        finalResidualNorm: 0.01,
        finalLambda: 0.001
      };

      const formatted = formatResult(result);
      expect(formatted).toContain('Final lambda');
    });

    it('should format ConstrainedGaussNewtonResult', () => {
      const result: ConstrainedGaussNewtonResult = {
        finalParameters: new Float64Array([0.5]),
        parameters: new Float64Array([0.5]),
        finalStates: new Float64Array([0.5]),
        iterations: 10,
        converged: true,
        finalCost: 0.0,
        finalConstraintNorm: 1e-8
      };

      const formatted = formatResult(result);
      expect(formatted).toContain('Optimized states');
    });
  });

  describe('parameter array formatting', () => {
    it('should format small arrays individually', () => {
      const result: OptimizationResult = {
        finalParameters: new Float64Array([1.0, 2.0, 3.0]),
        parameters: new Float64Array([1.0, 2.0, 3.0]),
        iterations: 10,
        converged: true,
        finalCost: 0.001
      };

      const formatted = formatOptimizationResult(result);
      expect(formatted).toContain('p = 1');
      expect(formatted).toContain('x = 2');
      expect(formatted).toContain('y = 3');
    });

    it('should format medium arrays as array', () => {
      const result: OptimizationResult = {
        finalParameters: new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]),
        parameters: new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]),
        iterations: 10,
        converged: true,
        finalCost: 0.001
      };

      const formatted = formatOptimizationResult(result);
      expect(formatted).toMatch(/\[.*1\.0.*2\.0.*3\.0.*4\.0.*5\.0.*\]/);
    });

    it('should truncate large arrays', () => {
      const largeArray = new Float64Array(20);
      for (let i = 0; i < 20; i++) {
        largeArray[i] = i + 1;
      }

      const result: OptimizationResult = {
        finalParameters: largeArray,
        parameters: largeArray,
        iterations: 10,
        converged: true,
        finalCost: 0.001
      };

      const formatted = formatOptimizationResult(result);
      expect(formatted).toContain('... and');
      expect(formatted).toContain('more');
    });
  });

  describe('execution time formatting', () => {
    it('should include execution time when provided', () => {
      const result: OptimizationResult = {
        finalParameters: new Float64Array([1.0]),
        parameters: new Float64Array([1.0]),
        iterations: 10,
        converged: true,
        finalCost: 0.001
      };

      const formatted = formatOptimizationResult(result, {
        showExecutionTime: true,
        elapsedTimeMs: 100.5
      });
      expect(formatted).toContain('Execution time');
      expect(formatted).toContain('100.50');
      expect(formatted).toContain('Time per iteration');
    });

    it('should not include execution time by default', () => {
      const result: OptimizationResult = {
        finalParameters: new Float64Array([1.0]),
        parameters: new Float64Array([1.0]),
        iterations: 10,
        converged: true,
        finalCost: 0.001
      };

      const formatted = formatOptimizationResult(result);
      expect(formatted).not.toContain('Execution time');
    });
  });
});

