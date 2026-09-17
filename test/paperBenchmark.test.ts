/**
 * Unit tests for the paper-benchmark protocol helpers.
 * Does not run the timed suite.
 */

import {
  computeInterquartileRange,
  computeMedian,
  computeParameterError,
  createEvaluationCounters,
  isPaperSuccess,
  runWarmedRepeats,
  timePaperTrial,
  wrapCostFunction,
  wrapGradientFunction,
  wrapResidualFunction,
  type NumericSummary
} from '../scripts/benchmark-harness';
import {
  CHAIN_ADJOINT_DECISION_VARIABLE_COUNT,
  CHAIN_PENALTY_DECISION_VARIABLE_COUNT,
  METHOD_ADJOINT_GD,
  METHOD_BFGS,
  METHOD_CMA_ES,
  METHOD_CONSTRAINED_GN,
  METHOD_CONSTRAINED_LM,
  METHOD_GAUSS_NEWTON,
  METHOD_GRADIENT_DESCENT,
  METHOD_LBFGS,
  METHOD_LEVENBERG_MARQUARDT,
  METHOD_PENALTY_GN,
  METHOD_PENALTY_LM,
  PROBLEM_CHAIN,
  PROBLEM_CIRCLE,
  PROBLEM_EXPONENTIAL_FIT,
  PROBLEM_LINEAR_LEAST_SQUARES,
  PROBLEM_ROSENBROCK,
  PROBLEM_SPHERE
} from '../scripts/paper-benchmark/constants';
import { evaluateHypotheses } from '../scripts/paper-benchmark/hypotheses';
import {
  buildMarkdown,
  escapeMarkdownTableCell,
  formatNumber
} from '../scripts/paper-benchmark/report';
import { summarizeCase } from '../scripts/paper-benchmark/summarize';
import type { PaperRow } from '../scripts/paper-benchmark/types';

const MARKDOWN_TABLE_COLUMN_COUNT = 13;
const ENVIRONMENT = {
  timestamp: '2026-01-01T00:00:00.000Z',
  nodeVersion: 'v20.0.0',
  platform: 'test',
  architecture: 'x64',
  cpuModel: 'test-cpu',
  packageVersion: '0.0.0',
  gitCommit: 'test'
};

function summary(median: number): NumericSummary {
  return { median, interquartileRange: 0 };
}

function paperRow(
  problemName: string,
  methodName: string,
  overrides: Partial<PaperRow> = {}
): PaperRow {
  return {
    problemName,
    methodName,
    decisionVariableCount: 1,
    successRate: 1,
    allTrialsSucceeded: true,
    timeMs: summary(1),
    iterations: summary(10),
    costEvaluations: summary(10),
    gradientEvaluations: summary(10),
    residualEvaluations: summary(10),
    constraintEvaluations: summary(10),
    finalCost: summary(0),
    parameterError: summary(1e-8),
    ...overrides
  };
}

function passingRows(): PaperRow[] {
  return [
    paperRow(PROBLEM_SPHERE, METHOD_GRADIENT_DESCENT, { iterations: summary(18) }),
    paperRow(PROBLEM_SPHERE, METHOD_BFGS, { iterations: summary(2) }),
    paperRow(PROBLEM_SPHERE, METHOD_LBFGS, { iterations: summary(2) }),
    paperRow(PROBLEM_SPHERE, METHOD_CMA_ES),
    paperRow(PROBLEM_ROSENBROCK, METHOD_GRADIENT_DESCENT, { iterations: summary(100) }),
    paperRow(PROBLEM_ROSENBROCK, METHOD_BFGS, { iterations: summary(2) }),
    paperRow(PROBLEM_ROSENBROCK, METHOD_LBFGS, { iterations: summary(2) }),
    paperRow(PROBLEM_ROSENBROCK, METHOD_CMA_ES),
    paperRow(PROBLEM_LINEAR_LEAST_SQUARES, METHOD_GAUSS_NEWTON),
    paperRow(PROBLEM_LINEAR_LEAST_SQUARES, METHOD_LEVENBERG_MARQUARDT),
    paperRow(PROBLEM_EXPONENTIAL_FIT, METHOD_GAUSS_NEWTON),
    paperRow(PROBLEM_EXPONENTIAL_FIT, METHOD_LEVENBERG_MARQUARDT),
    paperRow(PROBLEM_CIRCLE, METHOD_ADJOINT_GD, { iterations: summary(14) }),
    paperRow(PROBLEM_CIRCLE, METHOD_CONSTRAINED_GN, { iterations: summary(4) }),
    paperRow(PROBLEM_CIRCLE, METHOD_CONSTRAINED_LM, { iterations: summary(4) }),
    paperRow(PROBLEM_CHAIN, METHOD_ADJOINT_GD, {
      decisionVariableCount: CHAIN_ADJOINT_DECISION_VARIABLE_COUNT,
      timeMs: summary(25)
    }),
    paperRow(PROBLEM_CHAIN, METHOD_PENALTY_GN, {
      decisionVariableCount: CHAIN_PENALTY_DECISION_VARIABLE_COUNT,
      timeMs: summary(1)
    }),
    paperRow(PROBLEM_CHAIN, METHOD_PENALTY_LM, {
      decisionVariableCount: CHAIN_PENALTY_DECISION_VARIABLE_COUNT,
      timeMs: summary(1)
    })
  ];
}

function markdownColumnCount(line: string): number {
  return line.split('|').length - 2;
}

describe('paper benchmark protocol', () => {
  describe('computeMedian', () => {
    it('returns the middle value for an odd-length sample', () => {
      expect(computeMedian([1, 3, 2])).toBe(2);
    });

    it('averages the central pair for an even-length sample', () => {
      expect(computeMedian([1, 2, 3, 4])).toBe(2.5);
    });

    it('rejects an empty sample', () => {
      expect(() => computeMedian([])).toThrow('Median requires at least one value');
    });
  });

  describe('computeInterquartileRange', () => {
    it('uses linear interpolation between order statistics', () => {
      expect(computeInterquartileRange([1, 2, 3, 4])).toBeCloseTo(1.5);
    });
  });

  describe('wrapCostFunction', () => {
    it('increments the cost counter on every evaluation', () => {
      const counters = createEvaluationCounters();
      const wrapped = wrapCostFunction((parameters) => parameters[0] * parameters[0], counters);
      wrapped(new Float64Array([2]));
      wrapped(new Float64Array([3]));
      expect(counters.cost).toBe(2);
      expect(counters.gradient).toBe(0);
    });
  });

  describe('wrapGradientFunction', () => {
    it('increments the gradient counter on every evaluation', () => {
      const counters = createEvaluationCounters();
      const wrapped = wrapGradientFunction((parameters) => new Float64Array([2 * parameters[0]]), counters);
      wrapped(new Float64Array([1]));
      expect(counters.gradient).toBe(1);
      expect(counters.cost).toBe(0);
    });
  });

  describe('wrapResidualFunction', () => {
    it('increments the residual counter on every evaluation', () => {
      const counters = createEvaluationCounters();
      const wrapped = wrapResidualFunction((parameters) => new Float64Array([parameters[0]]), counters);
      wrapped(new Float64Array([1]));
      wrapped(new Float64Array([2]));
      expect(counters.residual).toBe(2);
    });
  });

  describe('isPaperSuccess', () => {
    it('accepts a parameter error inside the tolerance and ignores solver converged flags', () => {
      expect(
        isPaperSuccess({
          parameterError: 1e-4,
          parameterTolerance: 1e-3
        })
      ).toBe(true);
    });

    it('rejects a parameter error outside the tolerance', () => {
      expect(
        isPaperSuccess({
          parameterError: 1e-2,
          parameterTolerance: 1e-3
        })
      ).toBe(false);
    });

    it('requires a finite constraint norm when a constraint tolerance is set', () => {
      expect(
        isPaperSuccess({
          parameterError: 1e-8,
          parameterTolerance: 1e-3,
          constraintTolerance: 1e-6
        })
      ).toBe(false);
      expect(
        isPaperSuccess({
          parameterError: 1e-8,
          parameterTolerance: 1e-3,
          constraintNorm: 1e-8,
          constraintTolerance: 1e-6
        })
      ).toBe(true);
      expect(
        isPaperSuccess({
          parameterError: 1e-8,
          parameterTolerance: 1e-3,
          constraintNorm: 1e-4,
          constraintTolerance: 1e-6
        })
      ).toBe(false);
    });
  });

  describe('computeParameterError', () => {
    it('returns the Euclidean distance to the expected parameters', () => {
      expect(computeParameterError(new Float64Array([3, 4]), new Float64Array([0, 0]))).toBe(5);
    });

    it('rejects mismatched lengths', () => {
      expect(() => computeParameterError(new Float64Array([1]), new Float64Array([0, 0]))).toThrow(
        'Parameter error requires matching lengths'
      );
    });
  });

  describe('runWarmedRepeats', () => {
    it('discards warmup calls and returns only timed repeats', () => {
      const calls: number[] = [];
      const results = runWarmedRepeats({
        warmupCount: 1,
        repeatCount: 2,
        run: (repeatIndex) => {
          calls.push(repeatIndex);
          return repeatIndex;
        }
      });
      expect(calls).toEqual([-1, 0, 1]);
      expect(results).toEqual([0, 1]);
    });
  });

  describe('timePaperTrial', () => {
    it('records a thrown trial as unsuccessful work, not a crash', () => {
      const trial = timePaperTrial(() => {
        throw new Error('projection failed');
      });
      expect(trial.threw).toBe(true);
      expect(trial.throwMessage).toBe('projection failed');
      expect(trial.parameterError).toBe(Number.POSITIVE_INFINITY);
    });
  });

  describe('summarizeCase', () => {
    it('counts only timed repeats toward the success rate', () => {
      const row = summarizeCase({
        problemName: PROBLEM_SPHERE,
        methodName: METHOD_GRADIENT_DESCENT,
        parameterTolerance: 1e-3,
        repeatCount: 2,
        runTrial: (repeatIndex) =>
          timePaperTrial(() => ({
            iterations: 1,
            finalCost: 0,
            parameterError: repeatIndex === 1 ? 1 : 1e-8,
            decisionVariableCount: 3
          }))
      });
      expect(row.successRate).toBe(0.5);
      expect(row.allTrialsSucceeded).toBe(false);
      expect(row.decisionVariableCount).toBe(3);
    });

    it('keeps the decision-variable count from a successful trial if an earlier timed trial throws', () => {
      const row = summarizeCase({
        problemName: PROBLEM_SPHERE,
        methodName: METHOD_GRADIENT_DESCENT,
        parameterTolerance: 1e-3,
        repeatCount: 2,
        runTrial: (repeatIndex) => {
          if (repeatIndex === 0) {
            return timePaperTrial(() => {
              throw new Error('projection failed');
            });
          }
          return timePaperTrial(() => ({
            iterations: 1,
            finalCost: 0,
            parameterError: 1e-8,
            decisionVariableCount: 4
          }));
        }
      });
      expect(row.decisionVariableCount).toBe(4);
    });
  });

  describe('markdown table', () => {
    it('replaces pipe characters so GitHub-flavored Markdown keeps one cell', () => {
      expect(escapeMarkdownTableCell('||p-p*||')).toBe('//p-p*//');
    });

    it('formats missing numbers as n/a', () => {
      expect(formatNumber(Number.NaN)).toBe('n/a');
    });

    it('keeps a 13-column results table when headers would otherwise contain pipes', () => {
      const markdown = buildMarkdown(
        ENVIRONMENT,
        [
          paperRow(PROBLEM_SPHERE, METHOD_BFGS, {
            timeMs: summary(2.5),
            constraintNorm: undefined
          })
        ],
        []
      );
      const header = markdown.split('\n').find((line) => line.startsWith('| Problem |'));
      const separator = markdown.split('\n').find((line) => line.startsWith('|---'));
      const data = markdown
        .split('\n')
        .find((line) => line.startsWith('| ') && line.includes(`| ${METHOD_BFGS} |`));
      expect(header).toBeDefined();
      expect(separator).toBeDefined();
      expect(data).toBeDefined();
      expect(markdownColumnCount(header ?? '')).toBe(MARKDOWN_TABLE_COLUMN_COUNT);
      expect(markdownColumnCount(separator ?? '')).toBe(MARKDOWN_TABLE_COLUMN_COUNT);
      expect(markdownColumnCount(data ?? '')).toBe(MARKDOWN_TABLE_COLUMN_COUNT);
      expect(header).not.toContain('||');
    });
  });

  describe('evaluateHypotheses', () => {
    it('passes the spike-aligned ranking and ignores wall-clock on the chain', () => {
      const results = evaluateHypotheses(passingRows());
      expect(results.every((result) => result.passed)).toBe(true);
    });

    it('fails when quasi-Newton does not use fewer Sphere iterations than gradient descent', () => {
      const rows = passingRows().map((row) =>
        row.problemName === PROBLEM_SPHERE && row.methodName === METHOD_BFGS
          ? { ...row, iterations: summary(20) }
          : row
      );
      const sphere = evaluateHypotheses(rows).find(
        (result) => result.id === 'sphere-quasi-newton-fewer-iterations'
      );
      expect(sphere?.passed).toBe(false);
    });

    it('fails when the chain adjoint does not stay one-dimensional', () => {
      const rows = passingRows().map((row) =>
        row.problemName === PROBLEM_CHAIN && row.methodName === METHOD_ADJOINT_GD
          ? { ...row, decisionVariableCount: CHAIN_PENALTY_DECISION_VARIABLE_COUNT }
          : row
      );
      const chain = evaluateHypotheses(rows).find(
        (result) => result.id === 'chain-decision-variables-and-success'
      );
      expect(chain?.passed).toBe(false);
    });

    it('throws when a required row is missing', () => {
      expect(() => evaluateHypotheses([])).toThrow('Missing paper-benchmark row');
    });
  });
});
