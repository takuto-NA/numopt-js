/**
 * Row, trial-case, and hypothesis shapes for the paper benchmark.
 */

import type { NumericSummary, PaperTrial } from '../benchmark-harness';

export type PaperRow = {
  problemName: string;
  methodName: string;
  decisionVariableCount: number;
  successRate: number;
  allTrialsSucceeded: boolean;
  timeMs: NumericSummary;
  iterations: NumericSummary;
  costEvaluations: NumericSummary;
  gradientEvaluations: NumericSummary;
  residualEvaluations: NumericSummary;
  constraintEvaluations: NumericSummary;
  finalCost: NumericSummary;
  parameterError: NumericSummary;
  constraintNorm?: NumericSummary;
};

export type HypothesisResult = {
  id: string;
  description: string;
  passed: boolean;
  detail: string;
};

export type PaperCase = {
  problemName: string;
  methodName: string;
  parameterTolerance: number;
  constraintTolerance?: number;
  repeatCount: number;
  runTrial: (repeatIndex: number) => PaperTrial;
};
