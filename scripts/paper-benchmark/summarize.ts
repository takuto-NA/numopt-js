/**
 * Aggregates warmed paper-benchmark trials into a result row.
 * Success uses parameter / constraint error, never `converged`.
 */

import {
  isPaperSuccess,
  runWarmedRepeats,
  summarizeNumeric,
  type NumericSummary,
  type PaperTrial
} from '../benchmark-harness';
import { WARMUP_RUN_COUNT } from './constants';
import type { PaperCase, PaperRow } from './types';

function summarizeFinite(values: number[]): NumericSummary {
  const finiteValues = values.filter((value) => Number.isFinite(value));
  if (finiteValues.length === 0) {
    return { median: Number.NaN, interquartileRange: Number.NaN };
  }
  return summarizeNumeric(finiteValues);
}

function trialSucceeded(
  trial: PaperTrial,
  parameterTolerance: number,
  constraintTolerance?: number
): boolean {
  if (trial.threw) {
    return false;
  }
  return isPaperSuccess({
    parameterError: trial.parameterError,
    parameterTolerance,
    constraintNorm: trial.constraintNorm,
    constraintTolerance
  });
}

export function summarizeCase(paperCase: PaperCase): PaperRow {
  const trials = runWarmedRepeats({
    warmupCount: WARMUP_RUN_COUNT,
    repeatCount: paperCase.repeatCount,
    run: paperCase.runTrial
  });
  const successes = trials.filter((trial) =>
    trialSucceeded(trial, paperCase.parameterTolerance, paperCase.constraintTolerance)
  ).length;
  const constraintValues = trials
    .map((trial) => trial.constraintNorm)
    .filter((value): value is number => value !== undefined);
  return {
    problemName: paperCase.problemName,
    methodName: paperCase.methodName,
    decisionVariableCount:
      trials.find((trial) => !trial.threw)?.decisionVariableCount ??
      trials[0]?.decisionVariableCount ??
      0,
    successRate: successes / paperCase.repeatCount,
    allTrialsSucceeded: successes === paperCase.repeatCount,
    timeMs: summarizeFinite(trials.map((trial) => trial.elapsedMs)),
    iterations: summarizeFinite(trials.map((trial) => trial.iterations)),
    costEvaluations: summarizeFinite(trials.map((trial) => trial.counters.cost)),
    gradientEvaluations: summarizeFinite(trials.map((trial) => trial.counters.gradient)),
    residualEvaluations: summarizeFinite(trials.map((trial) => trial.counters.residual)),
    constraintEvaluations: summarizeFinite(trials.map((trial) => trial.counters.constraint)),
    finalCost: summarizeFinite(trials.map((trial) => trial.finalCost)),
    parameterError: summarizeFinite(trials.map((trial) => trial.parameterError)),
    constraintNorm: constraintValues.length > 0 ? summarizeFinite(constraintValues) : undefined
  };
}
