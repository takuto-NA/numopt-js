/**
 * JSON and Markdown writers for the paper benchmark.
 * Table cells never contain raw pipes so GitHub-flavored Markdown stays aligned.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { PaperEnvironment } from '../benchmark-harness';
import {
  CMA_ES_ROSENBROCK_SUCCESS_TOLERANCE,
  CMA_ES_SEED_COUNT,
  CMA_ES_WARMUP_SEED,
  DETERMINISTIC_REPEAT_COUNT,
  HYPOTHESIS_FAIL_LABEL,
  HYPOTHESIS_PASS_LABEL,
  EXPONENTIAL_SUCCESS_PARAMETER_TOLERANCE,
  OUTPUT_BASENAME,
  OUTPUT_DIRECTORY_NAME,
  ROSENBROCK_GRADIENT_DESCENT_MAX_ITERATIONS,
  SHARED_SOLVER_TOLERANCE,
  SUCCESS_CONSTRAINT_TOLERANCE,
  SUCCESS_PARAMETER_TOLERANCE,
  WARMUP_RUN_COUNT
} from './constants';
import type { HypothesisResult, PaperRow } from './types';

const SCIENTIFIC_LOWER_THRESHOLD = 1e-3;
const SCIENTIFIC_UPPER_THRESHOLD = 1e4;
const FORMAT_SIGNIFICANT_DIGITS = 4;
const SCIENTIFIC_FRACTION_DIGITS = 3;
const MARKDOWN_PIPE = '|';
const MARKDOWN_PIPE_REPLACEMENT = '/';
const MISSING_TABLE_VALUE = 'n/a';

export function escapeMarkdownTableCell(value: string): string {
  return value.split(MARKDOWN_PIPE).join(MARKDOWN_PIPE_REPLACEMENT);
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return MISSING_TABLE_VALUE;
  }
  if (Math.abs(value) !== 0 && (Math.abs(value) < SCIENTIFIC_LOWER_THRESHOLD || Math.abs(value) >= SCIENTIFIC_UPPER_THRESHOLD)) {
    return value.toExponential(SCIENTIFIC_FRACTION_DIGITS);
  }
  return value.toPrecision(FORMAT_SIGNIFICANT_DIGITS);
}

function tableCell(value: string): string {
  return escapeMarkdownTableCell(value);
}

export function buildMarkdown(
  environment: PaperEnvironment,
  rows: PaperRow[],
  hypotheses: HypothesisResult[]
): string {
  const lines: string[] = [
    '# Paper solver benchmark',
    '',
    'Primary metrics are evaluation counts and success. Wall-clock is machine-dependent.',
    '',
    '## Environment',
    '',
    `- Timestamp: ${environment.timestamp}`,
    `- Package: ${environment.packageVersion}`,
    `- Git commit: ${environment.gitCommit}`,
    `- Node: ${environment.nodeVersion}`,
    `- Platform: ${environment.platform} ${environment.architecture}`,
    `- CPU: ${environment.cpuModel}`,
    '',
    '## Protocol',
    '',
    `- Warmup runs discarded: ${WARMUP_RUN_COUNT}`,
    `- Deterministic repeats: ${DETERMINISTIC_REPEAT_COUNT} (median and IQR)`,
    `- CMA-ES seeds: ${CMA_ES_SEED_COUNT} (seeds 1..${CMA_ES_SEED_COUNT}; warmup uses seed ${CMA_ES_WARMUP_SEED})`,
    `- Solver tolerances: ${SHARED_SOLVER_TOLERANCE}`,
    `- Success: parameter error, never \`converged\`. Default parameter tolerance ${SUCCESS_PARAMETER_TOLERANCE}; exponential ${EXPONENTIAL_SUCCESS_PARAMETER_TOLERANCE}; CMA-ES Rosenbrock ${CMA_ES_ROSENBROCK_SUCCESS_TOLERANCE}. Constraint tolerance ${SUCCESS_CONSTRAINT_TOLERANCE}.`,
    '- Derivatives: GD / BFGS / L-BFGS / Adjoint use analytical gradients or constraint Jacobians. GN / LM / Constrained GN/LM / Penalty use numeric Jacobians.',
    '- Residual-form Adjoint (circle) increments Resid, not Cost. Scalar-cost Adjoint (chain) increments Cost.',
    `- Rosenbrock CMA-ES uses IPOP restarts. Gradient descent on Rosenbrock is allowed ${ROSENBROCK_GRADIENT_DESCENT_MAX_ITERATIONS} iterations and may hit that cap; success is still parameter error.`,
    '- Chain final cost is the original objective (x_end-1)^2 for Adjoint and Penalty, not the penalized residual.',
    '',
    '## Results',
    '',
    '| Problem | Method | Vars | Success | Time ms (med/IQR) | Iters | Cost | Grad | Resid | Constr | Final cost | Param err | Constr norm |',
    '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|'
  ];
  for (const row of rows) {
    const constraintNormText =
      row.constraintNorm === undefined ? '—' : formatNumber(row.constraintNorm.median);
    lines.push(
      `| ${tableCell(row.problemName)} | ${tableCell(row.methodName)} | ${row.decisionVariableCount} | ${row.successRate.toFixed(2)} | ` +
        `${formatNumber(row.timeMs.median)} / ${formatNumber(row.timeMs.interquartileRange)} | ` +
        `${formatNumber(row.iterations.median)} | ${formatNumber(row.costEvaluations.median)} | ` +
        `${formatNumber(row.gradientEvaluations.median)} | ${formatNumber(row.residualEvaluations.median)} | ` +
        `${formatNumber(row.constraintEvaluations.median)} | ${formatNumber(row.finalCost.median)} | ` +
        `${formatNumber(row.parameterError.median)} | ${constraintNormText} |`
    );
  }
  lines.push('', '## Hypotheses', '');
  for (const hypothesis of hypotheses) {
    lines.push(
      `- ${hypothesis.passed ? HYPOTHESIS_PASS_LABEL : HYPOTHESIS_FAIL_LABEL} \`${hypothesis.id}\`: ${hypothesis.description} (${hypothesis.detail})`
    );
  }
  lines.push('');
  return lines.join('\n');
}

export function writeReports(
  environment: PaperEnvironment,
  rows: PaperRow[],
  hypotheses: HypothesisResult[]
): { jsonPath: string; markdownPath: string } {
  const outputDirectory = path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    '..',
    '..',
    OUTPUT_DIRECTORY_NAME
  );
  fs.mkdirSync(outputDirectory, { recursive: true });
  const jsonPath = path.join(outputDirectory, `${OUTPUT_BASENAME}.json`);
  const markdownPath = path.join(outputDirectory, `${OUTPUT_BASENAME}.md`);
  const payload = {
    environment,
    protocol: {
      warmupRunCount: WARMUP_RUN_COUNT,
      deterministicRepeatCount: DETERMINISTIC_REPEAT_COUNT,
      cmaEsSeedCount: CMA_ES_SEED_COUNT,
      cmaEsWarmupSeed: CMA_ES_WARMUP_SEED,
      sharedSolverTolerance: SHARED_SOLVER_TOLERANCE,
      successParameterTolerance: SUCCESS_PARAMETER_TOLERANCE,
      successConstraintTolerance: SUCCESS_CONSTRAINT_TOLERANCE,
      exponentialSuccessParameterTolerance: EXPONENTIAL_SUCCESS_PARAMETER_TOLERANCE,
      cmaEsRosenbrockSuccessTolerance: CMA_ES_ROSENBROCK_SUCCESS_TOLERANCE,
      rosenbrockGradientDescentMaxIterations: ROSENBROCK_GRADIENT_DESCENT_MAX_ITERATIONS,
      derivativePolicy:
        'analytical for GD/BFGS/L-BFGS/Adjoint; numeric Jacobians for GN/LM/Constrained/Penalty',
      residualFormAdjointCountsResidual: true,
      rosenbrockCmaEsRestartStrategy: 'ipop',
      chainFinalCost: 'original objective (x_end-1)^2',
      primaryMetrics: ['evaluations', 'success'],
      secondaryMetrics: ['wallClockMs']
    },
    rows,
    hypotheses
  };
  fs.writeFileSync(jsonPath, `${JSON.stringify(payload, null, 2)}\n`);
  fs.writeFileSync(markdownPath, buildMarkdown(environment, rows, hypotheses));
  return { jsonPath, markdownPath };
}
