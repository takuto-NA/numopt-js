/**
 * Format/print helpers for optimizer result objects (`print*` / `format*` overloads).
 */

import type {
  OptimizationResult,
  GradientDescentResult,
  LevenbergMarquardtResult,
  CmaEsResult,
  AdjointGradientDescentResult,
  ConstrainedGaussNewtonResult,
  ConstrainedLevenbergMarquardtResult
} from '../core/types.js';
import { formatNumberWithPrecision } from './formatting.js';

/**
 * Options for customizing result formatting.
 */
export interface ResultFormatterOptions {
  /**
   * Whether to show section headers (e.g., "=== Optimization Results ===").
   * Default: true
   */
  showSectionHeaders?: boolean;

  /**
   * Whether to show execution time information.
   * Default: false (requires elapsedTimeMs to be provided)
   */
  showExecutionTime?: boolean;

  /**
   * Execution time in milliseconds (for display).
   * Only used if showExecutionTime is true.
   */
  elapsedTimeMs?: number;

  /**
   * Maximum number of parameters to display before truncating.
   * Default: 10
   */
  maxParametersToShow?: number;

  /**
   * Precision for parameter values.
   * Default: 6
   */
  parameterPrecision?: number;

  /**
   * Precision for cost and norm values.
   * Default: 8
   */
  costPrecision?: number;

  /**
   * Precision for constraint violation values.
   * Default: 10
   */
  constraintPrecision?: number;
}

const DEFAULT_OPTIONS: Required<ResultFormatterOptions> = {
  showSectionHeaders: true,
  showExecutionTime: false,
  elapsedTimeMs: 0,
  maxParametersToShow: 10,
  parameterPrecision: 6,
  costPrecision: 8,
  constraintPrecision: 10
};

/**
 * Formats a parameter array for display.
 * Automatically switches between individual and array format based on size.
 */
function formatParameters(
  parameters: Float64Array,
  options: Required<ResultFormatterOptions>
): string {
  const length = parameters.length;
  const maxShow = options.maxParametersToShow;
  const precision = options.parameterPrecision;

  if (length === 0) {
    return '[]';
  }

  // For small arrays (≤3 elements), show individually with labels
  if (length <= 3) {
    const labels = ['p', 'x', 'y'];
    return Array.from(parameters)
      .map((value, index) => {
        const label = labels[index] || `param${index}`;
        return `${label} = ${formatNumberWithPrecision(value, precision)}`;
      })
      .join(', ');
  }

  // For medium arrays (4-10 elements), show as array
  if (length <= maxShow) {
    const formatted = Array.from(parameters)
      .map(value => formatNumberWithPrecision(value, precision))
      .join(', ');
    return `[${formatted}]`;
  }

  // For large arrays, show first 5 elements + "... and N more"
  const firstFew = Array.from(parameters.slice(0, 5))
    .map(value => formatNumberWithPrecision(value, precision))
    .join(', ');
  const remaining = length - 5;
  return `[${firstFew}, ... and ${remaining} more]`;
}

/**
 * Formats a state array for display (used in constrained optimization).
 */
function formatStates(
  states: Float64Array,
  options: Required<ResultFormatterOptions>
): string {
  return formatParameters(states, options);
}

/**
 * Formats basic optimization result information.
 */
function formatBasicResult(
  result: OptimizationResult,
  options: Required<ResultFormatterOptions>
): string[] {
  const lines: string[] = [];

  if (options.showSectionHeaders) {
    lines.push('\n=== Optimization Results ===');
  }

  // Parameters
  lines.push('Optimized parameters:');
  const paramStr = formatParameters(result.finalParameters, options);
  lines.push(`  ${paramStr}`);

  // Cost
  lines.push('\nCost:');
  lines.push(`  f(p) = ${formatNumberWithPrecision(result.finalCost, options.costPrecision)}`);

  // Convergence information
  lines.push('\nConvergence:');
  lines.push(`  Converged: ${result.converged}`);
  lines.push(`  Iterations: ${result.iterations}`);

  if (result.finalGradientNorm !== undefined) {
    lines.push(`  Final gradient norm: ${formatNumberWithPrecision(result.finalGradientNorm, options.costPrecision)}`);
  }

  if (result.finalResidualNorm !== undefined) {
    lines.push(`  Final residual norm: ${formatNumberWithPrecision(result.finalResidualNorm, options.costPrecision)}`);
  }

  // Execution time
  if (options.showExecutionTime && options.elapsedTimeMs > 0) {
    lines.push(`\nExecution time: ${formatNumberWithPrecision(options.elapsedTimeMs, 2)} ms`);
    if (result.iterations > 0) {
      const timePerIteration = options.elapsedTimeMs / result.iterations;
      lines.push(`Time per iteration: ${formatNumberWithPrecision(timePerIteration, 3)} ms`);
    }
  }

  return lines;
}

/**
 * Formats a basic OptimizationResult.
 */
export function formatOptimizationResult(
  result: OptimizationResult,
  options?: ResultFormatterOptions
): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const lines = formatBasicResult(result, opts);
  return lines.join('\n');
}

/**
 * Formats a GradientDescentResult.
 */
export function formatGradientDescentResult(
  result: GradientDescentResult,
  options?: ResultFormatterOptions
): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const lines = formatBasicResult(result, opts);

  // Add line search information
  const lineSearchIndex = lines.findIndex(line => line.includes('Final gradient norm') || line.includes('Final residual norm'));
  if (lineSearchIndex >= 0) {
    lines.splice(lineSearchIndex + 1, 0, `  Used line search: ${result.usedLineSearch}`);
  } else {
    lines.push(`  Used line search: ${result.usedLineSearch}`);
  }

  return lines.join('\n');
}

/**
 * Formats a LevenbergMarquardtResult.
 */
export function formatLevenbergMarquardtResult(
  result: LevenbergMarquardtResult,
  options?: ResultFormatterOptions
): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const lines = formatBasicResult(result, opts);

  // Add lambda information
  const lambdaIndex = lines.findIndex(line => line.includes('Final residual norm'));
  if (lambdaIndex >= 0) {
    lines.splice(lambdaIndex + 1, 0, `  Final lambda: ${formatNumberWithPrecision(result.finalLambda, 6)}`);
  } else {
    lines.push(`  Final lambda: ${formatNumberWithPrecision(result.finalLambda, 6)}`);
  }

  return lines.join('\n');
}

/**
 * Formats a CMA-ES result.
 */
export function formatCmaEsResult(
  result: CmaEsResult,
  options?: ResultFormatterOptions
): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const lines = formatBasicResult(result, opts);

  const insertionIndex = lines.findIndex(line => line.includes('Final gradient norm') || line.includes('Final residual norm'));
  const extraLines = [
    `  Population size (λ): ${result.populationSize}`,
    `  Function evaluations: ${result.functionEvaluations}`,
    `  Final step size (σ): ${formatNumberWithPrecision(result.finalStepSize, 6)}`,
    `  Final max std dev: ${formatNumberWithPrecision(result.finalMaxStdDev, opts.costPrecision)}`
  ];
  if (result.stopReason) {
    extraLines.push(`  Stop reason: ${result.stopReason}`);
  }
  if (result.profiling) {
    extraLines.push(
      `  Profiling (ms): total=${formatNumberWithPrecision(result.profiling.totalMs, 2)}, ` +
        `cost=${formatNumberWithPrecision(result.profiling.costMs, 2)}, ` +
        `cholesky=${formatNumberWithPrecision(result.profiling.choleskyMs, 2)}, ` +
        `sampling=${formatNumberWithPrecision(result.profiling.samplingMs, 2)}, ` +
        `update=${formatNumberWithPrecision(result.profiling.updateMs, 2)}`
    );
  }

  if (insertionIndex >= 0) {
    lines.splice(insertionIndex + 1, 0, ...extraLines);
  } else {
    lines.push(...extraLines);
  }

  return lines.join('\n');
}

/**
 * Formats a ConstrainedGaussNewtonResult.
 */
export function formatConstrainedGaussNewtonResult(
  result: ConstrainedGaussNewtonResult,
  options?: ResultFormatterOptions
): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const lines: string[] = [];

  if (opts.showSectionHeaders) {
    lines.push('\n=== Optimization Results ===');
  }

  // Parameters
  lines.push('Optimized parameters:');
  const paramStr = formatParameters(result.finalParameters, opts);
  lines.push(`  ${paramStr}`);

  // States
  lines.push('Optimized states:');
  const stateStr = formatStates(result.finalStates, opts);
  lines.push(`  ${stateStr}`);

  // Cost
  lines.push('\nCost:');
  lines.push(`  f(p, x) = ${formatNumberWithPrecision(result.finalCost, opts.costPrecision)}`);

  // Constraint satisfaction
  lines.push('\nConstraint satisfaction:');
  if (result.finalConstraintNorm !== undefined) {
    lines.push(`  ||c(p, x)|| = ${formatNumberWithPrecision(result.finalConstraintNorm, opts.constraintPrecision)}`);
  } else {
    lines.push(`  ||c(p, x)|| = N/A`);
  }

  // Convergence information
  lines.push('\nConvergence:');
  lines.push(`  Converged: ${result.converged}`);
  lines.push(`  Iterations: ${result.iterations}`);

  if (result.finalGradientNorm !== undefined) {
    lines.push(`  Final gradient norm: ${formatNumberWithPrecision(result.finalGradientNorm, opts.costPrecision)}`);
  }

  if (result.finalResidualNorm !== undefined) {
    lines.push(`  Final residual norm: ${formatNumberWithPrecision(result.finalResidualNorm, opts.costPrecision)}`);
  }

  // Execution time
  if (opts.showExecutionTime && opts.elapsedTimeMs > 0) {
    lines.push(`\nExecution time: ${formatNumberWithPrecision(opts.elapsedTimeMs, 2)} ms`);
    if (result.iterations > 0) {
      const timePerIteration = opts.elapsedTimeMs / result.iterations;
      lines.push(`Time per iteration: ${formatNumberWithPrecision(timePerIteration, 3)} ms`);
    }
  }

  return lines.join('\n');
}

/**
 * Formats a ConstrainedLevenbergMarquardtResult.
 */
export function formatConstrainedLevenbergMarquardtResult(
  result: ConstrainedLevenbergMarquardtResult,
  options?: ResultFormatterOptions
): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const lines: string[] = [];

  if (opts.showSectionHeaders) {
    lines.push('\n=== Optimization Results ===');
  }

  // Parameters
  lines.push('Optimized parameters:');
  const paramStr = formatParameters(result.finalParameters, opts);
  lines.push(`  ${paramStr}`);

  // States
  lines.push('Optimized states:');
  const stateStr = formatStates(result.finalStates, opts);
  lines.push(`  ${stateStr}`);

  // Cost
  lines.push('\nCost:');
  lines.push(`  f(p, x) = ${formatNumberWithPrecision(result.finalCost, opts.costPrecision)}`);

  // Constraint satisfaction
  lines.push('\nConstraint satisfaction:');
  if (result.finalConstraintNorm !== undefined) {
    lines.push(`  ||c(p, x)|| = ${formatNumberWithPrecision(result.finalConstraintNorm, opts.constraintPrecision)}`);
  } else {
    lines.push(`  ||c(p, x)|| = N/A`);
  }

  // Convergence information
  lines.push('\nConvergence:');
  lines.push(`  Converged: ${result.converged}`);
  lines.push(`  Iterations: ${result.iterations}`);

  if (result.finalGradientNorm !== undefined) {
    lines.push(`  Final gradient norm: ${formatNumberWithPrecision(result.finalGradientNorm, opts.costPrecision)}`);
  }

  if (result.finalResidualNorm !== undefined) {
    lines.push(`  Final residual norm: ${formatNumberWithPrecision(result.finalResidualNorm, opts.costPrecision)}`);
  }

  lines.push(`  Final lambda: ${formatNumberWithPrecision(result.finalLambda, 6)}`);

  // Execution time
  if (opts.showExecutionTime && opts.elapsedTimeMs > 0) {
    lines.push(`\nExecution time: ${formatNumberWithPrecision(opts.elapsedTimeMs, 2)} ms`);
    if (result.iterations > 0) {
      const timePerIteration = opts.elapsedTimeMs / result.iterations;
      lines.push(`Time per iteration: ${formatNumberWithPrecision(timePerIteration, 3)} ms`);
    }
  }

  return lines.join('\n');
}

/**
 * Formats an AdjointGradientDescentResult.
 */
export function formatAdjointGradientDescentResult(
  result: AdjointGradientDescentResult,
  options?: ResultFormatterOptions
): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const lines: string[] = [];

  if (opts.showSectionHeaders) {
    lines.push('\n=== Optimization Results ===');
  }

  // Parameters
  lines.push('Optimized parameters:');
  const paramStr = formatParameters(result.finalParameters, opts);
  lines.push(`  ${paramStr}`);

  // States
  lines.push('Optimized states:');
  const stateStr = formatStates(result.finalStates, opts);
  lines.push(`  ${stateStr}`);

  // Cost
  lines.push('\nCost:');
  lines.push(`  f(p, x) = ${formatNumberWithPrecision(result.finalCost, opts.costPrecision)}`);

  // Constraint satisfaction
  lines.push('\nConstraint satisfaction:');
  if (result.finalConstraintNorm !== undefined) {
    lines.push(`  ||c(p, x)|| = ${formatNumberWithPrecision(result.finalConstraintNorm, opts.constraintPrecision)}`);
  } else {
    lines.push(`  ||c(p, x)|| = N/A`);
  }

  // Convergence information
  lines.push('\nConvergence:');
  lines.push(`  Converged: ${result.converged}`);
  lines.push(`  Iterations: ${result.iterations}`);

  if (result.finalGradientNorm !== undefined) {
    lines.push(`  Final gradient norm: ${formatNumberWithPrecision(result.finalGradientNorm, opts.costPrecision)}`);
  }

  lines.push(`  Used line search: ${result.usedLineSearch}`);

  // Execution time
  if (opts.showExecutionTime && opts.elapsedTimeMs > 0) {
    lines.push(`\nExecution time: ${formatNumberWithPrecision(opts.elapsedTimeMs, 2)} ms`);
    if (result.iterations > 0) {
      const timePerIteration = opts.elapsedTimeMs / result.iterations;
      lines.push(`Time per iteration: ${formatNumberWithPrecision(timePerIteration, 3)} ms`);
    }
  }

  return lines.join('\n');
}

/**
 * Type-safe overloaded function for formatting any optimization result.
 */
export function formatResult(
  result: OptimizationResult,
  options?: ResultFormatterOptions
): string;
export function formatResult(
  result: GradientDescentResult,
  options?: ResultFormatterOptions
): string;
export function formatResult(
  result: LevenbergMarquardtResult,
  options?: ResultFormatterOptions
): string;
export function formatResult(
  result: CmaEsResult,
  options?: ResultFormatterOptions
): string;
export function formatResult(
  result: ConstrainedGaussNewtonResult,
  options?: ResultFormatterOptions
): string;
export function formatResult(
  result: ConstrainedLevenbergMarquardtResult,
  options?: ResultFormatterOptions
): string;
export function formatResult(
  result: AdjointGradientDescentResult,
  options?: ResultFormatterOptions
): string;
export function formatResult(
  result:
    | OptimizationResult
    | GradientDescentResult
    | LevenbergMarquardtResult
    | CmaEsResult
    | ConstrainedGaussNewtonResult
    | ConstrainedLevenbergMarquardtResult
    | AdjointGradientDescentResult,
  options?: ResultFormatterOptions
): string {
  // Type guards to determine which formatter to use
  if ('finalStates' in result && 'finalLambda' in result) {
    return formatConstrainedLevenbergMarquardtResult(result as ConstrainedLevenbergMarquardtResult, options);
  }
  if ('finalStates' in result && 'finalConstraintNorm' in result) {
    if ('usedLineSearch' in result) {
      return formatAdjointGradientDescentResult(result as AdjointGradientDescentResult, options);
    }
    return formatConstrainedGaussNewtonResult(result as ConstrainedGaussNewtonResult, options);
  }
  if ('finalLambda' in result) {
    return formatLevenbergMarquardtResult(result as LevenbergMarquardtResult, options);
  }
  if ('populationSize' in result && 'functionEvaluations' in result && 'finalStepSize' in result) {
    return formatCmaEsResult(result as CmaEsResult, options);
  }
  if ('usedLineSearch' in result) {
    return formatGradientDescentResult(result as GradientDescentResult, options);
  }
  return formatOptimizationResult(result, options);
}

/**
 * Prints an optimization result directly to console.
 */
export function printOptimizationResult(
  result: OptimizationResult,
  options?: ResultFormatterOptions
): void {
  console.log(formatOptimizationResult(result, options));
}

/**
 * Prints a gradient descent result directly to console.
 */
export function printGradientDescentResult(
  result: GradientDescentResult,
  options?: ResultFormatterOptions
): void {
  console.log(formatGradientDescentResult(result, options));
}

/**
 * Prints a Levenberg-Marquardt result directly to console.
 */
export function printLevenbergMarquardtResult(
  result: LevenbergMarquardtResult,
  options?: ResultFormatterOptions
): void {
  console.log(formatLevenbergMarquardtResult(result, options));
}

/**
 * Prints a CMA-ES result directly to console.
 */
export function printCmaEsResult(
  result: CmaEsResult,
  options?: ResultFormatterOptions
): void {
  console.log(formatCmaEsResult(result, options));
}

/**
 * Prints a constrained Gauss-Newton result directly to console.
 */
export function printConstrainedGaussNewtonResult(
  result: ConstrainedGaussNewtonResult,
  options?: ResultFormatterOptions
): void {
  console.log(formatConstrainedGaussNewtonResult(result, options));
}

/**
 * Prints a constrained Levenberg-Marquardt result directly to console.
 */
export function printConstrainedLevenbergMarquardtResult(
  result: ConstrainedLevenbergMarquardtResult,
  options?: ResultFormatterOptions
): void {
  console.log(formatConstrainedLevenbergMarquardtResult(result, options));
}

/**
 * Prints an adjoint gradient descent result directly to console.
 */
export function printAdjointGradientDescentResult(
  result: AdjointGradientDescentResult,
  options?: ResultFormatterOptions
): void {
  console.log(formatAdjointGradientDescentResult(result, options));
}

/**
 * Type-safe overloaded function for printing any optimization result.
 */
export function printResult(
  result: OptimizationResult,
  options?: ResultFormatterOptions
): void;
export function printResult(
  result: GradientDescentResult,
  options?: ResultFormatterOptions
): void;
export function printResult(
  result: LevenbergMarquardtResult,
  options?: ResultFormatterOptions
): void;
export function printResult(
  result: CmaEsResult,
  options?: ResultFormatterOptions
): void;
export function printResult(
  result: ConstrainedGaussNewtonResult,
  options?: ResultFormatterOptions
): void;
export function printResult(
  result: ConstrainedLevenbergMarquardtResult,
  options?: ResultFormatterOptions
): void;
export function printResult(
  result: AdjointGradientDescentResult,
  options?: ResultFormatterOptions
): void;
export function printResult(
  result:
    | OptimizationResult
    | GradientDescentResult
    | LevenbergMarquardtResult
    | CmaEsResult
    | ConstrainedGaussNewtonResult
    | ConstrainedLevenbergMarquardtResult
    | AdjointGradientDescentResult,
  options?: ResultFormatterOptions
): void {
  console.log(formatResult(result, options));
}

