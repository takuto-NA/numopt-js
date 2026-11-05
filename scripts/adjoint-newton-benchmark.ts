/**
 * Reproducible Newton-adjoint comparison: successes and omitted-curvature counterexamples.
 * Reports warmed median/IQR timing and evaluation counts; does not assume Newton is faster.
 */

import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { adjointBfgs, adjointGradientDescent, adjointNewton,
  constrainedGaussNewton, constrainedLevenbergMarquardt } from '../src/index';
import type { AdjointNewtonOptions, AdjointNewtonResult, AdjointGradientDescentResult,
  ConstrainedGaussNewtonResult, ConstrainedLevenbergMarquardtResult } from '../src/core/types';
import { createNewtonAdjointProblems, type NewtonAdjointProblem } from '../examples/adjoint-newton-problems';
import { collectPaperEnvironment, computeParameterError, runWarmedRepeats, summarizeNumeric } from './benchmark-harness';
import { vectorNorm } from '../src/utils/matrix';

const WARMUP_COUNT = 1;
const REPEAT_COUNT = 7;
const MAX_ITERATIONS = 2000;
const GRADIENT_TOLERANCE = 1e-6;
const CONSTRAINT_TOLERANCE = 1e-6;
const PARAMETER_TOLERANCE = 1e-5;
const OUTPUT_DIRECTORY = 'benchmark-results';
const OUTPUT_BASENAME = 'adjoint-newton';

type Method = 'Adjoint GD' | 'Adjoint BFGS' | 'Adjoint Newton' | 'Constrained GN' | 'Constrained LM';
type Mode = 'analytical' | 'numerical';
type Result = AdjointGradientDescentResult | AdjointNewtonResult | ConstrainedGaussNewtonResult | ConstrainedLevenbergMarquardtResult;
type Counts = { objective: number; constraint: number; firstPartials: number; secondPartials: number };
type Trial = {
  elapsedMs: number; iterations: number; cost: number; parameterError: number; constraintNorm: number;
  gradientNorm?: number; converged: boolean; success: boolean; counts: Counts;
  stopReason?: string; hessianEvaluations?: number; dampedSteps?: number; error?: string;
};

function countedDerivatives(options: AdjointNewtonOptions, counts: Counts): AdjointNewtonOptions {
  return {
    ...options,
    dfdp: (parameters, states) => { counts.firstPartials++; return options.dfdp!(parameters, states); },
    dfdx: (parameters, states) => { counts.firstPartials++; return options.dfdx!(parameters, states); },
    dcdp: (parameters, states) => { counts.firstPartials++; return options.dcdp!(parameters, states); },
    dcdx: (parameters, states) => { counts.firstPartials++; return options.dcdx!(parameters, states); },
    d2fdp2: (parameters, states) => { counts.secondPartials++; return options.d2fdp2!(parameters, states); },
    d2fdx2: (parameters, states) => { counts.secondPartials++; return options.d2fdx2!(parameters, states); }
  };
}

function runTrial(problem: NewtonAdjointProblem, method: Method, mode: Mode): Trial {
  const counts: Counts = { objective: 0, constraint: 0, firstPartials: 0, secondPartials: 0 };
  const cost = (parameters: Float64Array, states: Float64Array): number => {
    counts.objective++; return problem.cost(parameters, states);
  };
  const constraint = (parameters: Float64Array, states: Float64Array): Float64Array => {
    counts.constraint++; return problem.constraint(parameters, states);
  };
  const residual = (parameters: Float64Array, states: Float64Array): Float64Array => {
    counts.objective++; return problem.residual!(parameters, states);
  };
  const options: AdjointNewtonOptions = {
    ...(mode === 'analytical' ? countedDerivatives(problem.derivatives, counts) : {}),
    maxIterations: MAX_ITERATIONS, tolerance: GRADIENT_TOLERANCE, constraintTolerance: CONSTRAINT_TOLERANCE
  };
  const start = performance.now();
  try {
    let result: Result;
    if (method === 'Adjoint Newton') result = adjointNewton(problem.parameters, problem.states, cost, constraint, options);
    else if (method === 'Adjoint BFGS') result = adjointBfgs(problem.parameters, problem.states, cost, constraint, options);
    else if (method === 'Adjoint GD') result = adjointGradientDescent(problem.parameters, problem.states, cost, constraint, options);
    else if (method === 'Constrained GN') result = constrainedGaussNewton(problem.parameters, problem.states, residual, constraint, options);
    else result = constrainedLevenbergMarquardt(problem.parameters, problem.states, residual, constraint, options);
    const elapsedMs = performance.now() - start;
    const constraintNorm = vectorNorm(problem.constraint(result.finalParameters, result.finalStates));
    const parameterError = computeParameterError(result.finalParameters, problem.expected);
    return {
      elapsedMs, counts, iterations: result.iterations, cost: result.finalCost, gradientNorm: result.finalGradientNorm,
      constraintNorm, parameterError, converged: result.converged,
      success: Number.isFinite(result.finalCost) && parameterError <= PARAMETER_TOLERANCE && constraintNorm <= CONSTRAINT_TOLERANCE,
      ...('stopReason' in result ? { stopReason: result.stopReason, hessianEvaluations: result.hessianEvaluations,
        dampedSteps: result.dampedSteps } : {})
    };
  } catch (error) {
    return { elapsedMs: performance.now() - start, counts, iterations: MAX_ITERATIONS, cost: Infinity,
      parameterError: Infinity, constraintNorm: Infinity, converged: false, success: false, error: String(error) };
  }
}

const rows: { problem: string; method: Method; mode: Mode; trials: Trial[] }[] = [];
for (const problem of createNewtonAdjointProblems()) {
  for (const mode of ['analytical', 'numerical'] as const) {
    const methods: Method[] = ['Adjoint GD', 'Adjoint BFGS', 'Adjoint Newton'];
    if (mode === 'numerical' && problem.residual) methods.push('Constrained GN', 'Constrained LM');
    for (const method of methods) {
      const trials = runWarmedRepeats({ warmupCount: WARMUP_COUNT, repeatCount: REPEAT_COUNT,
        run: () => runTrial(problem, method, mode) });
      rows.push({ problem: problem.name, method, mode, trials });
      console.log(`${problem.name} / ${mode} / ${method}: ${trials[0].iterations} iterations, success=${trials.every(trial => trial.success)}`);
    }
  }
}

const environment = collectPaperEnvironment();
const lines = [
  '# Newton-adjoint comparison', '',
  `Environment: ${environment.nodeVersion}, ${environment.platform}/${environment.architecture}, ${environment.cpuModel}.`,
  `Commit: ${environment.gitCommit} (includes current working-tree implementation).`,
  `Warmup: ${WARMUP_COUNT}; repeats: ${REPEAT_COUNT}; time = median [IQR] milliseconds.`,
  `Success = parameter error ≤ ${PARAMETER_TOLERANCE} and constraint norm ≤ ${CONSTRAINT_TOLERANCE}; independent of converged.`,
  'Analytical: all scalar solvers receive the same first partials; Newton additionally receives both partial Hessians.',
  'Numerical: all derivatives are finite differences. GN/LM use residuals and appear only in this mode.',
  'Objective counts include residual calls for GN/LM. Partial callback counts do not measure their internal work.',
  'The quadratic is anisotropic; chain has 1 parameter and 20 states. The last two problems expose omitted terms.', '',
  '| Problem | Derivatives | Method | Success | Iterations | Objective calls | Constraint calls | First / second partial calls | Gradient norm | Time ms [IQR] |',
  '|---|---|---|---:|---:|---:|---:|---:|---:|---:|'
];
for (const row of rows) {
  const timing = summarizeNumeric(row.trials.map(trial => trial.elapsedMs));
  const first = row.trials[0];
  const median = (getValue: (trial: Trial) => number): number => summarizeNumeric(row.trials.map(getValue)).median;
  lines.push(`| ${row.problem} | ${row.mode} | ${row.method} | ${row.trials.filter(trial => trial.success).length}/${REPEAT_COUNT} | ` +
    `${median(trial => trial.iterations)} | ${median(trial => trial.counts.objective)} | ${median(trial => trial.counts.constraint)} | ` +
    `${first.counts.firstPartials} / ${first.counts.secondPartials} | ${first.gradientNorm?.toExponential(2) ?? 'n/a'} | ` +
    `${timing.median.toFixed(3)} [${timing.interquartileRange.toFixed(3)}] |`);
}
const outputDirectory = path.resolve(OUTPUT_DIRECTORY);
fs.mkdirSync(outputDirectory, { recursive: true });
fs.writeFileSync(path.join(outputDirectory, `${OUTPUT_BASENAME}.md`), `${lines.join('\n')}\n`);
fs.writeFileSync(path.join(outputDirectory, `${OUTPUT_BASENAME}.json`), JSON.stringify({ environment,
  protocol: { warmupCount: WARMUP_COUNT, repeatCount: REPEAT_COUNT, gradientTolerance: GRADIENT_TOLERANCE,
    constraintTolerance: CONSTRAINT_TOLERANCE, parameterTolerance: PARAMETER_TOLERANCE, maxIterations: MAX_ITERATIONS }, rows }, null, 2));
console.log(`Reports: ${outputDirectory}`);
// Counterexamples are intentional. A Newton failure remains an actionable regression, not a hidden speedup.
if (rows.filter(row => row.method === 'Adjoint Newton').some(row => row.trials.some(trial => !trial.success))) process.exitCode = 1;
