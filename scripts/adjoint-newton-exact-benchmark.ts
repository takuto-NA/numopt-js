/**
 * Full versus approximate Newton-adjoint and BFGS at a common independent stationarity target.
 * Includes derivative modes, dimension scaling, convergence traces, and warmed cost/time summaries.
 */

import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { adjointBfgs, adjointNewton, adjointNewtonExact } from '../src/index';
import type { AdjointNewtonOptions, AdjointNewtonExactOptions, AdjointNewtonResult, AdjointGradientDescentResult } from '../src/core/types';
import { createNewtonAdjointProblems, createChainProblem, createDesignScalingProblem,
  createQuadraticConvergenceProblem, type NewtonAdjointProblem } from '../examples/adjoint-newton-problems';
import { collectPaperEnvironment, computeParameterError, runWarmedRepeats, summarizeNumeric } from './benchmark-harness';
import { vectorNorm } from '../src/utils/matrix';

const REPEAT_COUNT = 7;
const WARMUP_COUNT = 1;
const MAX_ITERATIONS = 300;
const SOLVER_TOLERANCE = 1e-9;
const CONSTRAINT_TOLERANCE = 1e-12;
const COMMON_GRADIENT_TARGET = 1e-6;
const COMMON_PARAMETER_TARGET = 1e-6;
const OUTPUT_DIRECTORY = 'benchmark-results';
const OUTPUT_BASENAME = 'adjoint-newton-exact';

type Method = 'BFGS' | 'Approximate Newton' | 'Full Newton' | 'Full Newton HVP';
type Mode = 'analytical' | 'numerical';
type Counts = { objective: number; constraint: number; firstPartials: number; secondPartials: number };
type Checkpoint = { iteration: number; elapsedMs: number; counts: Counts; parameterError: number; trueGradientNorm: number };
type Trial = {
  target?: Checkpoint; trace: Checkpoint[]; totalMs: number; counts: Counts; iterations: number;
  converged: boolean; success: boolean; constraintNorm: number; reportedGradientNorm?: number;
  trueGradientNorm: number; dampedSteps?: number; stopReason?: string; error?: string;
  finalParameterError?: number;
};
type Row = { problem: string; parameters: number; states: number; method: Method; mode: Mode; trials: Trial[] };

function isNewtonResult(result: AdjointGradientDescentResult): result is AdjointNewtonResult {
  return 'hessianEvaluations' in result;
}

function analyticOptions(problem: NewtonAdjointProblem, method: Method, counts: Counts): AdjointNewtonOptions & AdjointNewtonExactOptions {
  const source = problem.derivatives;
  const options: AdjointNewtonOptions & AdjointNewtonExactOptions = {
    dfdp: (parameters, states) => { counts.firstPartials++; return source.dfdp!(parameters, states); },
    dfdx: (parameters, states) => { counts.firstPartials++; return source.dfdx!(parameters, states); },
    dcdp: (parameters, states) => { counts.firstPartials++; return source.dcdp!(parameters, states); },
    dcdx: (parameters, states) => { counts.firstPartials++; return source.dcdx!(parameters, states); }
  };
  if (method === 'Approximate Newton') {
    options.d2fdp2 = (parameters, states) => { counts.secondPartials++; return source.d2fdp2!(parameters, states); };
    options.d2fdx2 = (parameters, states) => { counts.secondPartials++; return source.d2fdx2!(parameters, states); };
  } else if (method === 'Full Newton') {
    options.lagrangianHessian = (parameters, states, lambda) => {
      counts.secondPartials++; return problem.exactDerivatives.lagrangianHessian!(parameters, states, lambda);
    };
  } else if (method === 'Full Newton HVP') {
    options.lagrangianHessianVector = (parameters, states, lambda, direction) => {
      counts.secondPartials++; return problem.exactDerivatives.lagrangianHessianVector!(parameters, states, lambda, direction);
    };
  }
  return options;
}

function runTrial(problem: NewtonAdjointProblem, method: Method, mode: Mode): Trial {
  const counts: Counts = { objective: 0, constraint: 0, firstPartials: 0, secondPartials: 0 };
  const trace: Checkpoint[] = [];
  let target: Checkpoint | undefined;
  const cost = (parameters: Float64Array, states: Float64Array): number => {
    counts.objective++; return problem.cost(parameters, states);
  };
  const constraint = (parameters: Float64Array, states: Float64Array): Float64Array => {
    counts.constraint++; return problem.constraint(parameters, states);
  };
  const start = performance.now();
  const options = {
    ...(mode === 'analytical' ? analyticOptions(problem, method, counts) : {}),
    maxIterations: MAX_ITERATIONS, tolerance: SOLVER_TOLERANCE, constraintTolerance: CONSTRAINT_TOLERANCE,
    onIteration: (iteration: number, _cost: number, parameters: Float64Array): void => {
      const point = { iteration, elapsedMs: performance.now() - start, counts: { ...counts },
        parameterError: computeParameterError(parameters, problem.expected), trueGradientNorm: vectorNorm(problem.reducedGradient(parameters)) };
      trace.push(point);
      if (!target && point.parameterError <= COMMON_PARAMETER_TARGET && point.trueGradientNorm <= COMMON_GRADIENT_TARGET) target = point;
    }
  };
  try {
    const result: AdjointNewtonResult | AdjointGradientDescentResult = method === 'BFGS'
      ? adjointBfgs(problem.parameters, problem.states, cost, constraint, options)
      : method === 'Approximate Newton' ? adjointNewton(problem.parameters, problem.states, cost, constraint, options)
      : adjointNewtonExact(problem.parameters, problem.states, cost, constraint, options);
    const totalMs = performance.now() - start;
    const constraintNorm = vectorNorm(problem.constraint(result.finalParameters, result.finalStates));
    const trueGradientNorm = vectorNorm(problem.reducedGradient(result.finalParameters));
    const finalParameterError = computeParameterError(result.finalParameters, problem.expected);
    if (!target && trueGradientNorm <= COMMON_GRADIENT_TARGET && finalParameterError <= COMMON_PARAMETER_TARGET) {
      target = { iteration: result.iterations, elapsedMs: totalMs, counts: { ...counts },
        parameterError: finalParameterError, trueGradientNorm };
    }
    return { target, trace, totalMs, counts, iterations: result.iterations, converged: result.converged,
      success: target !== undefined && constraintNorm <= CONSTRAINT_TOLERANCE && Number.isFinite(result.finalCost) &&
        trueGradientNorm <= COMMON_GRADIENT_TARGET && finalParameterError <= COMMON_PARAMETER_TARGET,
      constraintNorm, reportedGradientNorm: result.finalGradientNorm,
      trueGradientNorm, finalParameterError,
      ...(isNewtonResult(result) ? { dampedSteps: result.dampedSteps, stopReason: result.stopReason } : {}) };
  } catch (error) {
    return { target, trace, totalMs: performance.now() - start, counts, iterations: trace.length,
      converged: false, success: false, constraintNorm: Infinity, trueGradientNorm: Infinity, error: String(error) };
  }
}

const problems = [...createNewtonAdjointProblems(), createQuadraticConvergenceProblem(),
  createChainProblem(80), createDesignScalingProblem(4), createDesignScalingProblem(16)];
const rows: Row[] = [];
for (const problem of problems) {
  for (const mode of ['analytical', 'numerical'] as const) {
    const methods: Method[] = ['BFGS', 'Approximate Newton', 'Full Newton'];
    if (mode === 'analytical' && problem.exactDerivatives.lagrangianHessianVector) methods.push('Full Newton HVP');
    for (const method of methods) {
      const trials = runWarmedRepeats({ warmupCount: WARMUP_COUNT, repeatCount: REPEAT_COUNT,
        run: () => runTrial(problem, method, mode) });
      rows.push({ problem: problem.name, parameters: problem.parameters.length, states: problem.states.length, method, mode, trials });
      const first = trials[0];
      console.log(`${problem.name} / ${mode} / ${method}: target=${first.target?.iteration ?? 'missed'}, ` +
        `total=${first.iterations}, success=${trials.every(trial => trial.success)}${first.error ? ` ${first.error}` : ''}`);
    }
  }
}

const environment = collectPaperEnvironment();
const lines = ['# Full Newton–adjoint comparison', '',
  `Environment: ${environment.nodeVersion}, ${environment.platform}/${environment.architecture}, ${environment.cpuModel}.`,
  `Commit: ${environment.gitCommit} plus current implementation. Warmup ${WARMUP_COUNT}; retained repeats ${REPEAT_COUNT}.`,
  `Common target: independently evaluated true reduced gradient ≤ ${COMMON_GRADIENT_TARGET}, parameter error ≤ ${COMMON_PARAMETER_TARGET}.`,
  `All final states must satisfy ||c|| ≤ ${CONSTRAINT_TOLERANCE}; solvers use tolerance ${SOLVER_TOLERANCE}.`,
  'Primary iterations, counts and time stop at the first callback meeting this common target. Final solver iterations/flags are retained separately.',
  'Time includes solver work and identical monitoring; counters exclude the independent closed-form reference gradient.',
  'Analytical mode supplies first partials to all solvers and model-specific exact second derivatives to Newton methods.',
  'Numerical mode supplies no derivatives; full Newton samples the fixed-adjoint Lagrangian along tangent directions.',
  'All methods use the same corrected state/adjoint linear solves (unshifted first when regularization=0). The old approximate branch is preserved.',
  'Timing is median [IQR] ms and is secondary to accuracy/evaluation counts; these are small dense problems.', '',
  '| Problem (p,x) | Mode | Method | Success | Target iters | Objective / constraint calls | First / second calls | Target ms [IQR] | Final iters / converged | Damped steps |',
  '|---|---|---|---:|---:|---:|---:|---:|---|---:|'];
for (const row of rows) {
  const valid = row.trials.filter(trial => trial.target);
  const median = (getValue: (point: Checkpoint) => number): number => summarizeNumeric(valid.map(trial => getValue(trial.target!))).median;
  const timing = valid.length ? summarizeNumeric(valid.map(trial => trial.target!.elapsedMs)) : undefined;
  lines.push(`| ${row.problem} (${row.parameters},${row.states}) | ${row.mode} | ${row.method} | ` +
    `${row.trials.filter(trial => trial.success).length}/${REPEAT_COUNT} | ${valid.length ? median(point => point.iteration) : 'missed'} | ` +
    `${valid.length ? `${median(point => point.counts.objective)} / ${median(point => point.counts.constraint)}` : 'n/a'} | ` +
    `${valid.length ? `${median(point => point.counts.firstPartials)} / ${median(point => point.counts.secondPartials)}` : 'n/a'} | ` +
    `${timing ? `${timing.median.toFixed(3)} [${timing.interquartileRange.toFixed(3)}]` : 'n/a'} | ` +
    `${row.trials[0].iterations} / ${row.trials[0].converged} | ${row.trials[0].dampedSteps ?? 'n/a'} |`);
}
lines.push('', '## Local convergence trace (analytical, nonlinear problem)', '',
  '| Method | Iteration | Parameter error | True gradient norm | e_next / e_prev² |', '|---|---:|---:|---:|---:|');
for (const row of rows.filter(row => row.problem === createQuadraticConvergenceProblem().name && row.mode === 'analytical')) {
  row.trials[0].trace.forEach((point, index, trace) => {
    const quotient = index > 0 && trace[index - 1].parameterError > 0 ?
      point.parameterError / trace[index - 1].parameterError ** 2 : undefined;
    lines.push(`| ${row.method} | ${point.iteration} | ${point.parameterError.toExponential(5)} | ` +
      `${point.trueGradientNorm.toExponential(5)} | ${quotient?.toExponential(4) ?? 'n/a'} |`);
  });
}
const outputDirectory = path.resolve(OUTPUT_DIRECTORY);
fs.mkdirSync(outputDirectory, { recursive: true });
fs.writeFileSync(path.join(outputDirectory, `${OUTPUT_BASENAME}.md`), `${lines.join('\n')}\n`);
fs.writeFileSync(path.join(outputDirectory, `${OUTPUT_BASENAME}.json`), JSON.stringify({ environment,
  protocol: { warmupCount: WARMUP_COUNT, repeatCount: REPEAT_COUNT, solverTolerance: SOLVER_TOLERANCE,
    constraintTolerance: CONSTRAINT_TOLERANCE, commonGradientTarget: COMMON_GRADIENT_TARGET, commonParameterTarget: COMMON_PARAMETER_TARGET }, rows },
  (_key, value) => typeof value === 'number' && !Number.isFinite(value) ? String(value) : value, 2));
console.log(`Reports: ${outputDirectory}`);
if (rows.some(row => row.trials.some(trial => !trial.success))) process.exitCode = 1;
