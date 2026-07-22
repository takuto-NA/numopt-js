/**
 * Shared timing / table helpers for constrained optimizer benchmarks.
 * Suite scripts own problem definitions; this module owns orchestration primitives.
 */

import { performance } from 'node:perf_hooks';
import type { ResidualFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

export type BenchmarkRow = {
  Method: string;
  Iterations: number | string;
  TimeMs: number;
  Converged: boolean;
  FinalCost: number | string;
  ConstraintNorm: number | string;
  Error: string;
  Penalty?: number | string;
};

export type ConstrainedSolveResult = {
  finalParameters: Float64Array;
  finalStates: Float64Array;
  iterations: number;
  converged: boolean;
  finalCost: number;
};

export type BenchmarkSolver<Problem> = {
  name: string;
  run: (problem: Problem, initial: { parameters: Float64Array; states: Float64Array }) => ConstrainedSolveResult;
};

export function concatParameterAndState(parameters: Float64Array, states: Float64Array): Float64Array {
  const combined = new Float64Array(parameters.length + states.length);
  combined.set(parameters, 0);
  combined.set(states, parameters.length);
  return combined;
}

export function splitParameterAndState(
  combined: Float64Array,
  parameterCount: number
): { parameters: Float64Array; states: Float64Array } {
  return {
    parameters: combined.slice(0, parameterCount),
    states: combined.slice(parameterCount)
  };
}

function appendPenalizedConstraint(
  baseResidual: Float64Array,
  constraintValues: Float64Array,
  penaltyWeight: number
): Float64Array {
  const sqrtPenaltyWeight = Math.sqrt(penaltyWeight);
  const penalized = new Float64Array(baseResidual.length + constraintValues.length);
  penalized.set(baseResidual, 0);
  for (let index = 0; index < constraintValues.length; index++) {
    penalized[baseResidual.length + index] = sqrtPenaltyWeight * constraintValues[index];
  }
  return penalized;
}

export function buildPenaltyResidual(options: {
  parameterCount: number;
  residual: (parameters: Float64Array, states: Float64Array) => Float64Array;
  constraint: (parameters: Float64Array, states: Float64Array) => Float64Array;
  penaltyWeight: number;
}): ResidualFn {
  return (combined: Float64Array): Float64Array => {
    const { parameters, states } = splitParameterAndState(combined, options.parameterCount);
    return appendPenalizedConstraint(
      options.residual(parameters, states),
      options.constraint(parameters, states),
      options.penaltyWeight
    );
  };
}

export function buildStateOnlyPenaltyResidual(options: {
  residual: (states: Float64Array) => Float64Array;
  constraint: (states: Float64Array) => Float64Array;
  penaltyWeight: number;
}): ResidualFn {
  return (states: Float64Array): Float64Array => {
    return appendPenalizedConstraint(
      options.residual(states),
      options.constraint(states),
      options.penaltyWeight
    );
  };
}

export function constraintNormFor(
  constraint: (parameters: Float64Array, states: Float64Array) => Float64Array,
  result: ConstrainedSolveResult
): number {
  return vectorNorm(constraint(result.finalParameters, result.finalStates));
}

export function timeConstrainedSolve(options: {
  methodName: string;
  run: () => ConstrainedSolveResult;
  constraintNorm: (result: ConstrainedSolveResult) => number;
}): BenchmarkRow {
  const start = performance.now();
  try {
    const result = options.run();
    const elapsedMs = performance.now() - start;
    return {
      Method: options.methodName,
      Iterations: result.iterations,
      TimeMs: Number(elapsedMs.toFixed(3)),
      Converged: result.converged,
      FinalCost: Number(result.finalCost.toExponential(3)),
      ConstraintNorm: Number(options.constraintNorm(result).toExponential(3)),
      Error: ''
    };
  } catch (error) {
    const elapsedMs = performance.now() - start;
    const message = error instanceof Error ? error.message : String(error);
    return {
      Method: options.methodName,
      Iterations: 'error',
      TimeMs: Number(elapsedMs.toFixed(3)),
      Converged: false,
      FinalCost: 'error',
      ConstraintNorm: 'error',
      Error: message
    };
  }
}

export function runSolverTable<Problem extends {
  constraint: (parameters: Float64Array, states: Float64Array) => Float64Array;
}>(options: {
  problem: Problem;
  initial: { parameters: Float64Array; states: Float64Array };
  solvers: Array<BenchmarkSolver<Problem>>;
}): BenchmarkRow[] {
  return options.solvers.map((solver) =>
    timeConstrainedSolve({
      methodName: solver.name,
      run: () => solver.run(options.problem, options.initial),
      constraintNorm: (result) => constraintNormFor(options.problem.constraint, result)
    })
  );
}
