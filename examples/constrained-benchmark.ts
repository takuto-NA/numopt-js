import { performance } from 'node:perf_hooks';
import {
  adjointGradientDescent,
  constrainedGaussNewton,
  constrainedLevenbergMarquardt
} from '../src/index';
import type { ConstrainedResidualFn, ConstraintFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

type Problem = {
  name: string;
  residual: ConstrainedResidualFn;
  constraint: ConstraintFn;
  buildInitial: () => { p: Float64Array; x: Float64Array };
  options: {
    adjoint: Record<string, unknown>;
    gaussNewton: Record<string, unknown>;
    levenbergMarquardt: Record<string, unknown>;
  };
};

const rosenbrockResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
  const a = 1.0 - p[0];
  const b = x[0] - p[0] * p[0];
  return new Float64Array([a, 10.0 * b]);
};

const circleConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([p[0] * p[0] + x[0] * x[0] - 2.0]);
};

const illConditionedResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
  const scaledP = p[0] / 1000.0;
  const scaledX = 1000.0 * x[0];
  return new Float64Array([scaledP, scaledX]);
};

const simpleConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
  return new Float64Array([p[0] + x[0] - 1.0]);
};

const highDimResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
  const n = p.length;
  const residual = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    const target = i + 1;
    residual[i] = p[i] - target;
    residual[n + i] = x[i] - target;
  }
  return residual;
};

const highDimConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
  const constraint = new Float64Array(p.length);
  for (let i = 0; i < p.length; i++) {
    const target = 2 * (i + 1);
    constraint[i] = p[i] + x[i] - target;
  }
  return constraint;
};

const problems: Problem[] = [
  {
    name: 'Rosenbrock valley with circle constraint',
    residual: rosenbrockResidual,
    constraint: circleConstraint,
    buildInitial: () => {
      const p = new Float64Array([0.5]);
      const x = new Float64Array([Math.sqrt(2 - p[0] * p[0])]);
      return { p, x };
    },
    options: {
      adjoint: { maxIterations: 500, tolerance: 1e-4, constraintTolerance: 1e-5, useLineSearch: true },
      gaussNewton: { maxIterations: 300, tolerance: 1e-5, constraintTolerance: 1e-5 },
      levenbergMarquardt: {
        maxIterations: 300,
        tolGradient: 1e-6,
        tolStep: 1e-8,
        constraintTolerance: 1e-5,
        lambdaInitial: 1e-2
      }
    }
  },
  {
    name: 'Ill-conditioned single variable',
    residual: illConditionedResidual,
    constraint: simpleConstraint,
    buildInitial: () => ({ p: new Float64Array([2000.0]), x: new Float64Array([-1999.0]) }),
    options: {
      adjoint: { maxIterations: 500, tolerance: 1e-3, constraintTolerance: 1e-5, useLineSearch: true },
      gaussNewton: { maxIterations: 300, tolerance: 1e-3, constraintTolerance: 1e-5 },
      levenbergMarquardt: {
        maxIterations: 300,
        tolGradient: 1e-3,
        tolStep: 1e-6,
        constraintTolerance: 1e-5,
        lambdaInitial: 5e-2
      }
    }
  },
  {
    name: 'High-dimensional 10D affine constraint',
    residual: highDimResidual,
    constraint: highDimConstraint,
    buildInitial: () => {
      const n = 10;
      const p = new Float64Array(n).fill(5.0);
      const x = new Float64Array(n).map((_, i) => 2 * (i + 1) - 5.0);
      return { p, x };
    },
    options: {
      adjoint: { maxIterations: 400, tolerance: 1e-3, constraintTolerance: 1e-4, useLineSearch: true },
      gaussNewton: { maxIterations: 250, tolerance: 1e-3, constraintTolerance: 1e-4 },
      levenbergMarquardt: {
        maxIterations: 250,
        tolGradient: 1e-3,
        tolStep: 1e-6,
        constraintTolerance: 1e-4,
        lambdaInitial: 1e-2
      }
    }
  }
];

type Solver = {
  name: string;
  run: (initial: { p: Float64Array; x: Float64Array }, problem: Problem) => {
    parameters: Float64Array;
    finalStates: Float64Array;
    iterations: number;
    converged: boolean;
    finalCost: number;
  };
};

const solvers: Solver[] = [
  {
    name: 'Adjoint GD',
    run: (initial, problem) =>
      adjointGradientDescent(initial.p, initial.x, problem.residual, problem.constraint, problem.options.adjoint)
  },
  {
    name: 'Constrained GN',
    run: (initial, problem) =>
      constrainedGaussNewton(initial.p, initial.x, problem.residual, problem.constraint, problem.options.gaussNewton)
  },
  {
    name: 'Constrained LM',
    run: (initial, problem) =>
      constrainedLevenbergMarquardt(
        initial.p,
        initial.x,
        problem.residual,
        problem.constraint,
        problem.options.levenbergMarquardt
      )
  }
];

function runSolver(solver: Solver, problem: Problem) {
  const initial = problem.buildInitial();
  const start = performance.now();
  try {
    const result = solver.run(initial, problem);
    const elapsedMs = performance.now() - start;
    const constraintNorm = vectorNorm(problem.constraint(result.parameters, result.finalStates));

    return {
      Method: solver.name,
      Iterations: result.iterations,
      TimeMs: Number(elapsedMs.toFixed(3)),
      Converged: result.converged,
      FinalCost: Number(result.finalCost.toExponential(3)),
      ConstraintNorm: Number(constraintNorm.toExponential(3)),
      Error: ''
    };
  } catch (error) {
    const elapsedMs = performance.now() - start;
    const message = error instanceof Error ? error.message : String(error);
    return {
      Method: solver.name,
      Iterations: 'error',
      TimeMs: Number(elapsedMs.toFixed(3)),
      Converged: false,
      FinalCost: 'error',
      ConstraintNorm: 'error',
      Error: message
    };
  }
}

console.log('Benchmark: constrained optimizers (lower time/iterations is better while keeping constraints satisfied)');
for (const problem of problems) {
  console.log(`\n=== ${problem.name} ===`);
  const rows = solvers.map((solver) => runSolver(solver, problem));
  console.table(rows);
}
