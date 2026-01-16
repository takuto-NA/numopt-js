import { performance } from 'node:perf_hooks';
import {
  adjointGradientDescent,
  constrainedGaussNewton,
  constrainedLevenbergMarquardt,
  levenbergMarquardt,
  gaussNewton
} from '../src/index';
import type { ConstrainedResidualFn, ConstraintFn, ResidualFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

type Problem = {
  name: string;
  residual: ConstrainedResidualFn;
  constraint: ConstraintFn;
  buildInitial: () => { p: Float64Array; x: Float64Array };
  penaltyWeight: number;
  options: {
    adjoint: Record<string, unknown>;
    gaussNewton: Record<string, unknown>;
    levenbergMarquardt: Record<string, unknown>;
    penaltyGaussNewton: Record<string, unknown>;
    penaltyLevenbergMarquardt: Record<string, unknown>;
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
      // Start far off the constraint manifold to require many feasibility corrections
      const p = new Float64Array([1.5]);
      const x = new Float64Array([-1.0]);
      return { p, x };
    },
    penaltyWeight: 1e4,
    options: {
      adjoint: { maxIterations: 100, tolerance: 1e-8, constraintTolerance: 1e-8, useLineSearch: true, logLevel: 'WARN' },
      gaussNewton: { maxIterations: 100, tolerance: 1e-8, constraintTolerance: 1e-8 },
      levenbergMarquardt: {
        maxIterations: 100,
        tolGradient: 1e-8,
        tolStep: 1e-10,
        constraintTolerance: 1e-8,
        lambdaInitial: 1e-3
      },
      penaltyGaussNewton: { maxIterations: 100, tolerance: 1e-10 },
      penaltyLevenbergMarquardt: { maxIterations: 100, tolGradient: 1e-10, tolStep: 1e-12, lambdaInitial: 1e-3 }
    }
  },
  {
    name: 'Ill-conditioned single variable',
    residual: illConditionedResidual,
    constraint: simpleConstraint,
    buildInitial: () => ({ p: new Float64Array([5000.0]), x: new Float64Array([-4999.0]) }),
    penaltyWeight: 1e8,
    options: {
      adjoint: { maxIterations: 100, tolerance: 1e-6, constraintTolerance: 1e-8, useLineSearch: true, logLevel: 'WARN' },
      gaussNewton: { maxIterations: 100, tolerance: 1e-6, constraintTolerance: 1e-8 },
      levenbergMarquardt: {
        maxIterations: 100,
        tolGradient: 1e-6,
        tolStep: 1e-8,
        constraintTolerance: 1e-8,
        lambdaInitial: 1e-2
      },
      penaltyGaussNewton: { maxIterations: 100, tolerance: 1e-10 },
      penaltyLevenbergMarquardt: { maxIterations: 100, tolGradient: 1e-10, tolStep: 1e-12, lambdaInitial: 1e-2 }
    }
  },
  {
    name: 'High-dimensional 10D affine constraint',
    residual: highDimResidual,
    constraint: highDimConstraint,
    buildInitial: () => {
      const n = 30;
      const p = new Float64Array(n).fill(10.0);
      const x = new Float64Array(n).map((_, i) => 2 * (i + 1) - 10.0);
      return { p, x };
    },
    penaltyWeight: 1e5,
    options: {
      adjoint: { maxIterations: 100, tolerance: 1e-4, constraintTolerance: 1e-8, useLineSearch: true, logLevel: 'WARN' },
      gaussNewton: { maxIterations: 100, tolerance: 1e-4, constraintTolerance: 1e-8 },
      levenbergMarquardt: {
        maxIterations: 100,
        tolGradient: 1e-4,
        tolStep: 1e-8,
        constraintTolerance: 1e-8,
        lambdaInitial: 1e-3
      },
      penaltyGaussNewton: { maxIterations: 100, tolerance: 1e-10 },
      penaltyLevenbergMarquardt: { maxIterations: 100, tolGradient: 1e-10, tolStep: 1e-12, lambdaInitial: 1e-3 }
    }
  },
  {
    name: 'State-heavy linear trajectory fit (p ≪ x)',
    residual: (p: Float64Array, x: Float64Array) => {
      const n = x.length;
      const r = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const target =
          Math.sin((2 * Math.PI * i) / n) +
          0.1 * Math.sin((6 * Math.PI * i) / n);
        r[i] = x[i] - target;
      }
      return r;
    },
    constraint: (p: Float64Array, x: Float64Array) => {
      const n = x.length;
      const c = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        c[i] = x[i] - (p[0] + p[1] * i + p[2] * Math.sin((2 * Math.PI * i) / n));
      }
      return c;
    },
    buildInitial: () => {
      const n = 200; // many states, few parameters
      return { p: new Float64Array([0.1, 0.01, 0.0]), x: new Float64Array(n).fill(0) };
    },
    penaltyWeight: 1e5,
    options: {
      adjoint: { maxIterations: 100, tolerance: 1e-5, constraintTolerance: 1e-8, useLineSearch: true, logLevel: 'WARN' },
      gaussNewton: { maxIterations: 100, tolerance: 1e-5, constraintTolerance: 1e-8 },
      levenbergMarquardt: {
        maxIterations: 100,
        tolGradient: 1e-6,
        tolStep: 1e-8,
        constraintTolerance: 1e-8,
        lambdaInitial: 1e-3
      },
      penaltyGaussNewton: { maxIterations: 100, tolerance: 1e-10 },
      penaltyLevenbergMarquardt: { maxIterations: 100, tolGradient: 1e-10, tolStep: 1e-12, lambdaInitial: 1e-3 }
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
  },
  {
    name: 'Penalty GN',
    run: (initial, problem) => {
      const theta0 = concatState(initial);
      const residual: ResidualFn = buildPenaltyResidual(problem, problem.penaltyWeight);
      const result = gaussNewton(theta0, residual, problem.options.penaltyGaussNewton);
      const { p, x } = splitState(result.finalParameters, initial.p.length);
      return {
        parameters: p,
        finalStates: x,
        iterations: result.iterations,
        converged: result.converged,
        finalCost: result.finalCost ?? 0
      };
    }
  },
  {
    name: 'Penalty LM',
    run: (initial, problem) => {
      const theta0 = concatState(initial);
      const residual: ResidualFn = buildPenaltyResidual(problem, problem.penaltyWeight);
      const result = levenbergMarquardt(theta0, residual, problem.options.penaltyLevenbergMarquardt);
      const { p, x } = splitState(result.finalParameters, initial.p.length);
      return {
        parameters: p,
        finalStates: x,
        iterations: result.iterations,
        converged: result.converged,
        finalCost: result.finalCost
      };
    }
  }
];

function concatState(initial: { p: Float64Array; x: Float64Array }): Float64Array {
  const theta = new Float64Array(initial.p.length + initial.x.length);
  theta.set(initial.p, 0);
  theta.set(initial.x, initial.p.length);
  return theta;
}

function splitState(theta: Float64Array, pLen: number): { p: Float64Array; x: Float64Array } {
  const p = theta.slice(0, pLen);
  const x = theta.slice(pLen);
  return { p, x };
}

function buildPenaltyResidual(problem: Problem, penaltyWeight: number): ResidualFn {
  const sqrtMu = Math.sqrt(penaltyWeight);
  return (theta: Float64Array): Float64Array => {
    const { p, x } = splitState(theta, problem.buildInitial().p.length);
    const baseR = problem.residual(p, x);
    const c = problem.constraint(p, x);
    const out = new Float64Array(baseR.length + c.length);
    out.set(baseR, 0);
    for (let i = 0; i < c.length; i++) {
      out[baseR.length + i] = sqrtMu * c[i];
    }
    return out;
  };
}

function runSolver(solver: Solver, problem: Problem) {
  const initial = problem.buildInitial();
  const start = performance.now();
  try {
    const result = solver.run(initial, problem);
    const elapsedMs = performance.now() - start;
    const constraintNorm = vectorNorm(problem.constraint(result.finalParameters, result.finalStates));

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
