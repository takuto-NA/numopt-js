import { performance } from 'node:perf_hooks';
import {
  constrainedLevenbergMarquardt,
  constrainedGaussNewton,
  levenbergMarquardt,
  gaussNewton,
  adjointGradientDescent
} from '../src/index';
import type { ConstrainedResidualFn, ConstraintFn, ResidualFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

// Problem setup:
// - Parametric curve from A to B with many states (sampled points) but few parameters (Fourier coefficients).
// - Objective: minimize discrete bending energy (second-difference norm).
// - Constraint: fixed segment length |x_{i+1} - x_i| = l to enforce arc-length.

type CurveProblem = {
  name: string;
  n: number; // number of interior points (not counting endpoints)
  segmentLength: number;
  parameters: number; // Fourier coefficient pairs (cos/sin)
  A: [number, number];
  B: [number, number];
  penaltyWeights: number[];
  options: {
    constrainedLM: Record<string, unknown>;
    constrainedGN: Record<string, unknown>;
    penaltyLM: Record<string, unknown>;
    penaltyGN: Record<string, unknown>;
    adjoint?: Record<string, unknown>;
  };
};

// Generate curve from Fourier coefficients
function generateCurve(p: Float64Array, n: number, A: [number, number], B: [number, number]): Float64Array[] {
  const pts: Float64Array[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    let x = A[0] + (B[0] - A[0]) * t;
    let y = A[1] + (B[1] - A[1]) * t;
    // p packed as [a1,b1,a2,b2,...] for cos/sin terms
    for (let k = 0; k < p.length / 2; k++) {
      const a = p[2 * k];
      const b = p[2 * k + 1];
      x += a * Math.cos((k + 1) * Math.PI * t) + b * Math.sin((k + 1) * Math.PI * t);
      y += a * Math.sin((k + 1) * Math.PI * t) - b * Math.cos((k + 1) * Math.PI * t); // phase-shifted to mix
    }
    pts.push(new Float64Array([x, y]));
  }
  return pts;
}

// Residual: discrete bending energy terms (second differences)
function bendingResidualFromStates(x: Float64Array, problem: CurveProblem): Float64Array {
  // x packed as [x0,y0,x1,y1,...]
  const residual = new Float64Array(problem.n - 1);
  for (let i = 1; i < problem.n; i++) {
    const xm1x = x[2 * (i - 1)];
    const xm1y = x[2 * (i - 1) + 1];
    const xix = x[2 * i];
    const xiy = x[2 * i + 1];
    const xp1x = x[2 * (i + 1)];
    const xp1y = x[2 * (i + 1) + 1];
    const ddx = xp1x - 2 * xix + xm1x;
    const ddy = xp1y - 2 * xiy + xm1y;
    residual[i - 1] = Math.hypot(ddx, ddy);
  }
  return residual;
}

// Constraint: |x_{i+1} - x_i| - l = 0 for all segments
function arcLengthConstraintFromStates(x: Float64Array, problem: CurveProblem): Float64Array {
  const c = new Float64Array(problem.n);
  for (let i = 0; i < problem.n; i++) {
    const dx = x[2 * (i + 1)] - x[2 * i];
    const dy = x[2 * (i + 1) + 1] - x[2 * i + 1];
    c[i] = Math.hypot(dx, dy) - problem.segmentLength;
  }
  return c;
}

// Penalty residual builder using states
function buildPenaltyResidual(problem: CurveProblem, penaltyWeight: number): ResidualFn {
  const sqrtMu = Math.sqrt(penaltyWeight);
  return (state: Float64Array): Float64Array => {
    const base = bendingResidualFromStates(state, problem);
    const c = arcLengthConstraintFromStates(state, problem);
    const out = new Float64Array(base.length + c.length);
    out.set(base, 0);
    for (let i = 0; i < c.length; i++) {
      out[base.length + i] = sqrtMu * c[i];
    }
    return out;
  };
}

const problems: CurveProblem[] = [
  {
    name: 'Curve bending (p<<x) moderate',
    n: 80, // 81 points (fast demo)
    segmentLength: 0.01,
    parameters: 4, // 2 cosine/sine pairs
    A: [0, 0],
    B: [2, 0.5],
    penaltyWeights: [1e3, 1e4],
    options: {
      constrainedLM: { maxIterations: 150, tolGradient: 1e-6, tolStep: 1e-8, constraintTolerance: 1e-4, lambdaInitial: 1e-2, lambdaFactor: 5 },
      constrainedGN: { maxIterations: 150, tolerance: 1e-6, constraintTolerance: 1e-4 },
      penaltyLM: { maxIterations: 150, tolGradient: 1e-6, tolStep: 1e-8, lambdaInitial: 1e-2 },
      penaltyGN: { maxIterations: 150, tolerance: 1e-6 },
      adjoint: { maxIterations: 150, tolerance: 1e-4, constraintTolerance: 1e-4, useLineSearch: true }
    }
  }
];

type Solver = {
  name: string;
  run: (initial: { p: Float64Array; x: Float64Array }, problem: CurveProblem, penaltyWeight?: number) => {
    parameters?: Float64Array;
    finalStates?: Float64Array;
    iterations: number | string;
    converged: boolean;
    finalCost: number | string;
    constraintNorm?: number;
  };
};

const solvers: Solver[] = [
  {
    name: 'Constrained LM',
    run: (initial, problem) => {
      const constrainedResidual: ConstrainedResidualFn = (_p: Float64Array, x: Float64Array) => bendingResidualFromStates(x, problem);
      const constraint: ConstraintFn = (_p: Float64Array, x: Float64Array) => arcLengthConstraintFromStates(x, problem);
      const result = constrainedLevenbergMarquardt(
        initial.p,
        initial.x,
        constrainedResidual,
        constraint,
        problem.options.constrainedLM
      );
      return {
        parameters: result.parameters,
        finalStates: result.finalStates,
        iterations: result.iterations,
        converged: result.converged,
        finalCost: result.finalCost,
        constraintNorm: vectorNorm(constraint(result.parameters, result.finalStates ?? initial.x))
      };
    }
  },
  {
    name: 'Constrained GN',
    run: (initial, problem) => {
      const constrainedResidual: ConstrainedResidualFn = (_p: Float64Array, x: Float64Array) => bendingResidualFromStates(x, problem);
      const constraint: ConstraintFn = (_p: Float64Array, x: Float64Array) => arcLengthConstraintFromStates(x, problem);
      const result = constrainedGaussNewton(initial.p, initial.x, constrainedResidual, constraint, problem.options.constrainedGN);
      return {
        parameters: result.parameters,
        finalStates: result.finalStates,
        iterations: result.iterations,
        converged: result.converged,
        finalCost: result.finalCost,
        constraintNorm: vectorNorm(constraint(result.parameters, result.finalStates ?? initial.x))
      };
    }
  },
  {
    name: 'Penalty LM',
    run: (initial, problem, penaltyWeight) => {
      const penaltyResidual = buildPenaltyResidual(problem, penaltyWeight ?? problem.penaltyWeights[0]);
      const result = levenbergMarquardt(initial.x, penaltyResidual, problem.options.penaltyLM);
      const constraintNorm = vectorNorm(arcLengthConstraintFromStates(result.parameters, problem));
      return {
        finalStates: result.parameters,
        iterations: result.iterations,
        converged: result.converged,
        finalCost: result.finalCost,
        constraintNorm
      };
    }
  },
  {
    name: 'Penalty GN',
    run: (initial, problem, penaltyWeight) => {
      const penaltyResidual = buildPenaltyResidual(problem, penaltyWeight ?? problem.penaltyWeights[0]);
      const result = gaussNewton(initial.x, penaltyResidual, problem.options.penaltyGN);
      const constraintNorm = vectorNorm(arcLengthConstraintFromStates(result.parameters, problem));
      return {
        finalStates: result.parameters,
        iterations: result.iterations,
        converged: result.converged,
        finalCost: result.finalCost ?? 0,
        constraintNorm
      };
    }
  },
  {
    name: 'Adjoint GD',
    run: (initial, problem) => {
      const constrainedCost: ConstrainedResidualFn = (_p: Float64Array, x: Float64Array) => bendingResidualFromStates(x, problem);
      const constraint: ConstraintFn = (_p: Float64Array, x: Float64Array) => arcLengthConstraintFromStates(x, problem);
      (globalThis as any).__ADJOINT_REGULARIZATION__ = 1e-3;
      const result = adjointGradientDescent(
        initial.p,
        initial.x,
        constrainedCost,
        constraint,
        problem.options.adjoint ?? {}
      );
      const constraintNorm = vectorNorm(constraint(result.parameters, result.finalStates ?? initial.x));
      return {
        parameters: result.parameters,
        finalStates: result.finalStates,
        iterations: result.iterations,
        converged: result.converged,
        finalCost: result.finalCost,
        constraintNorm
      };
    }
  }
];

// simple deterministic RNG
function createRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

function randomInitial(problem: CurveProblem, rng: () => number): { p: Float64Array; x: Float64Array } {
  const p = new Float64Array(problem.parameters);
  for (let i = 0; i < p.length; i++) {
    p[i] = 0.2 * (rng() - 0.5);
  }
  const base = generateCurve(p, problem.n, problem.A, problem.B);
  // Add noise to states to violate constraints
  const x = new Float64Array((problem.n + 1) * 2);
  for (let i = 0; i < base.length; i++) {
    x[2 * i] = base[i][0] + 0.01 * (rng() - 0.5);
    x[2 * i + 1] = base[i][1] + 0.01 * (rng() - 0.5);
  }
  return { p, x };
}

function runSolver(solver: Solver, problem: CurveProblem, rng: () => number, penaltyWeight?: number) {
  const initial = randomInitial(problem, rng);
  const start = performance.now();
  try {
    const result = solver.run(initial, problem, penaltyWeight);
    const elapsed = performance.now() - start;
    return {
      Method: solver.name,
      Iterations: result.iterations,
      TimeMs: Number(elapsed.toFixed(2)),
      Converged: result.converged,
      FinalCost: Number(result.finalCost.toExponential(3)),
      ConstraintNorm: result.constraintNorm !== undefined ? Number(result.constraintNorm.toExponential(3)) : undefined,
      Error: ''
    };
  } catch (error) {
    const elapsed = performance.now() - start;
    const message = error instanceof Error ? error.message : String(error);
    return {
      Method: solver.name,
      Iterations: 'error',
      TimeMs: Number(elapsed.toFixed(2)),
      Converged: false,
      FinalCost: 'error',
      ConstraintNorm: 'error',
      Error: message
    };
  }
}

console.log('Benchmark: curve bending with arc-length constraints (p << x)');
for (let idx = 0; idx < problems.length; idx++) {
  const problem = problems[idx];
  const rng = createRng(1234 + idx * 17);
  console.log(`\n=== ${problem.name} (n=${problem.n}, params=${problem.parameters}) ===`);
  const rows: any[] = [];
  for (const solver of solvers) {
    if (solver.name.startsWith('Penalty')) {
      for (const mu of problem.penaltyWeights) {
        const row = runSolver(solver, problem, rng, mu);
        row.Penalty = mu;
        rows.push(row);
      }
    } else {
      const row = runSolver(solver, problem, rng);
      row.Penalty = '-';
      rows.push(row);
    }
  }
  console.table(rows);
}
