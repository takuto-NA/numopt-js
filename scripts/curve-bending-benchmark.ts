/**
 * Manual benchmark: curve bending energy with arc-length constraints (few parameters, many states).
 * Not run in CI. Invoke via `npm run benchmark:curve-bending`.
 */

import {
  constrainedLevenbergMarquardt,
  constrainedGaussNewton,
  levenbergMarquardt,
  gaussNewton,
  adjointGradientDescent
} from '../src/index';
import type {
  AdjointGradientDescentOptions,
  ConstrainedGaussNewtonOptions,
  ConstrainedLevenbergMarquardtOptions,
  ConstrainedResidualFn,
  ConstraintFn,
  GaussNewtonOptions,
  LevenbergMarquardtOptions
} from '../src/core/types';
import { createSeededRandom } from '../src/utils/random';
import { vectorNorm } from '../src/utils/matrix';
import {
  buildStateOnlyPenaltyResidual,
  timeConstrainedSolve,
  type BenchmarkRow,
  type BenchmarkSolver
} from './benchmark-harness';

const ADJOINT_REGULARIZATION = 1e-3;
const RANDOM_SEED = 1234;
const PARAMETER_NOISE_SCALE = 0.2;
const STATE_NOISE_SCALE = 0.01;

type CurveProblem = {
  name: string;
  interiorPointCount: number;
  segmentLength: number;
  parameterCount: number;
  startPoint: [number, number];
  endPoint: [number, number];
  penaltyWeights: number[];
  options: {
    constrainedLM: ConstrainedLevenbergMarquardtOptions;
    constrainedGN: ConstrainedGaussNewtonOptions;
    penaltyLM: LevenbergMarquardtOptions;
    penaltyGN: GaussNewtonOptions;
    adjoint: AdjointGradientDescentOptions;
  };
  // State-only constraint view used by penalty solvers and constraint-norm reporting.
  constraint: ConstraintFn;
};

function generateCurve(
  parameters: Float64Array,
  interiorPointCount: number,
  startPoint: [number, number],
  endPoint: [number, number]
): Float64Array[] {
  const points: Float64Array[] = [];
  for (let index = 0; index <= interiorPointCount; index++) {
    const fraction = index / interiorPointCount;
    let x = startPoint[0] + (endPoint[0] - startPoint[0]) * fraction;
    let y = startPoint[1] + (endPoint[1] - startPoint[1]) * fraction;
    for (let harmonic = 0; harmonic < parameters.length / 2; harmonic++) {
      const cosineCoefficient = parameters[2 * harmonic];
      const sineCoefficient = parameters[2 * harmonic + 1];
      const angle = (harmonic + 1) * Math.PI * fraction;
      x += cosineCoefficient * Math.cos(angle) + sineCoefficient * Math.sin(angle);
      y += cosineCoefficient * Math.sin(angle) - sineCoefficient * Math.cos(angle);
    }
    points.push(new Float64Array([x, y]));
  }
  return points;
}

function bendingResidualFromStates(states: Float64Array, problem: CurveProblem): Float64Array {
  const residual = new Float64Array(problem.interiorPointCount - 1);
  for (let index = 1; index < problem.interiorPointCount; index++) {
    const previousX = states[2 * (index - 1)];
    const previousY = states[2 * (index - 1) + 1];
    const currentX = states[2 * index];
    const currentY = states[2 * index + 1];
    const nextX = states[2 * (index + 1)];
    const nextY = states[2 * (index + 1) + 1];
    residual[index - 1] = Math.hypot(
      nextX - 2 * currentX + previousX,
      nextY - 2 * currentY + previousY
    );
  }
  return residual;
}

function arcLengthConstraintFromStates(states: Float64Array, problem: CurveProblem): Float64Array {
  const constraint = new Float64Array(problem.interiorPointCount);
  for (let index = 0; index < problem.interiorPointCount; index++) {
    const deltaX = states[2 * (index + 1)] - states[2 * index];
    const deltaY = states[2 * (index + 1) + 1] - states[2 * index + 1];
    constraint[index] = Math.hypot(deltaX, deltaY) - problem.segmentLength;
  }
  return constraint;
}

function createCurveProblem(): CurveProblem {
  const curveProblem: CurveProblem = {
    name: 'Curve bending (few parameters, many states)',
    interiorPointCount: 80,
    segmentLength: 0.01,
    parameterCount: 4,
    startPoint: [0, 0],
    endPoint: [2, 0.5],
    penaltyWeights: [1e3, 1e4],
    options: {
      constrainedLM: {
        maxIterations: 150,
        tolGradient: 1e-6,
        tolStep: 1e-8,
        constraintTolerance: 1e-4,
        lambdaInitial: 1e-2,
        lambdaFactor: 5
      },
      constrainedGN: { maxIterations: 150, tolerance: 1e-6, constraintTolerance: 1e-4 },
      penaltyLM: { maxIterations: 150, tolGradient: 1e-6, tolStep: 1e-8, lambdaInitial: 1e-2 },
      penaltyGN: { maxIterations: 150, tolerance: 1e-6 },
      adjoint: {
        maxIterations: 150,
        tolerance: 1e-4,
        constraintTolerance: 1e-4,
        useLineSearch: true,
        regularization: ADJOINT_REGULARIZATION
      }
    },
    constraint: (_parameters, states) => arcLengthConstraintFromStates(states, curveProblem)
  };
  return curveProblem;
}

const problem = createCurveProblem();

function randomInitial(
  curveProblem: CurveProblem,
  nextUniform: () => number
): { parameters: Float64Array; states: Float64Array } {
  const parameters = new Float64Array(curveProblem.parameterCount);
  for (let index = 0; index < parameters.length; index++) {
    parameters[index] = PARAMETER_NOISE_SCALE * (nextUniform() - 0.5);
  }
  const baseCurve = generateCurve(
    parameters,
    curveProblem.interiorPointCount,
    curveProblem.startPoint,
    curveProblem.endPoint
  );
  const states = new Float64Array((curveProblem.interiorPointCount + 1) * 2);
  for (let index = 0; index < baseCurve.length; index++) {
    states[2 * index] = baseCurve[index][0] + STATE_NOISE_SCALE * (nextUniform() - 0.5);
    states[2 * index + 1] = baseCurve[index][1] + STATE_NOISE_SCALE * (nextUniform() - 0.5);
  }
  return { parameters, states };
}

function createConstrainedResidual(curveProblem: CurveProblem): ConstrainedResidualFn {
  return (_parameters, states) => bendingResidualFromStates(states, curveProblem);
}

function createBaseSolvers(curveProblem: CurveProblem): Array<BenchmarkSolver<CurveProblem>> {
  const constrainedResidual = createConstrainedResidual(curveProblem);
  return [
    {
      name: 'Constrained LM',
      run: (problem, initial) =>
        constrainedLevenbergMarquardt(
          initial.parameters,
          initial.states,
          constrainedResidual,
          problem.constraint,
          problem.options.constrainedLM
        )
    },
    {
      name: 'Constrained GN',
      run: (problem, initial) =>
        constrainedGaussNewton(
          initial.parameters,
          initial.states,
          constrainedResidual,
          problem.constraint,
          problem.options.constrainedGN
        )
    },
    {
      name: 'Adjoint GD',
      run: (problem, initial) =>
        adjointGradientDescent(
          initial.parameters,
          initial.states,
          constrainedResidual,
          problem.constraint,
          problem.options.adjoint
        )
    }
  ];
}

function createPenaltySolvers(
  curveProblem: CurveProblem,
  penaltyWeight: number
): Array<BenchmarkSolver<CurveProblem>> {
  return [
    {
      name: `Penalty LM (mu=${penaltyWeight})`,
      run: (problem, initial) => {
        const penaltyResidual = buildStateOnlyPenaltyResidual({
          residual: (states) => bendingResidualFromStates(states, problem),
          constraint: (states) => arcLengthConstraintFromStates(states, problem),
          penaltyWeight
        });
        const result = levenbergMarquardt(initial.states, penaltyResidual, problem.options.penaltyLM);
        return {
          finalParameters: initial.parameters,
          finalStates: result.finalParameters,
          iterations: result.iterations,
          converged: result.converged,
          finalCost: result.finalCost
        };
      }
    },
    {
      name: `Penalty GN (mu=${penaltyWeight})`,
      run: (problem, initial) => {
        const penaltyResidual = buildStateOnlyPenaltyResidual({
          residual: (states) => bendingResidualFromStates(states, problem),
          constraint: (states) => arcLengthConstraintFromStates(states, problem),
          penaltyWeight
        });
        const result = gaussNewton(initial.states, penaltyResidual, problem.options.penaltyGN);
        return {
          finalParameters: initial.parameters,
          finalStates: result.finalParameters,
          iterations: result.iterations,
          converged: result.converged,
          finalCost: result.finalCost ?? 0
        };
      }
    }
  ];
}

function runSuite(curveProblem: CurveProblem): BenchmarkRow[] {
  const seededRandom = createSeededRandom(RANDOM_SEED);
  const initial = randomInitial(curveProblem, () => seededRandom.nextUniform());
  const rows: BenchmarkRow[] = [];

  for (const solver of createBaseSolvers(curveProblem)) {
    const row = timeConstrainedSolve({
      methodName: solver.name,
      run: () => solver.run(curveProblem, initial),
      constraintNorm: (result) =>
        vectorNorm(curveProblem.constraint(result.finalParameters, result.finalStates))
    });
    row.Penalty = '-';
    rows.push(row);
  }

  for (const penaltyWeight of curveProblem.penaltyWeights) {
    for (const solver of createPenaltySolvers(curveProblem, penaltyWeight)) {
      const row = timeConstrainedSolve({
        methodName: solver.name,
        run: () => solver.run(curveProblem, initial),
        constraintNorm: (result) =>
          vectorNorm(arcLengthConstraintFromStates(result.finalStates, curveProblem))
      });
      row.Penalty = penaltyWeight;
      rows.push(row);
    }
  }

  return rows;
}

console.log('Benchmark: curve bending with arc-length constraints (few parameters, many states)');
console.log(
  `\n=== ${problem.name} (n=${problem.interiorPointCount}, params=${problem.parameterCount}) ===`
);
console.table(runSuite(problem));
