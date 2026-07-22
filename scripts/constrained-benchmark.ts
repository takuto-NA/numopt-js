/**
 * Manual benchmark suite: classic constrained problems across Adjoint / Constrained GN/LM / Penalty.
 * Not run in CI. Invoke via `npm run benchmark:constrained`.
 */

import {
  adjointGradientDescent,
  constrainedGaussNewton,
  constrainedLevenbergMarquardt,
  levenbergMarquardt,
  gaussNewton
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
import {
  buildPenaltyResidual,
  concatParameterAndState,
  runSolverTable,
  splitParameterAndState,
  type BenchmarkSolver
} from './benchmark-harness';

const HIGH_DIMENSIONAL_VARIABLE_COUNT = 30;

type Problem = {
  name: string;
  residual: ConstrainedResidualFn;
  constraint: ConstraintFn;
  buildInitial: () => { parameters: Float64Array; states: Float64Array };
  penaltyWeight: number;
  options: {
    adjoint: AdjointGradientDescentOptions;
    gaussNewton: ConstrainedGaussNewtonOptions;
    levenbergMarquardt: ConstrainedLevenbergMarquardtOptions;
    penaltyGaussNewton: GaussNewtonOptions;
    penaltyLevenbergMarquardt: LevenbergMarquardtOptions;
  };
};

const rosenbrockResidual: ConstrainedResidualFn = (parameters, states) => {
  const a = 1.0 - parameters[0];
  const b = states[0] - parameters[0] * parameters[0];
  return new Float64Array([a, 10.0 * b]);
};

const circleConstraint: ConstraintFn = (parameters, states) => {
  return new Float64Array([parameters[0] * parameters[0] + states[0] * states[0] - 2.0]);
};

const illConditionedResidual: ConstrainedResidualFn = (parameters, states) => {
  return new Float64Array([parameters[0] / 1000.0, 1000.0 * states[0]]);
};

const simpleConstraint: ConstraintFn = (parameters, states) => {
  return new Float64Array([parameters[0] + states[0] - 1.0]);
};

const highDimResidual: ConstrainedResidualFn = (parameters, states) => {
  const dimension = parameters.length;
  const residual = new Float64Array(2 * dimension);
  for (let index = 0; index < dimension; index++) {
    const target = index + 1;
    residual[index] = parameters[index] - target;
    residual[dimension + index] = states[index] - target;
  }
  return residual;
};

const highDimConstraint: ConstraintFn = (parameters, states) => {
  const constraint = new Float64Array(parameters.length);
  for (let index = 0; index < parameters.length; index++) {
    constraint[index] = parameters[index] + states[index] - 2 * (index + 1);
  }
  return constraint;
};

const problems: Problem[] = [
  {
    name: 'Rosenbrock valley with circle constraint',
    residual: rosenbrockResidual,
    constraint: circleConstraint,
    buildInitial: () => ({
      parameters: new Float64Array([1.5]),
      states: new Float64Array([-1.0])
    }),
    penaltyWeight: 1e4,
    options: {
      adjoint: {
        maxIterations: 100,
        tolerance: 1e-8,
        constraintTolerance: 1e-8,
        useLineSearch: true,
        logLevel: 'WARN'
      },
      gaussNewton: { maxIterations: 100, tolerance: 1e-8, constraintTolerance: 1e-8 },
      levenbergMarquardt: {
        maxIterations: 100,
        tolGradient: 1e-8,
        tolStep: 1e-10,
        constraintTolerance: 1e-8,
        lambdaInitial: 1e-3
      },
      penaltyGaussNewton: { maxIterations: 100, tolerance: 1e-10 },
      penaltyLevenbergMarquardt: {
        maxIterations: 100,
        tolGradient: 1e-10,
        tolStep: 1e-12,
        lambdaInitial: 1e-3
      }
    }
  },
  {
    name: 'Ill-conditioned single variable',
    residual: illConditionedResidual,
    constraint: simpleConstraint,
    buildInitial: () => ({
      parameters: new Float64Array([5000.0]),
      states: new Float64Array([-4999.0])
    }),
    penaltyWeight: 1e8,
    options: {
      adjoint: {
        maxIterations: 100,
        tolerance: 1e-6,
        constraintTolerance: 1e-8,
        useLineSearch: true,
        logLevel: 'WARN'
      },
      gaussNewton: { maxIterations: 100, tolerance: 1e-6, constraintTolerance: 1e-8 },
      levenbergMarquardt: {
        maxIterations: 100,
        tolGradient: 1e-6,
        tolStep: 1e-8,
        constraintTolerance: 1e-8,
        lambdaInitial: 1e-2
      },
      penaltyGaussNewton: { maxIterations: 100, tolerance: 1e-10 },
      penaltyLevenbergMarquardt: {
        maxIterations: 100,
        tolGradient: 1e-10,
        tolStep: 1e-12,
        lambdaInitial: 1e-2
      }
    }
  },
  {
    name: `High-dimensional ${HIGH_DIMENSIONAL_VARIABLE_COUNT}D affine constraint`,
    residual: highDimResidual,
    constraint: highDimConstraint,
    buildInitial: () => {
      const parameters = new Float64Array(HIGH_DIMENSIONAL_VARIABLE_COUNT).fill(10.0);
      const states = new Float64Array(HIGH_DIMENSIONAL_VARIABLE_COUNT).map(
        (_, index) => 2 * (index + 1) - 10.0
      );
      return { parameters, states };
    },
    penaltyWeight: 1e5,
    options: {
      adjoint: {
        maxIterations: 100,
        tolerance: 1e-4,
        constraintTolerance: 1e-8,
        useLineSearch: true,
        logLevel: 'WARN'
      },
      gaussNewton: { maxIterations: 100, tolerance: 1e-4, constraintTolerance: 1e-8 },
      levenbergMarquardt: {
        maxIterations: 100,
        tolGradient: 1e-4,
        tolStep: 1e-8,
        constraintTolerance: 1e-8,
        lambdaInitial: 1e-3
      },
      penaltyGaussNewton: { maxIterations: 100, tolerance: 1e-10 },
      penaltyLevenbergMarquardt: {
        maxIterations: 100,
        tolGradient: 1e-10,
        tolStep: 1e-12,
        lambdaInitial: 1e-3
      }
    }
  },
  {
    name: 'State-heavy linear trajectory fit (p ≪ x)',
    residual: (_parameters, states) => {
      const residual = new Float64Array(states.length);
      for (let index = 0; index < states.length; index++) {
        const target =
          Math.sin((2 * Math.PI * index) / states.length) +
          0.1 * Math.sin((6 * Math.PI * index) / states.length);
        residual[index] = states[index] - target;
      }
      return residual;
    },
    constraint: (parameters, states) => {
      const constraint = new Float64Array(states.length);
      for (let index = 0; index < states.length; index++) {
        constraint[index] =
          states[index] -
          (parameters[0] +
            parameters[1] * index +
            parameters[2] * Math.sin((2 * Math.PI * index) / states.length));
      }
      return constraint;
    },
    buildInitial: () => ({
      parameters: new Float64Array([0.1, 0.01, 0.0]),
      states: new Float64Array(200).fill(0)
    }),
    penaltyWeight: 1e5,
    options: {
      adjoint: {
        maxIterations: 100,
        tolerance: 1e-5,
        constraintTolerance: 1e-8,
        useLineSearch: true,
        logLevel: 'WARN'
      },
      gaussNewton: { maxIterations: 100, tolerance: 1e-5, constraintTolerance: 1e-8 },
      levenbergMarquardt: {
        maxIterations: 100,
        tolGradient: 1e-6,
        tolStep: 1e-8,
        constraintTolerance: 1e-8,
        lambdaInitial: 1e-3
      },
      penaltyGaussNewton: { maxIterations: 100, tolerance: 1e-10 },
      penaltyLevenbergMarquardt: {
        maxIterations: 100,
        tolGradient: 1e-10,
        tolStep: 1e-12,
        lambdaInitial: 1e-3
      }
    }
  }
];

function createSolvers(): Array<BenchmarkSolver<Problem>> {
  return [
    {
      name: 'Adjoint GD',
      run: (problem, initial) =>
        adjointGradientDescent(
          initial.parameters,
          initial.states,
          problem.residual,
          problem.constraint,
          problem.options.adjoint
        )
    },
    {
      name: 'Constrained GN',
      run: (problem, initial) =>
        constrainedGaussNewton(
          initial.parameters,
          initial.states,
          problem.residual,
          problem.constraint,
          problem.options.gaussNewton
        )
    },
    {
      name: 'Constrained LM',
      run: (problem, initial) =>
        constrainedLevenbergMarquardt(
          initial.parameters,
          initial.states,
          problem.residual,
          problem.constraint,
          problem.options.levenbergMarquardt
        )
    },
    {
      name: 'Penalty GN',
      run: (problem, initial) => {
        const parameterCount = initial.parameters.length;
        const penaltyResidual = buildPenaltyResidual({
          parameterCount,
          residual: problem.residual,
          constraint: problem.constraint,
          penaltyWeight: problem.penaltyWeight
        });
        const result = gaussNewton(
          concatParameterAndState(initial.parameters, initial.states),
          penaltyResidual,
          problem.options.penaltyGaussNewton
        );
        const split = splitParameterAndState(result.finalParameters, parameterCount);
        return {
          finalParameters: split.parameters,
          finalStates: split.states,
          iterations: result.iterations,
          converged: result.converged,
          finalCost: result.finalCost ?? 0
        };
      }
    },
    {
      name: 'Penalty LM',
      run: (problem, initial) => {
        const parameterCount = initial.parameters.length;
        const penaltyResidual = buildPenaltyResidual({
          parameterCount,
          residual: problem.residual,
          constraint: problem.constraint,
          penaltyWeight: problem.penaltyWeight
        });
        const result = levenbergMarquardt(
          concatParameterAndState(initial.parameters, initial.states),
          penaltyResidual,
          problem.options.penaltyLevenbergMarquardt
        );
        const split = splitParameterAndState(result.finalParameters, parameterCount);
        return {
          finalParameters: split.parameters,
          finalStates: split.states,
          iterations: result.iterations,
          converged: result.converged,
          finalCost: result.finalCost
        };
      }
    }
  ];
}

const solvers = createSolvers();

console.log(
  'Benchmark: constrained optimizers (lower time/iterations is better while keeping constraints satisfied)'
);
for (const problem of problems) {
  console.log(`\n=== ${problem.name} ===`);
  console.table(
    runSolverTable({
      problem,
      initial: problem.buildInitial(),
      solvers
    })
  );
}
