/**
 * Analytical objectives for Newton-adjoint comparisons, including omitted-curvature counterexamples.
 * Shared by the benchmark, runnable example, and solver scenario tests.
 */

import { Matrix } from 'ml-matrix';
import type { AdjointNewtonOptions, AdjointNewtonExactOptions, ConstrainedCostFn, ConstrainedResidualFn, ConstraintFn } from '../src/core/types';

export type NewtonAdjointProblem = {
  name: string;
  parameters: Float64Array;
  states: Float64Array;
  expected: Float64Array;
  cost: ConstrainedCostFn;
  constraint: ConstraintFn;
  derivatives: AdjointNewtonOptions;
  exactDerivatives: AdjointNewtonExactOptions;
  reducedGradient: (parameters: Float64Array) => Float64Array;
  residual?: ConstrainedResidualFn;
};

function createAffineQuadratic(): NewtonAdjointProblem {
  const weights = [1, 100];
  const stateWeights = [9, 900];
  const target = [1, -2];
  const residual: ConstrainedResidualFn = (parameters, states) => new Float64Array([
    ...weights.map((weight, index) => Math.sqrt(weight) * (parameters[index] - target[index])),
    ...stateWeights.map((weight, index) => Math.sqrt(weight) * (states[index] - target[index]))
  ]);
  return {
    name: 'Affine quadratic (exact model)', parameters: new Float64Array([3, 1]), states: new Float64Array([3, 1]),
    expected: new Float64Array(target), residual,
    reducedGradient: parameters => Float64Array.from(parameters,
      (value, index) => (weights[index] + stateWeights[index]) * (value - target[index])),
    cost: (parameters, states) => residual(parameters, states).reduce((sum, value) => sum + value * value, 0) / 2,
    constraint: (parameters, states) => Float64Array.from(states, (value, index) => value - parameters[index]),
    derivatives: {
      dfdp: parameters => Float64Array.from(parameters, (value, index) => weights[index] * (value - target[index])),
      dfdx: (_parameters, states) => Float64Array.from(states, (value, index) => stateWeights[index] * (value - target[index])),
      dcdp: () => Matrix.eye(target.length).mul(-1), dcdx: () => Matrix.eye(target.length),
      d2fdp2: () => Matrix.diag(weights), d2fdx2: () => Matrix.diag(stateWeights)
    },
    exactDerivatives: { lagrangianHessian: () => Matrix.diag([...weights, ...stateWeights]) }
  };
}

export function createChainProblem(stateCount: number = 20): NewtonAdjointProblem {
  const initialParameter = 0.2;
  const residual: ConstrainedResidualFn = (_parameters, states) => new Float64Array([
    Math.SQRT2 * (states[stateCount - 1] - 1)
  ]);
  return {
    name: `Implicit chain (${stateCount} states)`, parameters: new Float64Array([initialParameter]),
    states: Float64Array.from({ length: stateCount }, (_, index) => (index + 1) * initialParameter),
    expected: new Float64Array([1 / stateCount]), residual,
    reducedGradient: parameters => new Float64Array([2 * stateCount * (stateCount * parameters[0] - 1)]),
    cost: (_parameters, states) => (states[stateCount - 1] - 1) ** 2,
    constraint: (parameters, states) => Float64Array.from(states,
      (value, index) => value - (index > 0 ? states[index - 1] : 0) - parameters[0]),
    derivatives: {
      dfdp: () => new Float64Array([0]),
      dfdx: (_parameters, states) => {
        const gradient = new Float64Array(stateCount);
        gradient[stateCount - 1] = 2 * (states[stateCount - 1] - 1);
        return gradient;
      },
      dcdp: () => Matrix.ones(stateCount, 1).mul(-1),
      dcdx: () => {
        const jacobian = Matrix.eye(stateCount);
        for (let index = 1; index < stateCount; index++) jacobian.set(index, index - 1, -1);
        return jacobian;
      },
      d2fdp2: () => Matrix.zeros(1, 1),
      d2fdx2: () => {
        const hessian = Matrix.zeros(stateCount, stateCount);
        hessian.set(stateCount - 1, stateCount - 1, 2);
        return hessian;
      }
    },
    exactDerivatives: {
      lagrangianHessian: () => {
        const hessian = Matrix.zeros(stateCount + 1, stateCount + 1);
        hessian.set(stateCount, stateCount, 2);
        return hessian;
      },
      lagrangianHessianVector: (_parameters, _states, _lambda, direction) => {
        const product = new Float64Array(stateCount + 1);
        product[stateCount] = 2 * direction[stateCount];
        return product;
      }
    }
  };
}

function createCircle(): NewtonAdjointProblem {
  const initialParameter = 0.2;
  const residual: ConstrainedResidualFn = (parameters, states) => new Float64Array([
    Math.SQRT2 * (parameters[0] - 1), Math.SQRT2 * (states[0] - 1)
  ]);
  return {
    name: 'Circle (vanishing omitted curvature)', parameters: new Float64Array([initialParameter]),
    states: new Float64Array([Math.sqrt(2 - initialParameter ** 2)]), expected: new Float64Array([1]), residual,
    reducedGradient: parameters => {
      const state = Math.sqrt(2 - parameters[0] ** 2);
      return new Float64Array([2 * (parameters[0] - 1) - 2 * (state - 1) * parameters[0] / state]);
    },
    cost: (parameters, states) => (parameters[0] - 1) ** 2 + (states[0] - 1) ** 2,
    constraint: (parameters, states) => new Float64Array([parameters[0] ** 2 + states[0] ** 2 - 2]),
    derivatives: {
      dfdp: parameters => new Float64Array([2 * (parameters[0] - 1)]),
      dfdx: (_parameters, states) => new Float64Array([2 * (states[0] - 1)]),
      dcdp: parameters => new Matrix([[2 * parameters[0]]]),
      dcdx: (_parameters, states) => new Matrix([[2 * states[0]]]),
      d2fdp2: () => new Matrix([[2]]), d2fdx2: () => new Matrix([[2]])
    },
    exactDerivatives: { lagrangianHessian: (_parameters, _states, lambda) => Matrix.eye(2).mul(2 - 2 * lambda[0]) }
  };
}

function createMixedPartialCounterexample(): NewtonAdjointProblem {
  const residual: ConstrainedResidualFn = (parameters, states) => new Float64Array([
    parameters[0] - states[0], parameters[0] - 1
  ]);
  return {
    name: 'Mixed partials omitted (linear convergence)', parameters: new Float64Array([0]), states: new Float64Array([0]),
    expected: new Float64Array([1]), residual,
    reducedGradient: parameters => new Float64Array([parameters[0] - 1]),
    cost: (parameters, states) => ((parameters[0] - states[0]) ** 2 + (parameters[0] - 1) ** 2) / 2,
    constraint: (parameters, states) => new Float64Array([states[0] - parameters[0]]),
    derivatives: {
      dfdp: (parameters, states) => new Float64Array([2 * parameters[0] - states[0] - 1]),
      dfdx: (parameters, states) => new Float64Array([states[0] - parameters[0]]),
      dcdp: () => new Matrix([[-1]]), dcdx: () => new Matrix([[1]]),
      d2fdp2: () => new Matrix([[2]]), d2fdx2: () => new Matrix([[1]])
    },
    exactDerivatives: { lagrangianHessian: () => new Matrix([[2, -1], [-1, 1]]) }
  };
}

function createStateCurvatureCounterexample(): NewtonAdjointProblem {
  return {
    name: 'State curvature omitted (backtracking needed)', parameters: new Float64Array([1]), states: new Float64Array([1]),
    expected: new Float64Array([1 / 3]),
    reducedGradient: parameters => new Float64Array([3 * parameters[0] - 1]),
    cost: (parameters, states) => (parameters[0] - 1) ** 2 / 2 + states[0],
    constraint: (parameters, states) => new Float64Array([states[0] - parameters[0] ** 2]),
    derivatives: {
      dfdp: parameters => new Float64Array([parameters[0] - 1]), dfdx: () => new Float64Array([1]),
      dcdp: parameters => new Matrix([[-2 * parameters[0]]]), dcdx: () => new Matrix([[1]]),
      d2fdp2: () => new Matrix([[1]]), d2fdx2: () => new Matrix([[0]])
    },
    exactDerivatives: { lagrangianHessian: (_parameters, _states, lambda) => new Matrix([[1 + 2 * lambda[0], 0], [0, 0]]) }
  };
}

export function createNewtonAdjointProblems(): NewtonAdjointProblem[] {
  return [createAffineQuadratic(), createChainProblem(), createCircle(),
    createMixedPartialCounterexample(), createStateCurvatureCounterexample()];
}

export function createQuadraticConvergenceProblem(): NewtonAdjointProblem {
  const curvature = (parameters: Float64Array): number => {
    const error = parameters[0] - 1;
    return 3 * error * error + 2 * error + 1;
  };
  return {
    name: 'Nonlinear quadratic convergence (nonzero adjoint)',
    parameters: new Float64Array([1.5]), states: new Float64Array([2.25]), expected: new Float64Array([1]),
    reducedGradient: parameters => {
      const error = parameters[0] - 1;
      return new Float64Array([error ** 3 + error ** 2 + 3 * error]);
    },
    cost: (parameters, states) => {
      const error = parameters[0] - 1;
      return error ** 4 / 4 + error ** 3 / 3 + (parameters[0] - 3) ** 2 / 2 + states[0];
    },
    constraint: (parameters, states) => new Float64Array([states[0] - parameters[0] ** 2]),
    derivatives: {
      dfdp: parameters => {
        const error = parameters[0] - 1;
        return new Float64Array([error ** 3 + error ** 2 + parameters[0] - 3]);
      },
      dfdx: () => new Float64Array([1]),
      dcdp: parameters => new Matrix([[-2 * parameters[0]]]), dcdx: () => new Matrix([[1]]),
      d2fdp2: parameters => new Matrix([[curvature(parameters)]]), d2fdx2: () => new Matrix([[0]])
    },
    exactDerivatives: {
      lagrangianHessian: (parameters, _states, lambda) => new Matrix([[curvature(parameters) + 2 * lambda[0], 0], [0, 0]])
    }
  };
}

export function createDesignScalingProblem(parameterCount: number): NewtonAdjointProblem {
  const target = Float64Array.from({ length: parameterCount }, (_, index) => (index + 1) / parameterCount);
  const weights = Float64Array.from({ length: parameterCount }, (_, index) => 1 + 9 * index / parameterCount);
  const hessianVector: NonNullable<AdjointNewtonExactOptions['lagrangianHessianVector']> =
    (_parameters, _states, _lambda, direction) => {
      const result = new Float64Array(2 * parameterCount);
      for (let index = 0; index < parameterCount; index++) {
        result[index] = (weights[index] + 1) * direction[index] - direction[index + parameterCount];
        result[index + parameterCount] = direction[index + parameterCount] - direction[index];
      }
      return result;
    };
  return {
    name: `Mixed quadratic (${parameterCount} parameters)`, parameters: new Float64Array(parameterCount),
    states: new Float64Array(parameterCount), expected: target,
    reducedGradient: parameters => Float64Array.from(parameters, (value, index) => weights[index] * (value - target[index])),
    cost: (parameters, states) => parameters.reduce((sum, value, index) =>
      sum + ((value - states[index]) ** 2 + weights[index] * (value - target[index]) ** 2) / 2, 0),
    constraint: (parameters, states) => Float64Array.from(states, (value, index) => value - parameters[index]),
    derivatives: {
      dfdp: (parameters, states) => Float64Array.from(parameters,
        (value, index) => value - states[index] + weights[index] * (value - target[index])),
      dfdx: (parameters, states) => Float64Array.from(states, (value, index) => value - parameters[index]),
      dcdp: () => Matrix.eye(parameterCount).mul(-1), dcdx: () => Matrix.eye(parameterCount),
      d2fdp2: () => Matrix.diag(Array.from(weights, value => value + 1)), d2fdx2: () => Matrix.eye(parameterCount)
    },
    exactDerivatives: {
      lagrangianHessianVector: hessianVector,
      lagrangianHessian: (parameters, states, lambda) => {
        const hessian = Matrix.zeros(2 * parameterCount, 2 * parameterCount);
        for (let column = 0; column < hessian.columns; column++) {
          const direction = new Float64Array(hessian.columns);
          direction[column] = 1;
          hessian.setColumn(column, Array.from(hessianVector(parameters, states, lambda, direction)));
        }
        return hessian;
      }
    }
  };
}
