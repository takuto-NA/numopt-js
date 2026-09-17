import { strongWolfeLineSearch } from '../src/core/lineSearch';
import type { CostFn, GradientFn } from '../src/core/types';

describe('Strong Wolfe Line Search', () => {
  /**
   * Quadratic function: f(x) = x^2
   * Minimum at x = 0
   * Gradient: f'(x) = 2x
   */
  const quadraticCost: CostFn = (parameters: Float64Array) => {
    return parameters[0] * parameters[0];
  };

  const quadraticGradient: GradientFn = (parameters: Float64Array) => {
    return new Float64Array([2 * parameters[0]]);
  };

  it('should return 0 for non-descent direction', () => {
    const currentParameters = new Float64Array([5.0]);
    const nonDescentDirection = new Float64Array([1.0]);

    const stepSize = strongWolfeLineSearch(
      quadraticCost,
      quadraticGradient,
      currentParameters,
      nonDescentDirection
    );

    expect(stepSize).toBe(0.0);
  });

  it('should satisfy Armijo and Strong Wolfe curvature conditions', () => {
    const currentParameters = new Float64Array([3.0]);
    const searchDirection = new Float64Array([-1.0]);

    const wolfeC1 = 1e-4;
    const wolfeC2 = 0.9;

    const stepSize = strongWolfeLineSearch(
      quadraticCost,
      quadraticGradient,
      currentParameters,
      searchDirection,
      { wolfeC1, wolfeC2 }
    );

    expect(stepSize).toBeGreaterThan(0);

    const currentCost = quadraticCost(currentParameters);
    const currentGradient = quadraticGradient(currentParameters);
    const directionalDerivativeAtZero = currentGradient[0] * searchDirection[0];

    const newParameters = new Float64Array([currentParameters[0] + stepSize * searchDirection[0]]);
    const newCost = quadraticCost(newParameters);
    const newGradient = quadraticGradient(newParameters);
    const newDirectionalDerivative = newGradient[0] * searchDirection[0];

    const armijoThreshold = currentCost + wolfeC1 * stepSize * directionalDerivativeAtZero;
    expect(newCost).toBeLessThanOrEqual(armijoThreshold);

    const curvatureLeftSide = Math.abs(newDirectionalDerivative);
    const curvatureRightSide = wolfeC2 * Math.abs(directionalDerivativeAtZero);
    expect(curvatureLeftSide).toBeLessThanOrEqual(curvatureRightSide);
  });

  it('should work for simple 2D quadratic', () => {
    const quadratic2DCost: CostFn = (parameters: Float64Array) => {
      return parameters[0] * parameters[0] + parameters[1] * parameters[1];
    };

    const quadratic2DGradient: GradientFn = (parameters: Float64Array) => {
      return new Float64Array([2 * parameters[0], 2 * parameters[1]]);
    };

    const currentParameters = new Float64Array([3.0, 4.0]);
    const searchDirection = new Float64Array([-1.0, -1.0]);

    const stepSize = strongWolfeLineSearch(
      quadratic2DCost,
      quadratic2DGradient,
      currentParameters,
      searchDirection
    );

    expect(stepSize).toBeGreaterThan(0);

    const newParameters = new Float64Array([
      currentParameters[0] + stepSize * searchDirection[0],
      currentParameters[1] + stepSize * searchDirection[1]
    ]);
    expect(quadratic2DCost(newParameters)).toBeLessThan(quadratic2DCost(currentParameters));
  });

  it('does not evaluate the gradient where the reduced cost is undefined', () => {
    const domainLowerBound = 0.0;
    const currentParameters = new Float64Array([0.5]);
    const searchDirection = new Float64Array([-1.0]);
    const gradientEvaluationLocations: number[] = [];

    const domainLimitedCost: CostFn = (parameters) => {
      if (parameters[0] < domainLowerBound) {
        return Number.POSITIVE_INFINITY;
      }
      return parameters[0] * parameters[0];
    };

    const domainLimitedGradient: GradientFn = (parameters) => {
      gradientEvaluationLocations.push(parameters[0]);
      if (parameters[0] < domainLowerBound) {
        throw new Error(`Gradient evaluated outside the reduced-space domain at ${parameters[0]}`);
      }
      return new Float64Array([2 * parameters[0]]);
    };

    const stepSize = strongWolfeLineSearch(
      domainLimitedCost,
      domainLimitedGradient,
      currentParameters,
      searchDirection
    );

    expect(stepSize).toBeGreaterThan(0);
    expect(currentParameters[0] + stepSize * searchDirection[0]).toBeGreaterThanOrEqual(domainLowerBound);
    expect(gradientEvaluationLocations.every((location) => location >= domainLowerBound)).toBe(true);
  });

  it('returns zero when every positive trial step is undefined', () => {
    const currentParameters = new Float64Array([1.0]);
    const searchDirection = new Float64Array([-1.0]);
    let gradientCallsAwayFromStart = 0;

    const onlyDefinedAtStart: CostFn = (parameters) => {
      if (Math.abs(parameters[0] - currentParameters[0]) > 1e-15) {
        return Number.POSITIVE_INFINITY;
      }
      return parameters[0] * parameters[0];
    };

    const onlyDefinedAtStartGradient: GradientFn = (parameters) => {
      if (Math.abs(parameters[0] - currentParameters[0]) > 1e-15) {
        gradientCallsAwayFromStart += 1;
        throw new Error('Gradient evaluated where the reduced cost is undefined');
      }
      return new Float64Array([2 * parameters[0]]);
    };

    const stepSize = strongWolfeLineSearch(
      onlyDefinedAtStart,
      onlyDefinedAtStartGradient,
      currentParameters,
      searchDirection
    );

    expect(stepSize).toBe(0.0);
    expect(gradientCallsAwayFromStart).toBe(0);
  });
});

