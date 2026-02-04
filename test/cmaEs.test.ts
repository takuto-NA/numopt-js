import { cmaEs } from '../src/core/cmaEs';
import type { CostFn } from '../src/core/types';

describe('CMA-ES', () => {
  const sphereCost: CostFn = (parameters: Float64Array) => {
    let sum = 0.0;
    for (let index = 0; index < parameters.length; index++) {
      const value = parameters[index];
      sum += value * value;
    }
    return sum;
  };

  it('should be deterministic with the same seed', () => {
    const initialParameters = new Float64Array([10.0, -7.0, 3.0]);
    const options = {
      maxIterations: 120,
      initialStepSize: 2.0,
      populationSize: 20,
      randomSeed: 123456,
      functionTolerance: 1e-14,
      parameterTolerance: 1e-12
    };

    const resultA = cmaEs(initialParameters, sphereCost, options);
    const resultB = cmaEs(initialParameters, sphereCost, options);

    expect(resultA.finalCost).toBe(resultB.finalCost);
    expect(Array.from(resultA.finalParameters)).toEqual(Array.from(resultB.finalParameters));
    expect(resultA.functionEvaluations).toBe(resultB.functionEvaluations);
  });

  it('should reduce Sphere function value substantially', () => {
    const initialParameters = new Float64Array([10.0, 10.0, 10.0, 10.0, 10.0]);
    const result = cmaEs(initialParameters, sphereCost, {
      maxIterations: 200,
      initialStepSize: 3.0,
      randomSeed: 42,
      populationSize: 30,
      targetCost: 1e-8
    });

    expect(result.finalCost).toBeLessThan(1e-4);
    expect(result.finalStepSize).toBeGreaterThan(0.0);
  });

  it('should stop when maxFunctionEvaluations is reached', () => {
    const initialParameters = new Float64Array([5.0, -5.0]);
    const result = cmaEs(initialParameters, sphereCost, {
      maxIterations: 1000,
      initialStepSize: 1.0,
      randomSeed: 7,
      populationSize: 10,
      maxFunctionEvaluations: 15
    });

    expect(result.converged).toBe(false);
    expect(result.functionEvaluations).toBeGreaterThanOrEqual(15);
  });
});

