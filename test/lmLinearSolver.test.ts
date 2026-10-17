/** Public LM solver selection and linear-system ownership contracts. */
import { Matrix } from 'ml-matrix';
import { createMlMatrixLinearSolver, levenbergMarquardt } from '../src/index';

const PARAMETER_TOLERANCE = 1e-3;

describe('LM linear solver selection', () => {
  it('runs the public synchronous LM with an explicitly selected ml-matrix solver', () => {
    const linearSolver = createMlMatrixLinearSolver();
    const result = levenbergMarquardt(
      new Float64Array([0]),
      parameters => new Float64Array([parameters[0] - 2]),
      { jacobian: () => new Matrix([[1]]), maxIterations: 10, linearSolver }
    );

    expect(result.converged).toBe(true);
    expect(result.finalParameters[0]).toBeCloseTo(2, -Math.log10(PARAMETER_TOLERANCE));
    expect(linearSolver.statistics.solveSuccesses).toBeGreaterThan(0);
    expect(linearSolver.statistics.fallbackAttempts).toBe(0);
  });

  it('reports nonconvergence when numerical solves keep failing', () => {
    const result = levenbergMarquardt(new Float64Array([0]),
      parameters => new Float64Array([parameters[0] - 2]), {
        jacobian: () => new Matrix([[1]]), maxIterations: 2,
        linearSolver: { solve() { throw new Error('numerical solve failed'); } }
      });
    expect(result.converged).toBe(false);
    expect(result.finalParameters[0]).toBe(0);
    expect(result.finalLambda).toBeGreaterThan(1e-3);
  });
});
