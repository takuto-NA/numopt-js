/**
 * Scaling fixtures and measurement comparisons: independent gradients, complete curvature and uncertainty labels.
 */

import { Matrix, solve } from 'ml-matrix';
import { createScalingProblem } from '../scripts/scaling/problems.mjs';
import { compareTiming, summarize } from '../scripts/scaling/statistics.mjs';

describe('scaling benchmark problems', () => {
  for (const family of ['easy', 'curved', 'heterogeneous']) {
    it(`has feasible implicit states and the declared reduced gradient: ${family}`, () => {
      const problem = createScalingProblem({ parameters: 3, states: 7, family });
      const parameters = problem.initialParameters;
      const states = problem.initialStates;
      expect(Array.from(problem.constraint(parameters, states))).toEqual(new Array(states.length).fill(0));
      const constraintState = problem.firstDerivatives.dcdx(parameters, states);
      const constraintParameter = problem.firstDerivatives.dcdp(parameters, states);
      const lambda = solve(constraintState.transpose(), Matrix.columnVector(Array.from(problem.firstDerivatives.dfdx(parameters, states))));
      const gradient = Matrix.columnVector(Array.from(problem.firstDerivatives.dfdp(parameters, states)))
        .sub(constraintParameter.transpose().mmul(lambda));
      const reference = problem.reducedGradient(parameters);
      reference.forEach((value, index) => expect(gradient.get(index, 0)).toBeCloseTo(value, 12));
      const sensitivities = solve(constraintState, constraintParameter.clone().mul(-1));
      const basis = Matrix.zeros(parameters.length + states.length, parameters.length);
      for (let column = 0; column < parameters.length; column++) {
        basis.set(column, column, 1);
        for (let row = 0; row < states.length; row++) basis.set(parameters.length + row, column, sensitivities.get(row, column));
      }
      const multiplier = new Float64Array(lambda.getColumn(0));
      const hessian = problem.lagrangianHessian(parameters, states, multiplier);
      const projected = basis.transpose().mmul(hessian).mmul(basis);
      for (let row = 0; row < parameters.length; row++) {
        const error = parameters[row] - 1;
        const expectedCurvature = family === 'easy' ? 1 : 3 * error * error + 2 * error + 3;
        for (let column = 0; column < parameters.length; column++) expect(projected.get(row, column))
          .toBeCloseTo(row === column ? expectedCurvature : 0, 12);
      }
      const direction = new Float64Array(hessian.columns).fill(0.3);
      const product = problem.lagrangianHessianVector(parameters, states, multiplier, direction);
      const expectedProduct = hessian.mmul(Matrix.columnVector(Array.from(direction))).getColumn(0);
      product.forEach((value, index) => expect(value).toBeCloseTo(expectedProduct[index], 12));
    });
  }

  it('supports more parameters than states', () => {
    const problem = createScalingProblem({ parameters: 8, states: 3, family: 'curved' });
    expect(problem.firstDerivatives.dcdp(problem.initialParameters).columns).toBe(8);
    expect(problem.reducedGradient(problem.initialParameters).length).toBe(8);
  });

  it('reports overlapping timings instead of inventing a crossover', () => {
    expect(summarize([2, 4, 6]).median).toBe(4);
    expect(compareTiming(summarize([3, 4, 5]), summarize([2, 4, 6])).classification).toBe('overlapping');
    expect(compareTiming(summarize([8, 9, 10]), summarize([2, 3, 4])).classification).toBe('slower');
    expect(compareTiming(undefined, summarize([1, 2, 3])).classification).toBe('unavailable');
  });
});
