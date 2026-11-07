/** End-to-end accuracy across objective units, with gradient tolerances scaled to the same physical error. */

import { describe, expect, it } from 'vitest';
import { bfgs, lbfgs } from '../src/index';

const OBJECTIVE_SCALES = [1e-6, 1, 1e6];
const EXPECTED_PARAMETER = 1.75;
const INITIAL_PARAMETER = -0.3;
const BASE_GRADIENT_TOLERANCE = 1e-9;
const MAXIMUM_ITERATIONS = 100;
const PARAMETER_TOLERANCE = 1e-8;

describe.each([{ name: 'BFGS', solver: bfgs }, { name: 'L-BFGS', solver: lbfgs }])('$name objective scaling', ({ solver }) => {
  it.each(OBJECTIVE_SCALES)('solves the same quadratic at cost scale %g', scale => {
    const result = solver(new Float64Array([INITIAL_PARAMETER]),
      parameters => scale * (parameters[0] - EXPECTED_PARAMETER) ** 2 / 2,
      parameters => new Float64Array([scale * (parameters[0] - EXPECTED_PARAMETER)]),
      { tolerance: BASE_GRADIENT_TOLERANCE * scale, maxIterations: MAXIMUM_ITERATIONS });
    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - EXPECTED_PARAMETER)).toBeLessThan(PARAMETER_TOLERANCE);
  });
});
