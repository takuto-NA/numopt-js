/** Run the public synchronous LM with an explicitly prepared kibo WASM solver. */
import { createKiboLinearSolver, levenbergMarquardt } from '../dist/index.js';
import createModule from '../.cache/kibo/kibo.mjs';
import { LinearAlgebra } from '../.cache/kibo/facade.mjs';

const inputs = [1, 2, 3, 4, 5];
const observations = [2.1, 3.9, 6.1, 8, 9.9];
const module = await createModule();
const linearSolver = createKiboLinearSolver(new LinearAlgebra(module, 2, 2));
try {
  const result = levenbergMarquardt(new Float64Array([0, 0]), parameters =>
    Float64Array.from(inputs, (value, index) => parameters[0] * value + parameters[1] - observations[index]),
    { linearSolver, maxIterations: 100, tolGradient: 1e-6 });
  console.log({ parameters: Array.from(result.finalParameters), cost: result.finalCost,
    converged: result.converged, statistics: linearSolver.statistics });
} finally { linearSolver.dispose(); }
